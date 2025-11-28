import torch
from typing import Sequence, Union, Iterable

class ActivationSteerer:
    """
    Add (coeff * steering_vector) to a chosen transformer block's output.
    Now handles blocks that return tuples and fails loudly if it can't
    locate a layer list.
    """

    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "model.layers",        # Llama/Mistral
    )

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vector: Union[torch.Tensor, Sequence[float]], # persona vector
        *, # Bắt buộc các tham số phía sau phải được truyền theo tên
        coeff: float = 1.0, # Hệ số nhân với steering_vector
        layer_idx: int = -1, # Chỉ số layer cần can thiệp (-1 = layer cuối cùng)
        positions: str = "all", # Vị trí chèn steer vector: tất cả tokens 
        debug: bool = False, # Bật debug in log
    ):
        self.model = model
        self.coeff = float(coeff)
        self.layer_idx = layer_idx
        self.positions = positions.lower()
        self.debug = debug
        self._handle = None # Lưu handle của một hook khi đăng ký một register_hook_forward
    
        # Build steering vector
        parameter = next(model.parameters()) # Lấy một tham số của model để xác định dtype và device
        self.vector = torch.as_tensor(
            steering_vector,
            dtype=parameter.dtype,
            device=parameter.device,
        ) # Chuyển steering_vector thành tensor với dtype và device phù hợp
        if self.vector.ndim != 1:
            raise ValueError("steering_vector must be 1-dimensional")
        hidden = getattr(model.config, "hidden_size", None)
        if hidden and self.vector.numel() != hidden:
            raise ValueError(
                f"Vector length {self.vector.numel()} ≠ model hidden_size {hidden}"
            ) # Kiểm tra kích thước vector phải khớp với hidden_size của model
        
        # Check if positions is valid
        valid_positions = {"all", "prompt", "response"}
        if self.positions not in valid_positions:
            raise ValueError("positions must be 'all', 'prompt', 'response'")
    
    # Tự động phát hiện cấu trúc model để tìm đúng layer cần hook
    def _locate_layer(self):
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:  # found a full match
                if not hasattr(cur, "__getitem__"):
                    continue  # not a list/ModuleList
                if not (-len(cur) <= self.layer_idx < len(cur)):
                    raise IndexError("layer_idx out of range")
                if self.debug:
                    print(f"[ActivationSteerer] hooking {path}[{self.layer_idx}]")
                return cur[self.layer_idx]

        raise ValueError(
            "Could not find layer list on the model. "
            "Add the attribute name to _POSSIBLE_LAYER_ATTRS."
        )
    
    def _hook_fn(self, module, input, output):
        steer = self.coeff * self.vector
        
        def _add_steer(t):
            if self.positions == "all":
                return t + steer.to(t.device)
            elif self.positions == "prompt":
                if t.shape[1] == 1:
                    return t  # Không làm gì nếu chỉ có 1 token
                else:
                    t2 = t.clone() # Cả t2 và t đều trỏ đến cùng một vùng nhớ, nên sửa t2 cũng làm thay đổi t gốc → có thể gây bug nguy hiểm khi dùng trong các hook (vì PyTorch giữ lại t để backward)
                    t2 += steer.to(t.device)
                    return t2
            elif self.positions == "response":
                t2 = t.clone()
                t2[:, -1, :] += steer.to(t.device)
                return t2 # Chỉ thêm steer vào token cuối cùng (câu trả lời)
            else:
                raise ValueError(f"Invalid positions: {self.positions}")
            
        if torch.is_tensor(output):
            new_output = _add_steer(output)
        # Nếu out là tuple/list (ví dụ như T5 trả về (output, hidden_states))
        elif isinstance(output, (tuple, list)):
            if not torch.is_tensor(output[0]):
                return output
            head = _add_steer(output[0]) # Chỉ áp dụng steer vào phần out[0], giữ nguyên các phần khác (attentions, v.v.)
            new_output = (head, *output[1:]) 
        else: # Không biết kiểu output gì
            return output 
        
        # Nếu debug=True, in ra độ lớn trung bình của thay đổi
        if self.debug:
            with torch.no_grad():
                delta = (new_output[0] if isinstance(new_output, tuple) else new_output) 
                delta -= (output[0] if isinstance(output, (tuple, list)) else output)
                print(
                    "[ActivationSteerer] |delta| (mean ± std): "
                    f"{delta.abs().mean():.4g} ± {delta.std():.4g}"
                )
        return new_output
    
    '''
    Phần context manager giúp sử dụng cú pháp như:
    with ActivationSteerer(...) as steerer:
        output = model(input_ids)
    → steering_vector sẽ tự động được gắn vào model khi vào khối with:
      và tự động gỡ bỏ (cleanup) khi thoát khỏi with:
    '''
    def __enter__(self):
        layer = self._locate_layer()
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None

class ActivationSteererMultiple:
    '''
    Cho phép gắn nhiều activation steerers cùng lúc vào nhiều layer khác nhau cùng lúc
    '''

    def __init__(
        self, 
        model: torch.nn.Module,
        instructions: Sequence[dict], # Danh sách các tham số cho từng ActivationSteerer
        *,
        debug: bool = False, # Bật debug in log
    ):
        self.model = model
        self.instructions = instructions
        self.debug = debug
        self._handles = []
        self._steerers = []

        for instr in self.instructions:
            steerer = ActivationSteerer(
                model, 
                instr["steering_vector"],
                coeff=instr.get("coeff", 1.0),
                layer_idx=instr.get("layer_idx", -1),
                positions=instr.get("positions", "all"),
                debug=debug,
            )
            self.steerers.append(steerer)

    def __enter__(self):
        for steerer in self._steerers:
            layer = steerer._locate_layer()
            handle = layer.register_forward_hook(steerer._hook_fn)
            self._handles.append(handle)
        return self
    
    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        for steerer in self._steerers:
            steerer.remove()
        self._handles.clear()