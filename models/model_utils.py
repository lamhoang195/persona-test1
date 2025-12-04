import os, re, json, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
from contextlib import contextmanager

from activation_steer import ActivationSteerer, ActivationSteererMultiple
from models.model import Model
from models.utils import sample_token, get_last_attn
from models.adapter_finetune import _pick_latest_checkpoint, _is_lora, _load_and_merge_lora, _load_tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@contextmanager
def dummy_context():
    yield
    
class ModelUtils(Model):
    def __init__(self, config):
        super().__init__(config)
        self.name = config["model_info"]["name"]
        self.max_output_tokens = int(config["params"]["max_output_tokens"])

        model_path = config["model_info"]["model_path"]
        latest_ckpt = _pick_latest_checkpoint(model_path)

        if _is_lora(latest_ckpt):
            # Nếu là model LoRA → load base model + merge LoRA
            print(f"[INFO] Detected LoRA model. Loading and merging from: {latest_ckpt}")
            self.model = _load_and_merge_lora(
                latest_ckpt,
                dtype=torch.bfloat16,
                device_map=device
            ).eval()
            # Dùng tokenizer từ base model
            cfg = PeftConfig.from_pretrained(latest_ckpt)
            self.tokenizer = _load_tokenizer(cfg.base_model_name_or_path)
        else:
            # Nếu không phải LoRA → load model gốc hoặc fine-tuned full
            print(f"[INFO] Loading base or full fine-tuned model from: {latest_ckpt}")
            self.model = AutoModelForCausalLM.from_pretrained(
                latest_ckpt,
                dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
                attn_implementation="eager"
            ).eval()
            self.tokenizer = _load_tokenizer(latest_ckpt)

        self.top_k = 50
        self.top_p = None

        if config["params"]["important_heads"] == "all":
            attn_size = self.get_map_dim()
            self.important_heads = [[i, j] for i in range(
                attn_size[0]) for j in range(attn_size[1])]
        else:
            self.important_heads = config["params"]["important_heads"]

    def get_map_dim(self):
        _, _, attention_maps, _, _, _ = self.inference("print hi", "")
        attention_map = attention_maps[0]
        return len(attention_map), attention_map[0].shape[1]
    
    def inference(self, instruction, data, max_output_tokens=None, steering_info=None):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": "Data: " + data}
        ]

        # Áp dụng template chat để tạo prompt đầy đủ (text)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        instruction_len = len(self.tokenizer.encode(instruction))
        data_len = len(self.tokenizer.encode(data))

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device) # Mã hóa text thành tensor (input_ids, attention_mask)
        input_tokens = self.tokenizer.convert_ids_to_tokens(model_inputs['input_ids'][0])

        # Vvùng chứa instruction và data
        if "qwen" in self.name:
            data_range = ((3, 3+instruction_len), (-5-data_len, -5))
        elif "phi3" in self.name:
            data_range = ((1, 1+instruction_len), (-2-data_len, -2))
        elif "llama3-8b" in self.name:
            data_range = ((5, 5+instruction_len), (-5-data_len, -5)) # ???
        elif "mistral-7b" in self.name:
            data_range = ((3, 3+instruction_len), (-1-data_len, -1)) # ???
        elif "granite3-8b" in self.name:
            data_range = ((3, 3+instruction_len), (-5-data_len, -5)) # ???
        else:
            raise NotImplementedError

        generated_tokens = [] # Danh sách lưu token được sinh
        generated_probs = [] # Danh sách lưu xác suất token được sinh
        input_ids = model_inputs.input_ids # Token ID
        attention_mask = model_inputs.attention_mask # Tensor có giá trị 1 ở vị trí có token, 0 ở vị trí padding

        attention_maps = [] # Danh sách lưu attention maps

        if max_output_tokens != None:
            n_tokens = max_output_tokens
        else:
            n_tokens = self.max_output_tokens

        # Nếu có steer vector => tạo ActivationSteerer context
        if steering_info is not None:
            steerer = ActivationSteerer(
                self.model,
                steering_vector=steering_info["steering_vector"],
                coeff=steering_info.get("coeff", 1.0),
                layer_idx=steering_info.get("layer_idx", -1),
                positions=steering_info.get("positions", "all"),
                debug=steering_info.get("debug", False)
            )
        else:
            steerer = None

        with steerer if steerer else dummy_context():
            with torch.inference_mode():
                # Bước 1: Sinh token đầu tiên với attention maps
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True # Yêu cầu trả về attention maps
                )

                '''
                Logit: output trước khi qua softmax để thành xác suất (điểm số thô cho mỗi token trong từ điển)

                '''
                logits = output.logits[:, -1, :] # Lấy logits của token tiếp theo
                probs = F.softmax(logits, dim=-1) # Chuyển logits thành xác suất
                next_token_id = sample_token(
                    logits[0],
                    top_k=self.top_k, 
                    top_p=self.top_p, 
                    temperature=1.0
                )[0] # Lấy token tiếp theo bằng sampling (lấy token tiếp theo dựa trên xác suất)

                generated_probs.append(probs[0, next_token_id.item()].item()) # Lưu xác suất token được chọn
                generated_tokens.append(next_token_id.item()) # Lưu token được chọn

                # Có list attention maps, mỗi phần tử là một layer, có shape: (batch_size, num_heads, seq_len, seq_len), seq_len: độ dài chuỗi input (prompt)
                attention_map = [att.detach().cpu().half() for att in output['attentions']]
                # Một số giá trị attention bị NaN (ví dụ do chia 0 trong softmax). torch.nan_to_num() thay toàn bộ NaN thành 0.0 để tránh lỗi về sau (visualization, thống kê...).
                attention_map = [torch.nan_to_num(att, nan=0.0) for att in attention_map]
                # Lấy attention map của token cuối cùng (token được sinh)
                attention_map = get_last_attn(attention_map)
                attention_maps = [attention_map]
                del output  # free references

                # Kiểm tra nếu token sinh ra là eos_token (kết thúc) thì dừng luôn
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    output_tokens = [self.tokenizer.decode(t, skip_special_tokens=True) for t in generated_tokens]
                    return generated_text, output_tokens, attention_maps, input_tokens, data_range, generated_probs

                # Chuẩn bị input cho bước tiếp theo
                input_ids = torch.cat((input_ids, next_token_id.unsqueeze(0).unsqueeze(0)), dim=-1) # Thêm token mới vào input_ids
                attention_mask = torch.cat((attention_mask, torch.tensor([[1]], device=input_ids.device)), dim=-1) # Cập nhật attention_mask

                # Bước 2+: Sinh các token còn lại mà không cần attention maps
                remaining = (max_output_tokens if max_output_tokens is not None else self.max_output_tokens) - 1
                for _ in range(remaining):
                    out2 = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=False
                    )
                    logits = out2.logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    next_token_id = sample_token(logits[0], top_k=self.top_k, top_p=self.top_p, temperature=1.0)[0]
                    generated_probs.append(probs[0, next_token_id.item()].item())
                    generated_tokens.append(next_token_id.item())
                    if next_token_id.item() == self.tokenizer.eos_token_id:
                        break
                    input_ids = torch.cat((input_ids, next_token_id.unsqueeze(0).unsqueeze(0)), dim=-1)
                    attention_mask = torch.cat((attention_mask, torch.tensor([[1]], device=input_ids.device)), dim=-1)

            output_tokens = [self.tokenizer.decode(t, skip_special_tokens=True) for t in generated_tokens]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return generated_text, output_tokens, attention_maps, input_tokens, data_range, generated_probs

    