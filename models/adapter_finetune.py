import re, torch, os
from pathlib import Path
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)") # Pattern để nhận diện tên thư mục checkpoint

'''
Khi fine-tune một LLM, quá trình training thường chạy rất lâu (vài giờ đến vài ngày).
Để tránh mất dữ liệu nếu gặp sự cố, thư viện training sẽ lưu model định kỳ theo các mốc:
my_model/
├── checkpoint-500/      # Lưu sau 500 steps
'''
def _pick_latest_checkpoint(model_path: str) -> str:
    # Tìm tất cả thư mục con khớp pattern checkpoint-<number>
    ckpts = [(int(m.group(1)), p) for p in Path(model_path).iterdir()
             if (m := _CHECKPOINT_RE.fullmatch(p.name)) and p.is_dir()]
    # Trả về thư mục có số lớn nhất (checkpoint mới nhất)
    return str(max(ckpts, key=lambda x: x[0])[1]) if ckpts else model_path

'''
Mục đích: Kiểm tra có phải là mô hình LoRA không
Output: True nếu có file adapter_config.json, ngược lại False
'''
def _is_lora(path: str) -> bool:
    return Path(path, "adapter_config.json").exists()

'''
Mục đích: Load mô hình LoRA và merge vào base model
Cách hoạt động:
    cfg = PeftConfig.from_pretrained(lora_path): Load config của LoRA
    base = AutoModelForCausalLM.from_pretrained(...): Load mô hình base gốc
    PeftModel.from_pretrained(base, lora_path): Gắn LoRA adapter vào base
    .merge_and_unload(): Merge LoRA weights vào base model và remove adapter
Output: Mô hình đầy đủ đã merge (không còn tách biệt LoRA adapter)
'''
def _load_and_merge_lora(lora_path: str, dtype, device_map):
    cfg = PeftConfig.from_pretrained(lora_path)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path, torch_dtype=dtype, device_map=device_map
    )
    return PeftModel.from_pretrained(base, lora_path).merge_and_unload()

'''
Mục đích: Load tokenizer và cấu hình cho batch processing
Cách hoạt động:
    Load tokenizer từ path hoặc Hugging Face Hub
    Set pad_token = eos_token: Dùng token kết thúc để padding
    padding_side = "left": Padding bên trái (quan trọng cho generation)
Output: Tokenizer đã cấu hình
'''
def _load_tokenizer(path_or_id: str):
    token = AutoTokenizer.from_pretrained(path_or_id)
    token.pad_token = token.eos_token
    token.pad_token_id = token.eos_token_id
    token.padding_side = "left"
    return token

def load_model(model_path: str, dtype=torch.bfloat16):
    if not os.path.exists(model_path):               # ---- Hub ----
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto"
        )
        tok = _load_tokenizer(model_path)
        return model, tok

    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    if _is_lora(resolved):
        model = _load_and_merge_lora(resolved, dtype, "auto")
        tok = _load_tokenizer(model.config._name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved, torch_dtype=dtype, device_map="auto"
        )
        tok = _load_tokenizer(resolved)
    return model, tok

def load_model_basic(model_path: str, dtype=torch.bfloat16):
    """
    Load model using transformers (not vLLM) but with same logic as load_vllm_model.
    Returns (model, tokenizer, lora_path) to maintain compatibility.
    This function avoids vLLM dependency and CUDA linking issues.
    """
    if not os.path.exists(model_path):               # ---- Hub ----
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto"
        )
        tok = _load_tokenizer(model_path)
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "left"
        return model, tok, None

    # ---- 本地 ----
    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    is_lora = _is_lora(resolved)

    if is_lora:
        # Load LoRA adapter (merge it, so lora_path is returned but model is merged)
        model = _load_and_merge_lora(resolved, dtype, "auto")
        tok = _load_tokenizer(model.config._name_or_path)
        lora_path = resolved  # Return path for reference, but model is already merged
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved, torch_dtype=dtype, device_map="auto"
        )
        tok = _load_tokenizer(resolved)
        lora_path = None

    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return model, tok, lora_path
