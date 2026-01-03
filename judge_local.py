import re
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Thêm cache ở đầu file
_judge_model_cache = {}

class LocalJudge:
    def __init__(
        self,
        judge_model: str,
        prompt_template: str,
        top_k: int = 20,
        verbose: bool = False,
        **kwargs,
    ):
        self.judge_model = judge_model
        self.top_k = top_k
        self.prompt_template = prompt_template
        self.verbose = verbose

        # Sử dụng cache để tránh load model nhiều lần
        if judge_model not in _judge_model_cache:
            print(f"[Judge Model] Loading from: {judge_model}")
            print(f"[Judge Model] This may take a while...")
            if self.verbose:
                print(f"Loading tokenizer from {judge_model}...")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    judge_model, 
                    use_fast=False,
                    trust_remote_code=True
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer: {e}")

            if self.verbose:
                print(f"Loading model from {judge_model}...")
            
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    judge_model,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="cuda",
                    attn_implementation="eager"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            _judge_model_cache[judge_model] = {
                'model': model,
                'tokenizer': tokenizer
            }
            print(f"[Judge Model] Loaded and cached successfully")
        else:
            if self.verbose:
                print(f"Using cached model for {judge_model}...")

        # Lấy model và tokenizer từ cache
        cached = _judge_model_cache[judge_model]
        self.model = cached['model']
        self.tokenizer = cached['tokenizer']

    def __call__(self, **kwargs):
        return self.judge(**kwargs)

    def judge(self, **kwargs):
        prompt = self.prompt_template.format(**kwargs)
        
        # Phương pháp 1: Generate và parse số từ output (chính xác hơn)
        score = self.generate_score(prompt)
        if score is not None:
            return score
        
        # Fallback: Phương pháp logprob (nếu generate thất bại)
        probs = self.logprob_probs(prompt)
        return self.aggregate_0_100_score(probs)
    
    def generate_score(self, prompt: str) -> float | None:
        """Generate response và extract số từ output."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=15,  # Đủ cho "Score: 100" hoặc tương tự
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode phần generated
            prompt_len = inputs["input_ids"].shape[1]
            generated = self.tokenizer.decode(
                output[0][prompt_len:], 
                skip_special_tokens=True
            ).strip()
            
            if self.verbose:
                print(f"[DEBUG] Generated: '{generated}'")
            
            # Extract số 0-100 từ output
            match = re.search(r'\b(\d{1,3})\b', generated)
            if match:
                score = int(match.group(1))
                if 0 <= score <= 100:
                    return float(score)
            
            return None
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] Generate error: {e}")
            return None

    def logprob_probs(self, prompt: str) -> dict:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                out = self.model(**inputs)

            logits = out.logits[:, -1, :]
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            topk_vals, topk_idx = torch.topk(logprobs, self.top_k, dim=-1)
            
            result = {}
            for val, idx in zip(topk_vals[0], topk_idx[0]):
                token_str = self.tokenizer.decode([idx.item()], clean_up_tokenization_spaces=False)
                token_str = token_str.strip()
                result[token_str] = float(math.exp(val.item()))
            
            return result
        except Exception as e:
            if self.verbose:
                print(f"Error during inference: {e}")
            return {}

    def aggregate_0_100_score(self, scores: dict) -> float | None:
        total = 0
        sum_ = 0
        for token, prob in scores.items():
            try:
                int_token = int(token)
            except ValueError:
                continue
            if 0 <= int_token <= 100:
                sum_ += int_token * prob
                total += prob
        if total == 0:
            return None
        return sum_ / total
