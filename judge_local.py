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
                    dtype=torch.bfloat16,
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
            print(f"[Judge Model] Using cached model: {judge_model}")
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
        probs = self.logprob_probs(prompt)
        score = self.aggregate_0_100_score(probs)
        return score

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
