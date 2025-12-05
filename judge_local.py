import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LocalJudge:
    def __init__(
        self,
        model_path: str | None = None,
        prompt_template: str | None = None,
        top_k: int = 50,
        **kwargs,
    ):
        if model_path is None:
            model_path = kwargs.pop("model_name", None)
        if model_path is None:
            raise ValueError("LocalJudge requires model_path or model_name.")
        if prompt_template is None:
            raise ValueError("LocalJudge requires a prompt_template.")

        self.model_path = model_path
        self.top_k = top_k
        self.prompt_template = prompt_template
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="eager"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, **kwargs):
        return self.judge(**kwargs)

    def judge(self, **kwargs):
        prompt = self.prompt_template.format(**kwargs)
        probs = self.logprob_probs(prompt)
        score = self.aggregate_0_100_score(probs)
        return score

    def logprob_probs(self, prompt: str) -> dict:
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

    def aggregate_0_100_score(self, scores: dict) -> float:
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
