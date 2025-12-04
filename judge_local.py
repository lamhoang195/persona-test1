import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalJudge:
    """
    Local Judge dùng HuggingFace model thay cho OpenAI API.
    Chỉ hỗ trợ eval_type="0_100".
    """
    def __init__(
        self,
        model_path: str | None = None,
        prompt_template: str | None = None,
        device: str = "cuda",
        top_k: int = 50,
        **kwargs,
    ):
        # eval_persona.py still passes model_name/eval_type; keep them for compatibility.
        if model_path is None:
            model_path = kwargs.pop("model_name", None)
        if model_path is None:
            raise ValueError("LocalJudge requires model_path or model_name.")
        if prompt_template is None:
            raise ValueError("LocalJudge requires a prompt_template.")

        self.model_path = model_path
        self.device = device
        self.top_k = top_k
        self.prompt_template = prompt_template
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device
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
        """
        Lấy top-k token probabilities cho token tiếp theo.
        Giống logic của OpenAI logprobs.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)

        logits = out.logits[:, -1, :]  # token cuối
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        topk_vals, topk_idx = torch.topk(logprobs, self.top_k, dim=-1)

        result = {}
        for val, idx in zip(topk_vals[0], topk_idx[0]):
            token_str = self.tokenizer.decode(idx.item()).strip()
            result[token_str] = float(math.exp(val.item()))
 
        return result

    def aggregate_0_100_score(self, score_dict: dict) -> float:
        total = 0
        sum_ = 0

        for token, prob in score_dict.items():
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
