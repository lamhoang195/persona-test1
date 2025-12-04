import math
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch


class LocalJudge:
    def __init__(
        self,
        model_name: str,
        prompt_template: str,
        eval_type: str = "0_100",
        device: str = None,
    ):
        self.model_name = model_name
        assert eval_type in ["0_100"], "eval_type must be 0_100"
        self.eval_type = eval_type
        self.prompt_template = prompt_template

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # ---- FIX rope_scaling for Llama-3.1 ----
        config = AutoConfig.from_pretrained(model_name)

        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            rs = config.rope_scaling

            # Transformers < 4.40 requires: {"type": ..., "factor": ...}
            # Llama 3.1 gives: {"factor": 8, "rope_type": "llama3", ... }
            if "type" not in rs:
                config.rope_scaling = {
                    "type": rs.get("rope_type", "linear"),
                    "factor": float(rs.get("factor", 1.0)),
                }

        # Load model safely
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
        ).to(self.device)

        self.model.eval()

    def judge(self, **kwargs) -> float:
        prompt_text = self.prompt_template.format(**kwargs)
        response_probs = self.logprob_probs(prompt_text)
        score = self._aggregate_0_100_score(response_probs)
        return score

    def logprob_probs(self, prompt: str, top_k=150) -> dict:
        """Compute logprobs for next token using HuggingFace models."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)

        logits = out.logits[:, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)

        topk_vals, topk_idx = torch.topk(logprobs, k=top_k, dim=-1)

        result = {}
        for logprob, idx in zip(topk_vals[0], topk_idx[0]):
            token = self.tokenizer.decode([idx.item()]).strip()
            prob = torch.exp(logprob).item()
            result[token] = prob

        return result

    def _aggregate_0_100_score(self, score: dict) -> float:
        total_weight = 0
        weighted_sum = 0

        for token_str, prob in score.items():
            try:
                token_score = int(token_str)
            except ValueError:
                continue
            if token_score < 0 or token_score > 100:
                continue

            weighted_sum += token_score * prob
            total_weight += prob

        if total_weight == 0:
            return None
        return weighted_sum / total_weight

    def __call__(self, **kwargs):
        return self.judge(**kwargs)
