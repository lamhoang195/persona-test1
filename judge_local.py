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

        # Load tokenizer as usual
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load and patch config to avoid rope_scaling validation error on older transformers
        config = AutoConfig.from_pretrained(model_name)
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            # Older transformers expect a dict with only "type" and "factor"
            if "type" not in rope_scaling and "factor" in rope_scaling:
                # Best-effort fallback: keep the factor and set a generic type
                config.rope_scaling = {
                    "type": rope_scaling.get("rope_type", "linear"),
                    "factor": rope_scaling["factor"],
                }

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
        ).to(self.device)
        self.model.eval()

    def judge(self, **kwargs) -> float:
        prompt_text = self.prompt_template.format(**kwargs)
        response_probs = self._logprob_prob(prompt_text)
        score = self._aggregate_0_100_score(response_probs)
        return score

    def logprob_probs(self, prompt: str, top_k=150) -> dict:
        """Tính logprobs token tiếp theo bằng HuggingFace."""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)

        logits = out.logits[:, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)

        topk_vals, topk_idx = torch.topk(logprobs, k=top_k, dim=-1)

        result = {}
        for logprob, idx in zip(topk_vals[0], topk_idx[0]):
            token = self.tokenizer.decode([idx.item()]).strip()  # FIXED
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
