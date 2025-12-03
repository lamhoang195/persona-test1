import math
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LocalJudge:
    def __init__(self, model_name: str, prompt_template: str, eval_type: str = "0_100", device: str = None):
        self.model_name = model_name
        assert eval_type in ["0_100"], "eval_type must be 0_100"
        self.eval_type = eval_type
        self.prompt_template = prompt_template

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def judge(self, **kwargs) -> float:
        prompt_text = self.prompt_template.format(**kwargs)
        response_probs = self._logprob_prob(prompt_text)
        score = self._aggregate_0_100_score(response_probs)
        return score

    def _logprob_prob(self, prompt_text: str) -> dict:
        # Tokenize prompt
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Lấy token tiếp theo sau prompt (max_output_tokens=1)
        next_token_logits = logits[0, -1, :]
        log_probs = torch.log_softmax(next_token_logits, dim=-1)

        result = {}
        for i in range(10):  # token "0" đến "9"
            token_id = self.tokenizer.convert_tokens_to_ids(str(i))
            if token_id is not None:
                result[str(i)] = math.exp(log_probs[token_id].item())

        return result

    def _aggregate_0_100_score(self, score: dict) -> float:
        total_weight = 0
        weighted_sum = 0
        for token_str, prob in score.items():
            try:
                token_score = int(token_str)
            except ValueError:
                continue
            if token_score < 0 or token_score > 9:
                continue
            weighted_sum += token_score * prob
            total_weight += prob
        if total_weight == 0:
            return None
        return weighted_sum / total_weight

    def __call__(self, **kwargs):
        return self.judge(**kwargs)
