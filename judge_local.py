import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

class LocalJudge:
    def __init__(
        self,
        model_name: str,
        prompt_template: str,
        eval_type: str = "0_100",
        device: str = None,
        top_k: int = 150
    ):
        self.model_name = model_name
        assert eval_type in ["0_100"]
        self.eval_type = eval_type
        self.prompt_template = prompt_template
        self.top_k = top_k

        # Tokenizer luôn dùng CPU
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Load config
        config = AutoConfig.from_pretrained(model_name)

        # Multi-GPU auto split
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

        print("Device map:", self.model.hf_device_map)

    def judge(self, **kwargs) -> float:
        prompt_text = self.prompt_template.format(**kwargs)
        response_probs = self.logprob_probs(prompt_text)
        return self._aggregate_0_100_score(response_probs)

    def logprob_probs(self, prompt: str) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # ❗Không .to(device) khi dùng device_map="auto"

        with torch.no_grad():
            out = self.model(**inputs)

        logits = out.logits[:, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)
        vals, idxs = torch.topk(logprobs, k=self.top_k, dim=-1)

        result = {}
        for logprob, idx in zip(vals[0], idxs[0]):
            decoded = self.tokenizer.decode([idx.item()])
            prob = torch.exp(logprob).item()

            match = re.findall(r"-?\d+", decoded)
            if len(match) == 1:
                result[match[0]] = prob

        return result

    def _aggregate_0_100_score(self, score: dict) -> float:
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if 0 <= int_key <= 100:
                sum_ += int_key * val
                total += val

        if total == 0:
            return None
        return sum_ / total

    def __call__(self, **kwargs):
        return self.judge(**kwargs)
