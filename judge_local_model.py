import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LocalJudge:
    """
    Replacement for GeminiJudge but using LOCAL HuggingFace models (Qwen2, Llama3, Mistral...)
    Supports: 0_100, 0_10, binary, binary_text
    """

    def __init__(self, model_path: str, prompt_template: str,
                 eval_type: str = "0_100", top_k: int = 20, device: str = None):

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )

        self.prompt_template = prompt_template
        self.eval_type = eval_type
        self.top_k = top_k

        # Select aggregate function
        if eval_type == "0_100":
            self.aggregate_score = self._aggregate_0_100
        elif eval_type == "0_10":
            self.aggregate_score = self._aggregate_0_10
        elif eval_type == "binary":
            self.aggregate_score = self._aggregate_binary
        elif eval_type == "binary_text":
            self.aggregate_score = self._aggregate_binary_text
        else:
            raise ValueError(f"Unsupported eval_type: {eval_type}")

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)

    async def judge(self, **kwargs):
        prompt = self.prompt_template.format(**kwargs)

        if self.eval_type == "binary_text":
            output = await self._query_text(prompt)
            return self.aggregate_score(output)

        logprob_dict = await self._get_topk_logprobs(prompt)
        return self.aggregate_score(logprob_dict)

    # =====================================================================
    # ===             LOCAL MODEL: TOP-K LOGPROB EXTRACTION             ===
    # =====================================================================

    async def _get_topk_logprobs(self, prompt: str) -> dict:
        """
        Trả về dict: { token_string : probability }
        y hệt logic GeminiJudge._logprob_probs()
        """

        # Encode input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Disable generation → only forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Lấy logits của token cuối cùng
        logits = outputs.logits[:, -1, :]  # shape (1, vocab_size)

        # Lấy TOP-K
        topk = torch.topk(logits, self.top_k, dim=-1)

        tokens = topk.indices[0].tolist()
        logprobs = topk.values[0]
        probs = torch.softmax(logits, dim=-1)[0][tokens]

        result = {}
        for tok_id, prob in zip(tokens, probs):
            token_text = self.tokenizer.decode([tok_id]).strip()
            if token_text:
                result[token_text] = float(prob)

        return result

    # =====================================================================
    # ===                  LOCAL TEXT MODE (binary_text)                 ===
    # =====================================================================

    async def _query_text(self, prompt: str) -> str:
        """Generate 1 token or a short answer"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text[len(prompt):].strip()

    # =====================================================================
    # ===                     AGGREGATE FUNCTIONS                        ===
    # =====================================================================

    def _aggregate_0_100(self, score: dict):
        total, sum_ = 0, 0
        for key, val in score.items():
            try:
                x = int(key)
                if not (0 <= x <= 100):
                    continue
            except:
                continue
            sum_ += x * val
            total += val
        return None if total < 0.25 else sum_ / total

    def _aggregate_0_10(self, score: dict):
        if "REFUSAL" in score and score["REFUSAL"] > max(score.get(str(i), 0) for i in range(10)):
            return None
        total, sum_ = 0, 0
        for key, val in score.items():
            try:
                x = int(key)
                if not (0 <= x <= 9):
                    continue
            except:
                continue
            sum_ += x * val
            total += val
        return None if total < 0.25 else sum_ / total

    def _aggregate_binary(self, score: dict):
        yes_p = score.get("YES", 0)
        no_p = score.get("NO", 0)
        refusal = score.get("REFUSAL", 0)

        if refusal > yes_p and refusal > no_p:
            return None
        s = yes_p + no_p
        return None if s < 0.25 else yes_p / s

    def _aggregate_binary_text(self, text: str):
        text = text.upper()
        if "REFUSAL" in text:
            return None
        if "YES" in text:
            return 1
        if "NO" in text:
            return 0
        return None
