import math
import re
from vertexai.generative_models import GenerativeModel, GenerationConfig

class GeminiJudge:
    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        self.model_name = model
        self.model = GenerativeModel(model_name=self.model_name)
        self.prompt_template = prompt_template
        assert eval_type in ["0_100", "0_10", "binary", "binary_text"], f"Unsupported eval_type: {eval_type}"
        self.eval_type = eval_type

        if self.eval_type == "0_100":
            self.aggregate_score = self._aggregate_0_100_score
        elif self.eval_type == "0_10":
            self.aggregate_score = self._aggregate_0_10_score
        elif self.eval_type == "binary":
            self.aggregate_score = self._aggregate_binary_score
        elif self.eval_type == "binary_text":
            self.aggregate_score = self._aggregate_binary_text_score

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)

    async def judge(self, **kwargs):
        prompt_text = self.prompt_template.format(**kwargs)
        contents = [{"role": "user", "parts": [{"text": prompt_text}]}]

        if self.eval_type == "binary_text":
            response_text = await self._query_full_text(contents)
            return self.aggregate_score(response_text)

        logprobs = await self._logprob_probs(contents)
        if logprobs:
            return self.aggregate_score(logprobs)

    async def _logprob_probs(self, contents) -> dict:
        generation_config = GenerationConfig(
            max_output_tokens=1,
            temperature=0,
            response_logprobs=True,
            logprobs=19,
            seed=0
        )
        try:
            response = await self.model.generate_content_async(
                contents=contents,
                generation_config=generation_config
            )

            logprobs_result = response.candidates[0].logprobs_result

            token_entries = getattr(logprobs_result, "token_logprobs", None)
            if token_entries is None:
                token_entries = getattr(logprobs_result, "tokens", None)
            if not token_entries:
                return {}

            entry = token_entries[0]

            top_candidates = getattr(entry, "top_logprobs", None)
            if top_candidates is None:
                top_candidates = getattr(entry, "top_tokens", None)
            if not top_candidates:
                return {}

        except (Exception, IndexError, AttributeError) as e:
            print(f"⚠️ Lỗi khi gọi hoặc phân tích Vertex AI logprobs: {e}")
            return {}

        result = {}
        for candidate in top_candidates:
            token = getattr(candidate, "token", getattr(candidate, "text", "")).strip()
            logprob = getattr(candidate, "logprob", None)
            if token and logprob is not None:
                result[token] = float(math.exp(logprob))

        return result

    async def _query_full_text(self, contents: list) -> str:
        try:
            response = await self.model.generate_content_async(
                contents=contents,
                generation_config={"temperature": 0}
            )
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ Vertex AI full text error: {e}")
            return ""

    def _tokens_from_text(self, response_text: str) -> dict[str, float]:
        text = (response_text or "").strip()
        if not text:
            return {}

        if self.eval_type in ("0_100", "0_10"):
            numbers = re.findall(r"-?\d+", text)
            if not numbers:
                return {}
            value = numbers[-1]
            limit = 100 if self.eval_type == "0_100" else 9
            try:
                as_int = int(value)
            except ValueError:
                return {}
            if 0 <= as_int <= limit:
                return {str(as_int): 1.0}
            return {}

        if self.eval_type == "binary":
            upper = text.upper()
            if "REFUSAL" in upper:
                return {"REFUSAL": 1.0}
            if "YES" in upper:
                return {"YES": 1.0}
            if "NO" in upper:
                return {"NO": 1.0}
            return {}

        return {}

    def _aggregate_0_100_score(self, score: dict) -> float:
        total, sum_ = 0, 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if not (0 <= int_key <= 100):
                continue
            sum_ += int_key * val
            total += val
        return None if total < 0.25 else sum_ / total

    def _aggregate_0_10_score(self, score: dict) -> float:
        if "REFUSAL" in score and score["REFUSAL"] > max(score.get(str(i), 0) for i in range(10)):
            return None
        total, sum_ = 0, 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if not (0 <= int_key <= 9):
                continue
            sum_ += int_key * val
            total += val
        return None if total < 0.25 else sum_ / total

    def _aggregate_binary_score(self, score: dict) -> float:
        yes_prob = score.get("YES", 0.0)
        no_prob = score.get("NO", 0.0)
        refusal_prob = score.get("REFUSAL", 0.0)
        if refusal_prob > yes_prob and refusal_prob > no_prob:
            return None
        denominator = yes_prob + no_prob
        return None if denominator < 0.25 else yes_prob / denominator

    def _aggregate_binary_text_score(self, response_text: str) -> float | None:
        if "<answer>REFUSAL</answer>" in response_text:
            return None
        elif "<answer>NO</answer>" in response_text:
            return 0
        elif "<answer>YES</answer>" in response_text:
            return 1
        return None