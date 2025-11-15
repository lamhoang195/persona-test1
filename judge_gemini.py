import math
from gemini_key_manager import gemini_key_manager
from google.generativeai import GenerativeModel

class GeminiJudge:
    """
    Gemini Judge for scoring using logprobs or binary text output.
    Rotates API keys automatically using gemini_key_manager.
    """

    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        self.model_name = model
        self.prompt_template = prompt_template
        assert eval_type in ["0_100", "0_10", "binary", "binary_text"], f"Unsupported eval_type: {eval_type}"
        self.eval_type = eval_type

        # Chọn hàm tổng hợp (aggregate) phù hợp với loại đánh giá
        if self.eval_type == "0_100":
            self.aggregate_score = self._aggregate_0_100_score
        elif self.eval_type == "0_10":
            self.aggregate_score = self._aggregate_0_10_score
        elif self.eval_type == "binary":
            self.aggregate_score = self._aggregate_binary_score
        elif self.eval_type == "binary_text":
            self.aggregate_score = self._aggregate_binary_text_score

        # Khởi tạo model (Gemini API)
        self.model = GenerativeModel(model)

    async def judge(self, **kwargs):
        prompt = self.prompt_template.format(**kwargs)
        gemini_key_manager.increment()

        if self.eval_type == "binary_text":
            response_text = await self._query_full_text(prompt)
            return self.aggregate_score(response_text)
        else:
            logprobs = await self._logprob_probs(prompt)
            return self.aggregate_score(logprobs)

    async def _logprob_probs(self, prompt: str) -> dict:
        """Gọi Gemini để lấy logprobs (dự đoán 1 token)."""
        try:
            response = await self.model.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 1,
                    "logprobs": 20,
                    "response_mime_type": "application/json",
                    "response_logprobs": True,
                }
            )

            logprobs_result = response.candidates[0].logprobs_result
            if not logprobs_result or not logprobs_result.top_logprobs:
                return {}

            # top_logprobs: List[List[TopLogProb]]
            top = logprobs_result.top_logprobs[0]
            return {token.token: math.exp(token.logprob) for token in top}

        except Exception as e:
            print(f"⚠️ Gemini logprob error: {e}")
            return {}

    async def _query_full_text(self, prompt: str) -> str:
        """Dùng cho binary_text, không lấy logprobs mà lấy văn bản trả về."""
        try:
            response = await self.model.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                generation_config={"temperature": 0}
            )
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ Gemini full text error: {e}")
            return ""

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

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)
