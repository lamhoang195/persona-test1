import math
from google import genai
from google.genai.types import GenerateContentConfig


class GeminiJudge:
    """
    Judge using Gemini via Vertex AI GenAI.
    Supports:
        - eval_type="0_100" → dùng logprobs để lấy phân phối số 0–100
    """

    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        self.model_name = model
        self.client = genai.Client(vertexai=True)
        self.prompt_template = prompt_template

        assert eval_type in ["0_100"], f"Unsupported eval_type: {eval_type}"
        self.eval_type = eval_type

        if self.eval_type == "0_100":
            self.aggregate_score = self._aggregate_0_100_score

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)

    async def judge(self, **kwargs):
        prompt = self.prompt_template.format(**kwargs)

        # Dùng logprob nếu đánh giá số
        logprobs = await self._logprob_probs(prompt)
        score = self.aggregate_score(logprobs)
        return score

    # ======================================================================
    # 🔥 LẤY LOGPROB ĐÚNG CHUẨN GIỐNG NHƯ DEMO GOOGLE NGÀY 15-11-2024
    # ======================================================================
    async def _logprob_probs(self, prompt_text: str) -> dict:
        """
        Trả về dict: token → probability
        Chỉ nhận token dạng số (digit).
        """

        config = GenerateContentConfig(
            temperature=0,
            max_output_tokens=1,
            response_logprobs=True,
            logprobs=20,      # giống code Colab
            seed=0,
        )

        # Định dạng contents đúng chuẩn GenAI
        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ]

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
        except Exception as e:
            print("❌ Logprob API error:", e)
            return {}

        # Không có logprobs_result
        if (
            not response.candidates 
            or not response.candidates[0].logprobs_result
        ):
            return {}

        lp = response.candidates[0].logprobs_result

        # Token đầu tiên được model sinh ra
        top_candidates = lp.top_candidates[0].candidates

        probs = {}

        for cand in top_candidates:
            token = cand.token.strip()
            prob = math.exp(cand.log_probability)

            # Chỉ nhận token dạng số
            if token.isdigit():
                probs[token] = prob

        return probs

    # ======================================================================
    # 🔥 AGGREGATE SCORE 0–100
    # ======================================================================
    def _aggregate_0_100_score(self, score: dict) -> float:
        """
        Tính expected value của phân phối:
            sum(p_i * i)
        Chỉ nhận i từ 0–100.
        """

        if not score:
            return None

        total_p = 0
        weighted_sum = 0

        for token, prob in score.items():
            try:
                num = int(token)
            except ValueError:
                continue

            if 0 <= num <= 100:
                weighted_sum += num * prob
                total_p += prob

        # nếu phân phối quá loãng (model không tự tin)
        if total_p < 0.25:
            return None

        return weighted_sum / total_p
