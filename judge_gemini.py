import math
from google import genai
from google.genai.types import GenerateContentConfig
from config_api import config

class GeminiJudge:
    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        self.model_name = model
        self.client = genai.Client(
            vertexai=True,
            project=config.vertex_project_id,
            location=config.vertex_location,
        )
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

    async def _logprob_probs(self, prompt_text: str) -> dict:
        config = GenerateContentConfig(
            temperature=0,
            max_output_tokens=1,
            response_logprobs=True,
            logprobs=19,
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

        # Debug: In ra response để kiểm tra
        print(f"🔍 Response candidates: {len(response.candidates) if response.candidates else 0}")
        
        # Không có logprobs_result
        if (
            not response.candidates 
            or not response.candidates[0].logprobs_result
        ):
            print("⚠️ No logprobs_result in response")
            # Debug: Kiểm tra xem có text response không
            if response.candidates and hasattr(response.candidates[0], 'content'):
                print(f"📝 Text response: {response.candidates[0].content}")
            return {}

        lp = response.candidates[0].logprobs_result

        # Debug: Kiểm tra cấu trúc logprobs
        print(f"🔍 Logprobs structure: top_candidates length = {len(lp.top_candidates) if hasattr(lp, 'top_candidates') and lp.top_candidates else 0}")
        
        # Kiểm tra top_candidates có tồn tại và không rỗng
        if not hasattr(lp, 'top_candidates') or not lp.top_candidates:
            print("⚠️ No top_candidates in logprobs_result")
            return {}
        
        if len(lp.top_candidates) == 0:
            print("⚠️ top_candidates is empty")
            return {}

        # Token đầu tiên được model sinh ra
        top_candidates = lp.top_candidates[0].candidates
        
        if not top_candidates:
            print("⚠️ No candidates in top_candidates[0]")
            return {}

        probs = {}
        print(f"🔍 Found {len(top_candidates)} candidates")
        for cand in top_candidates:
            token = cand.token.strip()
            prob = math.exp(cand.log_probability)
            
            print(f"  Token: '{token}' (prob: {prob:.4f})")

            # Chỉ nhận token dạng số
            if token.isdigit():
                probs[token] = prob
                print(f"  ✅ Added digit token: {token}")
            else:
                print(f"  ❌ Skipped non-digit token: '{token}'")
        
        print(f"📊 Final probs dict: {probs}")
        return probs

    def _aggregate_0_100_score(self, score: dict) -> float:
        if not score:
            print("⚠️ Empty score dict in aggregate")
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
        if total_p == 0:
            print("⚠️ total_p is 0, no valid tokens found")
            return None

        result = weighted_sum / total_p
        print(f"✅ Calculated score: {result:.2f}")
        return result
