import math
import re
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
        
        # Debug: In toàn bộ structure của candidate
        if response.candidates:
            candidate = response.candidates[0]
            print(f"🔍 Candidate attributes: {dir(candidate)}")
            print(f"🔍 Has logprobs_result: {hasattr(candidate, 'logprobs_result')}")
            if hasattr(candidate, 'logprobs_result'):
                print(f"🔍 logprobs_result value: {candidate.logprobs_result}")
            
            # Thử lấy text response
            if hasattr(candidate, 'content'):
                print(f"🔍 Content: {candidate.content}")
            if hasattr(candidate, 'parts'):
                print(f"🔍 Parts: {candidate.parts}")
                if candidate.parts:
                    for i, part in enumerate(candidate.parts):
                        print(f"  Part {i}: {part}")
                        if hasattr(part, 'text'):
                            print(f"    Text: {part.text}")
        
        # Không có logprobs_result - thử fallback parse text
        if (
            not response.candidates 
            or not hasattr(response.candidates[0], 'logprobs_result')
            or not response.candidates[0].logprobs_result
        ):
            print("⚠️ No logprobs_result in response, trying text fallback...")
            
            # Fallback: Parse text response để lấy số
            text_response = None
            if response.candidates:
                candidate = response.candidates[0]
                # Thử nhiều cách lấy text
                if hasattr(candidate, 'content') and candidate.content is not None:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts is not None:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_response = part.text
                                break
                if not text_response and hasattr(candidate, 'parts') and candidate.parts is not None:
                    for part in candidate.parts:
                        if hasattr(part, 'text') and part.text:
                            text_response = part.text
                            break
                if not text_response and hasattr(candidate, 'text') and candidate.text:
                    text_response = candidate.text
            
            if text_response:
                print(f"📝 Text response: '{text_response}'")
                # Parse số từ text (tìm số đầu tiên trong khoảng 0-100)
                numbers = re.findall(r'\b(\d{1,2}|100)\b', text_response)
                if numbers:
                    # Lấy số đầu tiên hợp lệ
                    for num_str in numbers:
                        num = int(num_str)
                        if 0 <= num <= 100:
                            print(f"✅ Parsed score from text: {num}")
                            # Trả về dict với prob = 1.0 cho số này
                            return {num_str: 1.0}
                print("⚠️ No valid number (0-100) found in text response")
            else:
                print("⚠️ No text response found")
            
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
