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
        # Thử 2 cách: một với logprobs, một với text generation
        config_logprob = GenerateContentConfig(
            temperature=0,
            max_output_tokens=10,  # Tăng lên để model có thể generate
            response_logprobs=True,
            logprobs=19,
            seed=0,
        )
        
        config_text = GenerateContentConfig(
            temperature=0,
            max_output_tokens=10,  # Đủ để generate số
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

        # Thử lấy logprobs trước
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config_logprob,
            )
        except Exception as e:
            print("❌ Logprob API error:", e)
            # Fallback: thử generate text thông thường
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config_text,
                )
            except Exception as e2:
                print("❌ Text generation API error:", e2)
                return {}

        # Debug: In ra response để kiểm tra
        print(f"🔍 Response candidates: {len(response.candidates) if response.candidates else 0}")
        
        if not response.candidates:
            print("⚠️ No candidates in response")
            return {}
        
        candidate = response.candidates[0]
        
        # Kiểm tra logprobs_result
        has_logprobs = (
            hasattr(candidate, 'logprobs_result') 
            and candidate.logprobs_result is not None
        )
        
        if has_logprobs:
            print("✅ Found logprobs_result")
            lp = candidate.logprobs_result
            
            if not hasattr(lp, 'top_candidates') or not lp.top_candidates:
                print("⚠️ No top_candidates in logprobs_result")
            else:
                if len(lp.top_candidates) == 0:
                    print("⚠️ top_candidates is empty")
                else:
                    top_candidates = lp.top_candidates[0].candidates
                    
                    if not top_candidates:
                        print("⚠️ No candidates in top_candidates[0]")
                    else:
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
                        
                        if probs:
                            print(f"📊 Final probs dict: {probs}")
                            return probs
        
        # Fallback: Parse text response
        print("⚠️ No logprobs available, trying text fallback...")
        
        # Thử nhiều cách lấy text
        text_response = None
        
        # Cách 1: candidate.content.parts
        if hasattr(candidate, 'content') and candidate.content is not None:
            if hasattr(candidate.content, 'parts') and candidate.content.parts is not None:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_response = part.text.strip()
                        print(f"📝 Got text from candidate.content.parts: '{text_response}'")
                        break
        
        # Cách 2: candidate.parts
        if not text_response and hasattr(candidate, 'parts') and candidate.parts is not None:
            for part in candidate.parts:
                if hasattr(part, 'text') and part.text:
                    text_response = part.text.strip()
                    print(f"📝 Got text from candidate.parts: '{text_response}'")
                    break
        
        # Cách 3: candidate.text
        if not text_response and hasattr(candidate, 'text') and candidate.text:
            text_response = candidate.text.strip()
            print(f"📝 Got text from candidate.text: '{text_response}'")
        
        # Cách 4: response.text (nếu có)
        if not text_response and hasattr(response, 'text') and response.text:
            text_response = response.text.strip()
            print(f"📝 Got text from response.text: '{text_response}'")
        
        # Cách 5: Thử gọi lại API chỉ để lấy text (nếu vẫn không có)
        if not text_response:
            print("⚠️ No text found, trying separate text generation call...")
            try:
                text_response_obj = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config_text,
                )
                if text_response_obj.candidates:
                    text_candidate = text_response_obj.candidates[0]
                    if hasattr(text_candidate, 'content') and text_candidate.content:
                        if hasattr(text_candidate.content, 'parts') and text_candidate.content.parts:
                            for part in text_candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    text_response = part.text.strip()
                                    print(f"📝 Got text from separate API call: '{text_response}'")
                                    break
            except Exception as e:
                print(f"❌ Separate text generation error: {e}")
        
        if text_response:
            print(f"📝 Final text response: '{text_response}'")
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
            print("⚠️ No text response found at all")
        
        return {}

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
