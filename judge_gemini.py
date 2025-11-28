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
            max_output_tokens=1,  # Giữ như code mẫu của Google
            response_logprobs=True,
            logprobs=19,  # Top 19 logprobs
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

        # Debug: In ra toàn bộ response structure
        print(f"🔍 Response type: {type(response)}")
        print(f"🔍 Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
        print(f"🔍 Response candidates: {len(response.candidates) if response.candidates else 0}")
        
        if not response.candidates:
            print("⚠️ No candidates in response")
            return {}
        
        candidate = response.candidates[0]
        
        # Debug: In toàn bộ candidate structure
        print(f"🔍 Candidate type: {type(candidate)}")
        print(f"🔍 Candidate attributes: {[attr for attr in dir(candidate) if not attr.startswith('_')]}")
        
        # Kiểm tra logprobs_result
        if hasattr(candidate, 'logprobs_result'):
            print(f"🔍 logprobs_result type: {type(candidate.logprobs_result)}")
            print(f"🔍 logprobs_result value: {candidate.logprobs_result}")
            
            if candidate.logprobs_result is not None:
                lp = candidate.logprobs_result
                print(f"🔍 logprobs_result attributes: {[attr for attr in dir(lp) if not attr.startswith('_')]}")
                
                if hasattr(lp, 'top_candidates'):
                    print(f"🔍 top_candidates: {lp.top_candidates}")
                    if lp.top_candidates:
                        print(f"🔍 top_candidates length: {len(lp.top_candidates)}")
                        if len(lp.top_candidates) > 0:
                            print(f"🔍 top_candidates[0] type: {type(lp.top_candidates[0])}")
                            print(f"🔍 top_candidates[0] attributes: {[attr for attr in dir(lp.top_candidates[0]) if not attr.startswith('_')]}")
                            
                            if hasattr(lp.top_candidates[0], 'candidates'):
                                top_candidates = lp.top_candidates[0].candidates
                                print(f"🔍 candidates: {top_candidates}")
                                print(f"🔍 candidates length: {len(top_candidates) if top_candidates else 0}")
                                
                                if top_candidates:
                                    probs = {}
                                    print(f"🔍 Found {len(top_candidates)} candidates")
                                    for cand in top_candidates:
                                        print(f"🔍 Candidate type: {type(cand)}")
                                        print(f"🔍 Candidate attributes: {[attr for attr in dir(cand) if not attr.startswith('_')]}")
                                        
                                        if hasattr(cand, 'token'):
                                            token = cand.token.strip() if cand.token else ""
                                        elif hasattr(cand, 'text'):
                                            token = cand.text.strip() if cand.text else ""
                                        else:
                                            token = str(cand).strip()
                                        
                                        if hasattr(cand, 'log_probability'):
                                            prob = math.exp(cand.log_probability)
                                        elif hasattr(cand, 'logprob'):
                                            prob = math.exp(cand.logprob)
                                        else:
                                            print(f"⚠️ No log_probability found in candidate")
                                            continue
                                        
                                        print(f"  Token: '{token}' (prob: {prob:.4f})")

                                        # Chỉ nhận token dạng số
                                        if token.isdigit():
                                            probs[token] = prob
                                            print(f"  ✅ Added digit token: {token}")
                                    
                                    if probs:
                                        print(f"📊 Final probs dict: {probs}")
                                        return probs
        
        # Fallback: Thử lấy text response
        print("⚠️ No logprobs available, trying text fallback...")
        
        # Debug: Thử nhiều cách lấy text
        text_response = None
        
        # In toàn bộ để debug
        print(f"🔍 candidate.content: {candidate.content}")
        print(f"🔍 candidate.parts: {candidate.parts}")
        if hasattr(candidate, 'text'):
            print(f"🔍 candidate.text: {candidate.text}")
        if hasattr(response, 'text'):
            print(f"🔍 response.text: {response.text}")
        
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
        
        # Cách 4: response.text
        if not text_response and hasattr(response, 'text') and response.text:
            text_response = response.text.strip()
            print(f"📝 Got text from response.text: '{text_response}'")
        
        if text_response:
            print(f"📝 Final text response: '{text_response}'")
            # Parse số từ text
            numbers = re.findall(r'\b(\d{1,2}|100)\b', text_response)
            if numbers:
                for num_str in numbers:
                    num = int(num_str)
                    if 0 <= num <= 100:
                        print(f"✅ Parsed score from text: {num}")
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
