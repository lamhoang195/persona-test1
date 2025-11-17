import os
import math
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import google.generativeai as genai

# # config = setup_credentials()
# # genai.configure(api_key=config.gemini_api_keys[0])

# try:
#     PROJECT_ID = "YOUR-PROJECT-ID"  # 👈 THAY THẾ BẰNG PROJECT ID CỦA BẠN
#     LOCATION = "us-central1"       # 👈 THAY THẾ BẰNG REGION (ví dụ: us-central1)
    
#     vertexai.init(project=PROJECT_ID, location=LOCATION)
# except Exception as e:
#     print(f"Lỗi khi khởi tạo Vertex AI. Bạn đã cài đặt 'gcloud' và xác thực chưa?")
#     print(f"Chạy: gcloud auth application-default login")
#     print(f"Lỗi gốc: {e}")
#     # Thoát nếu không thể khởi tạo
#     exit(1)

class GeminiJudge:
    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        self.model_name = model
        self.model = GenerativeModel(model_name=self.model_name)
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

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)

    async def judge(self, **kwargs):
        # Chuyển đổi 'content' thành 'parts' theo cú pháp của Vertex AI
        prompt_text = self.prompt_template.format(**kwargs)
        contents = [{"role": "user", "parts": [{"text": prompt_text}]}]

        if self.eval_type == "binary_text":
            response_text = await self._query_full_text(contents)
            return self.aggregate_score(response_text)
        else:
            logprobs = await self._logprob_probs(contents)
            return self.aggregate_score(logprobs)

    async def _logprob_probs(self, contents) -> dict:
        generation_config = GenerationConfig(
            max_output_tokens=1,     # Chỉ lấy 1 token
            temperature=0,           # Để kết quả mang tính quyết định
            response_logprobs=True,  # Bật logprobs cho token ĐÃ CHỌN
            logprobs=20,             # YêuCầu TOP 20 token thay thế
            seed=0                   # Đặt seed để kết quả có thể tái tạo
        )
        try:
            # Gọi API Vertex AI bằng generate_content_async
            response = await self.model.generate_content_async(
                contents=contents,
                generation_config=generation_config
            )
            
            # Phân tích cấu trúc trả về của Vertex AI
            # [0] đầu tiên vì chúng ta chỉ yêu cầu 1 token (max_output_tokens=1)
            logprobs_data = response.candidates[0].logprobs_result.token_logprobs[0]
            
            # logprobs_data.top_logprobs là một danh sách các đối tượng TokenLogprob
            top_logprobs = logprobs_data.top_logprobs

        except (Exception, IndexError, AttributeError) as e:
            print(f"⚠️ Lỗi khi gọi hoặc phân tích Vertex AI logprobs: {e}")
            return {}

        result = {}
        for el in top_logprobs:
            # el.token là chuỗi token (ví dụ: "YES", "NO", "10")
            # el.logprob là điểm log-probability (float, ví dụ: -0.01)
            # Cần .strip() để đảm bảo không có khoảng trắng thừa
            token = el.token.strip()
            if token: # Đảm bảo token không rỗng
                result[token] = float(math.exp(el.logprob))
        
        return result

    async def _query_full_text(self, contents: list) -> str:
        """
        CHANGED: Cập nhật để dùng self.model của Vertex AI và nhận 'contents'.
        """
        try:
            response = await self.model.generate_content_async(
                contents=contents,
                generation_config={"temperature": 0}
            )
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ Vertex AI full text error: {e}")
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