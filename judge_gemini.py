import math
from config_api import setup_credentials
from google import genai
from google.genai.types import GenerateContentConfig

# Set up credentials and environment
config = setup_credentials()

class GeminiJudge:
    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        self.model = model
        assert eval_type in ["0_100"], "eval_type must be either 0_100"
        self.eval_type = eval_type

        if self.eval_type == "0_100":
            self.aggregate_score = self._aggregate_0_100_score
        else:
            raise ValueError(f"Invalid eval_type: {self.eval_type}")

        self.prompt_template = prompt_template
        # Khởi tạo Gemini client (dùng credentials đã setup ở trên)
        self.client = genai.Client()
        
    async def judge(self, **kwargs):
        messages = [dict(role='user', content=self.prompt_template.format(**kwargs))]
        if self.eval_type == "binary_text":
            response_text = await self.query_full_text(messages)
            score = self.aggregate_score(response_text) # aggregate_score is _aggregate_binary_text_score
        else:
            logprobs = await self.logprob_probs(messages)
            score = self.aggregate_score(logprobs) # aggregate_score is one of the other three
        return score

    async def logprob_probs(self, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        # Lấy nội dung prompt từ messages (giống cách bạn đang xây dựng ở judge)
        if not messages:
            return {}
        prompt = messages[-1].get("content", "")
        if not prompt:
            return {}

        # Gọi Gemini với logprobs (tương đương logprobs=True, top_logprobs=20, max_tokens=1, temperature=0)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1,
                response_logprobs=True,
                logprobs=19,  # số lượng top tokens
            ),
        )

        if not response.candidates:
            return {}

        candidate = response.candidates[0]
        logprobs_result = getattr(candidate, "logprobs_result", None)
        if not logprobs_result or not logprobs_result.top_candidates:
            return {}

        # Lấy top candidates của token đầu tiên (giống completion.choices[0].logprobs.content[0].top_logprobs)
        top_candidates = logprobs_result.top_candidates[0].candidates

        result = {}
        for el in top_candidates:
            # log_probability là log p(token), exponentiate để ra xác suất giống code cũ
            result[el.token] = float(math.exp(el.log_probability))

        return result
    
    async def query_full_text(self, messages) -> str:
        """Requests a full text completion. Used for binary_text eval_type."""
        completion = await openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            seed=0
        )
        try:
            return completion.choices[0].message.content
        except (IndexError, AttributeError):
            return ""

    def _aggregate_0_100_score(self, score: dict) -> float:
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val
        if total < 0.25:
            return None
        return sum_ / total

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)