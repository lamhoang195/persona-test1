import math
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google import genai
from google.genai.types import GenerateContentConfig

class GeminiJudge:
    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        self.model_name = model
        #self.model = GenerativeModel(model_name=self.model_name)
        self.client = genai.Client(vertexai=True)
        self.prompt_template = prompt_template
        assert eval_type in ["0_100", "0_10", "binary", "binary_text"], f"Unsupported eval_type: {eval_type}"
        self.eval_type = eval_type

        if self.eval_type == "0_100":
            self.aggregate_score = self._aggregate_0_100_score
    
    async def aclose(self):
        try:
            await self.client.aio.close()
        except Exception as e:
            print("⚠️ Error closing Gemini client:", e)

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)

    async def judge(self, **kwargs):
        # Format prompt
        prompt_content = self.prompt_template.format(**kwargs)
        
        if self.eval_type == "binary_text":
            response_text = await self._query_full_text(prompt_content)
            score = self.aggregate_score(response_text)
        else:
            logprobs = await self._logprob_probs(prompt_content)
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

        # Format content đúng chuẩn GenAI
        contents = [{
            "role": "user",
            "parts": [{"text": prompt_text}]
        }]

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            if not response.candidates or not response.candidates[0].logprobs_result:
                return {}

            lp = response.candidates[0].logprobs_result

            # candidate 0 → token đầu tiên
            first_token_cands = lp.top_candidates[0].candidates

        except Exception as e:
            print("Error in logprobs:", e)
            return {}

        probs = {}
        for cand in first_token_cands:
            token = cand.token.strip()
            prob = math.exp(cand.log_probability)

            # CHỈ LẤY TOKEN LÀ CHUỖI SỐ
            if token.isdigit():
                probs[token] = prob

        return probs

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
        if total < 0.25:
            return None
        return sum_ / total