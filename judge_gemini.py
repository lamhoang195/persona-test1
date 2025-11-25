import math
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
        config = GenerationConfig(
            temperature=0,
            max_output_tokens=1,
            response_logprobs=True,
            logprobs=19,
            seed=0
        )

        try:
            response = await self.model.generate_content_async(
                prompt_text,
                generation_config=config
            )
            
            if not response.candidates or not response.candidates[0].logprobs_result:
                return {}
                
            first_token_candidates = response.candidates[0].logprobs_result.top_candidates[0].candidates
            
        except (IndexError, AttributeError, Exception) as e:
            # Xử lý lỗi nếu API không trả về logprobs hoặc bị block
            print(f"Error fetching logprobs: {e}")
            return {}

        result = {}
        for candidate in first_token_candidates:
            token = candidate.token or candidate.text
            if token:
                result[token] = float(math.exp(candidate.log_probability))
        
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