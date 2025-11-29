import math
from config_api import setup_credentials
from google import genai
from google.genai.types import GenerateContentConfig

# Set up credentials and environment
config = setup_credentials()

class GeminiJudge:
    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        self.model = model
        assert eval_type in ["0_100"], "eval_type must be 0_100"
        self.eval_type = eval_type
        self.prompt_template = prompt_template

        self.client = genai.Client(
            vertexai=True,
            project=config.vertex_project_id,
            location=config.vertex_location,
        )

    async def judge(self, **kwargs) -> float:
        """
        Return score 0-100 for the given prompt/answer pair
        """
        prompt_text = self.prompt_template.format(**kwargs)
        response = await self._generate_logprobs(prompt_text)
        score = self._aggregate_0_100_score(response)
        return score

    async def _generate_logprobs(self, prompt_text: str) -> dict:
        """
        Call Vertex AI to get logprobs for the model's first token.
        Return a dict mapping token -> probability.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt_text],
            config=GenerateContentConfig(
                max_output_tokens=5,
                temperature=0.0,
                response_logprobs=True,
                logprobs=19,  # số lượng top candidates để đánh giá
                seed=0,
            ),
        )

        # Parse logprobs_result
        # Đây là giả định dựa trên cấu trúc VertexAI GenAI
        result = {}
        try:
            top_candidates = response.candidates[0].logprobs_result.top_candidates[0].candidates
            for tc in top_candidates:
                token = getattr(tc, "token", str(tc))
                prob = math.exp(tc.log_probability)
                score = prob * 100
                result[token] = score
        except Exception:
            # fallback
            result = {}
        return result

    def _aggregate_0_100_score(self, score_dict: dict) -> float:
        """
        Convert logprob dict to a weighted score between 0-100
        """
        total_weight = 0
        weighted_sum = 0
        for token_str, prob in score_dict.items():
            try:
                token_score = int(token_str)
            except ValueError:
                continue
            if token_score < 0 or token_score > 100:
                continue
            weighted_sum += token_score * prob
            total_weight += prob
        if total_weight < 0.25:
            return None
        return weighted_sum / total_weight

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)
