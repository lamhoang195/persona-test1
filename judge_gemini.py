import math
from config_api import setup_credentials
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Set up credentials and environment
config = setup_credentials()

class GeminiJudge:
    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        self.model = model
        assert eval_type in ["0_100"], "eval_type must be 0_100"
        self.eval_type = eval_type
        self.prompt_template = prompt_template
        
        # Dùng GenerativeModel thay vì genai.Client
        self.client = GenerativeModel(model_name=model)

    async def judge(self, **kwargs) -> float:
        prompt_text = self.prompt_template.format(**kwargs)
        response = await self.logprob_prob(prompt_text)
        score = self._aggregate_0_100_score(response)
        return score

    async def logprob_prob(self, prompt_text: str) -> dict:
        # Dùng generate_content với GenerationConfig
        response = self.client.generate_content(
            prompt_text,
            generation_config=GenerationConfig(
                max_output_tokens=1,
                temperature=0,
                candidate_count=0,
                logprobs=19,
                response_logprobs=True,
                top_k=10
            ),
        )

        result = {}
        
        try:
            # Cấu trúc giống code của bạn
            top_candidates = response.candidates[0].logprobs_result.top_candidates[0].candidates
        except Exception as e:
            print(f"Không lấy được logprobs. Error: {type(e).__name__}: {e}")
            top_candidates = []

        # Chuyển logprob -> probability
        for c in top_candidates:
            result[c.token] = math.exp(c.log_probability)

        # In ra
        print(result)
        return result

    def _aggregate_0_100_score(self, score_dict: dict) -> float:
        """
        Convert logprob dict to a weighted score between 0-100
        """
        if score_dict is None or len(score_dict) == 0:
            return None
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
