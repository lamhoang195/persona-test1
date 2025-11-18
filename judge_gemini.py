import math
from vertexai.generative_models import GenerativeModel, GenerationConfig
import vertexai

class GeminiJudge:
    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        assert eval_type in ["0_100", "0_10", "binary", "binary_text"], "eval_type must be either 0_100 or binary"
        self.model = GenerativeModel(model)
        self.eval_type = eval_type
        self.prompt_template = prompt_template

        if self.eval_type == "0_100":
            self.aggregate_score = self._aggregate_0_100_score
        elif self.eval_type == "0_10":
            self.aggregate_score = self._aggregate_0_10_score
        elif self.eval_type == "binary":
            self.aggregate_score = self._aggregate_binary_score
        elif self.eval_type == "binary_text":
            self.aggregate_score = self._aggregate_binary_text_score

    async def judge(self, **kwargs):
        prompt = self.prompt_template.format(**kwargs)
        if self.eval_type == "binary_text":
            response_text = await self.query_full_text(prompt)
            score = self.aggregate_score(response_text)
        else:
            probs = await self.logprob_probs(prompt)
            score = self.aggregate_score(probs)
        return score

    async def logprob_probs(self, prompt: str) -> dict:
        config = GenerationConfig(
            temperature=0,
            max_output_tokens=1,
            top_k=20,
            top_p=1,
            candidate_count=1
        )
        response = await self.model.generate_content_async(
            prompt,
            generation_config=config,
            stream=False
        )
        try:
            candidates = response.candidates
            logprob_map = candidates[0].token_logprobs[0]
        except (IndexError, AttributeError):
            return {}

        # Convert logprobs to token-prob dict
        result = {}
        for token, logprob in zip(logprob_map["tokens"], logprob_map["log_probs"]):
            result[token] = math.exp(logprob)
        return result

    async def query_full_text(self, prompt: str) -> str:
        response = await self.model.generate_content_async(prompt)
        try:
            return response.text
        except AttributeError:
            return ""

    def _aggregate_0_100_score(self, score: dict) -> float:
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if 0 <= int_key <= 100:
                sum_ += int_key * val
                total += val
        if total < 0.25:
            return None
        return sum_ / total

    def _aggregate_0_10_score(self, score: dict) -> float:
        if "REFUSAL" in score and score["REFUSAL"] > max(score.get(str(i), 0) for i in range(10)):
            return None
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if 0 <= int_key <= 9:
                sum_ += int_key * val
                total += val
        if total < 0.25:
            return None
        return sum_ / total

    def _aggregate_binary_score(self, score: dict) -> float:
        yes_prob = score.get("YES", 0.0)
        no_prob = score.get("NO", 0.0)
        refusal_prob = score.get("REFUSAL", 0.0)

        if refusal_prob > max(yes_prob, no_prob):
            return None
        denominator = yes_prob + no_prob
        if denominator < 0.25:
            return None
        return yes_prob / denominator

    def _aggregate_binary_text_score(self, response_text: str) -> float:
        if "<answer>REFUSAL</answer>" in response_text:
            return None
        elif "<answer>NO</answer>" in response_text:
            return 0
        elif "<answer>YES</answer>" in response_text:
            return 1
        return None

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)
