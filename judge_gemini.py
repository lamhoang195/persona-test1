import os
import math
import asyncio
from config_api import setup_credentials
from google import genai
from google.genai.types import GenerateContentConfig
import vertexai

# Set up credentials and environment
config = setup_credentials()
vertexai.init(
    project=config.vertex_project_id,
    location=config.vertex_location
)

# Create genai client
client = genai.Client(vertexai=True, project=config.vertex_project_id, location=config.vertex_location)


class GeminiJudge:
    def __init__(self, model: str, prompt: str, eval_type: str = "0_100"):
        self.model = model
        self.prompt = prompt
        self.eval_type = eval_type
        
    async def judge(self, **kwargs):
        prompt_text = self.prompt.format(**kwargs)
        if self.eval_type == "binary_text":
            response_text = await self.query_full_text(prompt_text)
            score = self.aggregate_score(response_text)
        else:
            logprobs = await self.logprob_probs(prompt_text)
            score = self.aggregate_score(logprobs)
        return score

    async def logprob_probs(self, prompt_text: str) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model,
                contents=prompt_text,
                config=GenerateContentConfig(
                    max_output_tokens=1,
                    temperature=0,
                    response_logprobs=True,
                    logprobs=19,  # top_logprobs=19 like in original code
                )
            )
            
            result = {}
            if response.candidates and response.candidates[0].logprobs_result:
                logprobs_result = response.candidates[0].logprobs_result
                
                # Get chosen token (first token generated)
                if logprobs_result.chosen_candidates:
                    chosen = logprobs_result.chosen_candidates[0]
                    result[chosen.token] = float(math.exp(chosen.log_probability))
                
                # Get top alternative tokens
                if logprobs_result.top_candidates and len(logprobs_result.top_candidates) > 0:
                    top_alternatives = logprobs_result.top_candidates[0].candidates
                    for alt_token_info in top_alternatives:
                        token = alt_token_info.token
                        # Skip if already added (chosen token)
                        if token not in result:
                            result[token] = float(math.exp(alt_token_info.log_probability))
            
            return result
        except Exception as e:
            print(f"Error in logprob_probs: {e}")
            return {}
    
    async def query_full_text(self, prompt_text: str) -> str:
        """Requests a full text completion. Used for binary_text eval_type."""
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model,
                contents=prompt_text,
                config=GenerateContentConfig(
                    temperature=0,
                )
            )
            if response and response.text:
                return response.text
            return ""
        except Exception as e:
            print(f"Error in query_full_text: {e}")
            return ""

    def aggregate_score(self, score_input) -> float:
        """Aggregate score from either logprobs dict or text response."""
        if isinstance(score_input, dict):
            return self._aggregate_0_100_score(score_input)
        elif isinstance(score_input, str):
            # For binary_text eval_type, extract number from text
            try:
                # Try to extract a number from the response
                import re
                numbers = re.findall(r'\d+', score_input)
                if numbers:
                    score = int(numbers[0])
                    if 0 <= score <= 100:
                        return float(score)
            except:
                pass
            return None
        return None

    def _aggregate_0_100_score(self, score: dict) -> float:
        #   NOTE: we don't check for refusals explcitly. Instead we assume that
        #   if there's at least 0.25 total weight on numbers, it's not a refusal.
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
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)