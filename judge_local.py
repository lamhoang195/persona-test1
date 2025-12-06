import math
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.utils import cached_file

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _fix_rope_scaling_config(config_dict):
    """Fix rope_scaling format to be compatible with transformers library."""
    if 'rope_scaling' in config_dict and config_dict['rope_scaling'] is not None:
        rope_scaling = config_dict['rope_scaling']
        if isinstance(rope_scaling, dict):
            # Check if it has the old format (rope_type, factor, etc.)
            if 'rope_type' in rope_scaling or 'type' not in rope_scaling:
                # Convert to expected format
                rope_type = rope_scaling.get('rope_type', rope_scaling.get('type', 'linear'))
                factor = rope_scaling.get('factor', 1.0)
                # Map rope_type to type (llama3 uses linear scaling)
                if rope_type == 'llama3' or rope_type == 'llama3linear':
                    type_str = 'linear'
                else:
                    type_str = rope_type if rope_type in ['linear', 'dynamic', 'yarn'] else 'linear'
                
                config_dict['rope_scaling'] = {
                    "type": type_str,
                    "factor": float(factor)
                }
    return config_dict

def _load_config_with_fix(model_path):
    """Load config and fix rope_scaling format if needed."""
    try:
        # Use cached_file to get config.json path (works for both local and hub models)
        try:
            config_file = cached_file(model_path, "config.json")
            if config_file and os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                config_dict = _fix_rope_scaling_config(config_dict)
                return AutoConfig.from_dict(config_dict, trust_remote_code=True)
        except Exception:
            # If cached_file fails, try direct path
            if os.path.exists(model_path):
                config_path = os.path.join(model_path, 'config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_dict = json.load(f)
                    config_dict = _fix_rope_scaling_config(config_dict)
                    return AutoConfig.from_dict(config_dict, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not pre-fix config from file, attempting direct load: {e}")
    
    # Fallback: try loading normally (this might still fail, but we handle it)
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Try to fix after loading if possible (shouldn't reach here if error occurs during from_pretrained)
        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            if isinstance(config.rope_scaling, dict) and ('rope_type' in config.rope_scaling or 'type' not in config.rope_scaling):
                rope_type = config.rope_scaling.get('rope_type', 'linear')
                factor = config.rope_scaling.get('factor', 1.0)
                type_str = 'linear' if rope_type in ['llama3', 'llama3linear'] else rope_type
                config.rope_scaling = {"type": type_str, "factor": float(factor)}
        return config
    except ValueError as ve:
        if 'rope_scaling' in str(ve):
            raise ValueError(f"Failed to fix rope_scaling. Please update the model's config.json manually. Error: {ve}")
        raise

class LocalJudge:
    def __init__(
        self,
        model_path: str | None = None,
        prompt_template: str | None = None,
        top_k: int = 50,
        **kwargs,
    ):
        if model_path is None:
            model_path = kwargs.pop("model_name", None)
        if model_path is None:
            raise ValueError("LocalJudge requires model_path or model_name.")
        if prompt_template is None:
            raise ValueError("LocalJudge requires a prompt_template.")

        self.model_path = model_path
        self.top_k = top_k
        self.prompt_template = prompt_template
        
        # Load and fix config before loading model
        config = _load_config_with_fix(model_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="eager"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, **kwargs):
        return self.judge(**kwargs)

    def judge(self, **kwargs):
        prompt = self.prompt_template.format(**kwargs)
        probs = self.logprob_probs(prompt)
        score = self.aggregate_0_100_score(probs)
        return score

    def logprob_probs(self, prompt: str) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = self.model(**inputs)

        logits = out.logits[:, -1, :]
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(logprobs, self.top_k, dim=-1)
        result = {}
        for val, idx in zip(topk_vals[0], topk_idx[0]):
            token_str = self.tokenizer.decode([idx.item()], clean_up_tokenization_spaces=False)
            token_str = token_str.strip()
            result[token_str] = float(math.exp(val.item()))
        return result

    def aggregate_0_100_score(self, scores: dict) -> float:
        total = 0
        sum_ = 0
        for token, prob in scores.items():
            try:
                int_token = int(token)
            except ValueError:
                continue
            if 0 <= int_token <= 100:
                sum_ += int_token * prob
                total += prob
        if total == 0:
            return None
        return sum_ / total
