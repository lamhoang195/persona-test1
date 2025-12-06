import math
import json
import os
import shutil
import tempfile
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download, snapshot_download

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
    config_file = None
    
    # Try local path first
    if os.path.exists(model_path):
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            config_file = config_path
    
    # If not local, try downloading from HuggingFace Hub
    if config_file is None:
        try:
            config_file = hf_hub_download(model_path, "config.json")
        except Exception as e:
            # If that fails, try snapshot_download as fallback
            try:
                local_dir = snapshot_download(model_path, ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth"])
                config_path = os.path.join(local_dir, 'config.json')
                if os.path.exists(config_path):
                    config_file = config_path
            except Exception:
                pass
    
    # Load and fix config
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Check if we need to fix rope_scaling
        needs_fix = False
        if 'rope_scaling' in config_dict and config_dict['rope_scaling'] is not None:
            rope_scaling = config_dict['rope_scaling']
            if isinstance(rope_scaling, dict) and ('rope_type' in rope_scaling or 'type' not in rope_scaling):
                needs_fix = True
        
        if needs_fix:
            config_dict = _fix_rope_scaling_config(config_dict)
            # Write fixed config to a temporary directory and load from there
            # Use temp directory context manager to ensure cleanup
            tmp_dir = tempfile.mkdtemp()
            try:
                temp_config_path = os.path.join(tmp_dir, 'config.json')
                with open(temp_config_path, 'w', encoding='utf-8') as tmp_file:
                    json.dump(config_dict, tmp_file, indent=2)
                # Load config from temp directory
                config = AutoConfig.from_pretrained(tmp_dir, trust_remote_code=True)
            finally:
                # Clean up temp directory
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass
            return config
        else:
            # No fix needed, load normally
            return AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    else:
        raise FileNotFoundError(
            f"Could not find config.json for model {model_path}. "
            f"Tried local path and HuggingFace Hub."
        )

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
