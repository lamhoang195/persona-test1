# judge_local.py
import json
import tempfile
import math
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

# optional import (will be used only on exception)
from huggingface_hub import hf_hub_download


class LocalJudge:
    def __init__(
        self,
        model_name: str,
        prompt_template: str,
        eval_type: str = "0_100",
        device: str = None,
    ):
        self.model_name = model_name
        assert eval_type in ["0_100"], "eval_type must be 0_100"
        self.eval_type = eval_type
        self.prompt_template = prompt_template

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Try to load config normally; if it errors (rope_scaling validation),
        # fetch config.json from HF cache, patch rope_scaling, overwrite, then retry.
        try:
            config = AutoConfig.from_pretrained(model_name)
        except Exception as e:
            # Attempt to download config.json from HF cache and patch it
            try:
                print(f"[LocalJudge] AutoConfig.load failed ({e}). Attempting to patch remote config.json...")
                # Download cached config.json (this returns the local cached path)
                cfg_path = hf_hub_download(repo_id=model_name, filename="config.json")
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)

                # Patch rope_scaling if present and in new Llama3 format
                if "rope_scaling" in cfg and isinstance(cfg["rope_scaling"], dict):
                    rs = cfg["rope_scaling"]
                    # If transformers expects {"type", "factor"} but we're given llama3-style fields,
                    # convert to minimal accepted format.
                    if "type" not in rs and "factor" in rs:
                        patched = {
                            "type": rs.get("rope_type", "linear"),
                            "factor": float(rs.get("factor", 1.0)),
                        }
                        cfg["rope_scaling"] = patched
                        print(f"[LocalJudge] Patched rope_scaling: {patched}")

                # Overwrite cached config.json (so AutoConfig.from_pretrained will read patched version)
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)

                # Retry loading config
                config = AutoConfig.from_pretrained(model_name)
                print("[LocalJudge] Successfully loaded patched config.")
            except Exception as e2:
                # If patching fails, re-raise original exception with context
                raise RuntimeError(f"Failed to patch/load config for {model_name}: {e2}") from e

        # Finally load the model
        # Use device_map="auto" + float16 to reduce memory footprint; remove if unsupported in your env.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            # If device_map puts model on multiple devices, no need to call .to(self.device).
            # But for single-device envs ensure model on the chosen device:
            if not hasattr(self.model, "is_loaded") or not getattr(self.model, "is_loaded"):
                self.model.to(self.device)
        except TypeError:
            # Fallback when device_map / torch_dtype not supported in environment
            self.model = AutoModelForCausalLM.from_pretrained(model_name, config=config).to(self.device)

        self.model.eval()

    def judge(self, **kwargs) -> float:
        prompt_text = self.prompt_template.format(**kwargs)
        response_probs = self.logprob_probs(prompt_text)
        score = self._aggregate_0_100_score(response_probs)
        return score

    def logprob_probs(self, prompt: str, top_k=150) -> dict:
        """Compute logprobs for next token using HuggingFace models."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)

        logits = out.logits[:, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)

        topk_vals, topk_idx = torch.topk(logprobs, k=top_k, dim=-1)

        result = {}
        for logprob, idx in zip(topk_vals[0], topk_idx[0]):
            token = self.tokenizer.decode([idx.item()]).strip()
            prob = torch.exp(logprob).item()
            result[token] = prob

        return result

    def _aggregate_0_100_score(self, score: dict) -> float:
        total_weight = 0
        weighted_sum = 0

        for token_str, prob in score.items():
            try:
                token_score = int(token_str)
            except ValueError:
                continue
            if token_score < 0 or token_score > 100:
                continue

            weighted_sum += token_score * prob
            total_weight += prob

        if total_weight == 0:
            return None
        return weighted_sum / total_weight

    def __call__(self, **kwargs):
        return self.judge(**kwargs)
