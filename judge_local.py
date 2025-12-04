# judge_local.py
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig
)
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
        assert eval_type in ["0_100"]
        self.eval_type = eval_type
        self.prompt_template = prompt_template

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # --- FIX: hard override rope_scaling to linear ---
        config = self._load_patched_config(model_name)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    def _load_patched_config(self, model_name):
        """Load config but force rope_scaling into a Transformers-compatible format."""
        try:
            return AutoConfig.from_pretrained(model_name)
        except Exception as e:
            print(f"[LocalJudge] Config load failed: {e}")
            print("[LocalJudge] Patching config.json…")

            cfg_path = hf_hub_download(repo_id=model_name, filename="config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            # Force patch (ignore all original rope fields)
            cfg["rope_scaling"] = {
                "type": "linear",
                "factor": float(cfg.get("rope_scaling", {}).get("factor", 8.0)),
            }

            # Remove any incompatible fields
            if "rope_type" in cfg:
                del cfg["rope_type"]

            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            print("[LocalJudge] rope_scaling forced to:", cfg["rope_scaling"])
            return AutoConfig.from_pretrained(model_name)

    def judge(self, **kwargs) -> float:
        prompt_text = self.prompt_template.format(**kwargs)
        response_probs = self.logprob_probs(prompt_text)
        return self._aggregate_0_100_score(response_probs)

    def logprob_probs(self, prompt: str, top_k=150) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)

        logits = out.logits[:, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)

        vals, idxs = torch.topk(logprobs, k=top_k, dim=-1)

        result = {}
        for logprob, idx in zip(vals[0], idxs[0]):
            token = self.tokenizer.decode([idx.item()]).strip()
            prob = torch.exp(logprob).item()
            result[token] = prob

        return result

    def _aggregate_0_100_score(self, score_dict: dict):
        total_w, weighted = 0, 0
        for tok, prob in score_dict.items():
            try:
                sc = int(tok)
                if 0 <= sc <= 100:
                    total_w += prob
                    weighted += prob * sc
            except ValueError:
                continue

        return weighted / total_w if total_w > 0 else None

    def __call__(self, **kwargs):
        return self.judge(**kwargs)
