import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

class LocalJudge:
    # Class-level cache để chia sẻ model/tokenizer giữa các instances
    _model_cache = {}
    _tokenizer_cache = {}
    
    def __init__(
        self,
        model_name: str,
        prompt_template: str,
        eval_type: str = "0_100",
        device: str = None,
        top_k: int = 150
    ):
        self.model_name = model_name
        assert eval_type in ["0_100"]
        self.eval_type = eval_type
        self.prompt_template = prompt_template
        self.top_k = top_k

        # Dùng cache để tránh load model nhiều lần
        if model_name not in self._model_cache:
            print(f"Loading model {model_name} (first time)...")
            
            # Tokenizer luôn dùng CPU, fallback về slow nếu fast không có
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            except (ValueError, OSError, ImportError) as e:
                error_msg = str(e).lower()
                if "sentencepiece" in error_msg or "Cannot instantiate" in str(e) or "LlamaTokenizer requires" in str(e):
                    print(f"Warning: Fast tokenizer not available, trying slow tokenizer for {model_name}")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                    except ImportError as import_err:
                        if "sentencepiece" in str(import_err).lower():
                            raise ImportError(
                                f"LlamaTokenizer requires the SentencePiece library. "
                                f"Please install it with: pip install sentencepiece\n"
                                f"Original error: {import_err}"
                            ) from import_err
                        raise
                else:
                    raise
            
            # Load config, xử lý lỗi rope_scaling nếu có
            try:
                config = AutoConfig.from_pretrained(model_name)
            except ValueError as e:
                if "rope_scaling" in str(e):
                    # Fix rope_scaling format bằng cách load config_dict và sửa
                    import json
                    from pathlib import Path
                    from huggingface_hub import hf_hub_download
                    
                    try:
                        config_path = hf_hub_download(
                            repo_id=model_name,
                            filename="config.json"
                        )
                    except Exception:
                        config_path = Path(model_name) / "config.json"
                    
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    
                    # Fix rope_scaling format
                    if "rope_scaling" in config_dict and isinstance(config_dict["rope_scaling"], dict):
                        rs = config_dict["rope_scaling"]
                        factor = rs.get("factor", 1.0)
                        rope_type = rs.get("rope_type", "linear")
                        
                        # Map rope_type to transformers format
                        if "llama3" in str(rope_type).lower():
                            type_str = "llama3"
                        else:
                            type_str = "linear"
                        
                        config_dict["rope_scaling"] = {"type": type_str, "factor": float(factor)}
                    
                    config = AutoConfig.from_dict(config_dict)
                else:
                    raise
            
            # Multi-GPU auto split
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                dtype=torch.float16,
                device_map="auto",
            )
            model.eval()
            
            self._model_cache[model_name] = model
            self._tokenizer_cache[model_name] = tokenizer
            
            print(f"✓ Model {model_name} loaded (cached for reuse). Device map:", model.hf_device_map)
        
        self.model = self._model_cache[model_name]
        self.tokenizer = self._tokenizer_cache[model_name]

    def judge(self, **kwargs) -> float:
        prompt_text = self.prompt_template.format(**kwargs)
        response_probs = self.logprob_probs(prompt_text)
        return self._aggregate_0_100_score(response_probs)

    def logprob_probs(self, prompt: str) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # ❗Không .to(device) khi dùng device_map="auto"

        with torch.no_grad():
            out = self.model(**inputs)

        logits = out.logits[:, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)
        vals, idxs = torch.topk(logprobs, k=self.top_k, dim=-1)

        result = {}
        for logprob, idx in zip(vals[0], idxs[0]):
            decoded = self.tokenizer.decode([idx.item()])
            prob = torch.exp(logprob).item()

            match = re.findall(r"-?\d+", decoded)
            if len(match) == 1:
                result[match[0]] = prob

        return result

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

        if total == 0:
            return None
        return sum_ / total

    def __call__(self, **kwargs):
        return self.judge(**kwargs)
