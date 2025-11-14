import torch
from .model import Model
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelUtilsNsys(Model):
    def __init__(self, config):
        super().__init__(config)
        self.name = config["model_info"]["name"]
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        
        # Kiểm tra training mode
        self.training_mode = config["params"].get("training_mode", False)

        # Sử dụng load_model thay vì viết lại
        model_path = config["model_info"]["model_path"]
        self.model, self.tokenizer = self.load_model(model_path)

        # Chỉ eval nếu không phải training mode
        if not self.training_mode:
            self.model.eval()
        else:
            self.model.train()

        self.top_k = 50
        self.top_p = None
        if config["params"]["important_heads"] == "all":
            attn_size = self.get_map_dime()
            self.important_heads = [[i, j] for i in range(attn_size[0])
                                            for j in range(attn_size[1])]
        else:
            self.important_heads = config["params"]["important_heads"]