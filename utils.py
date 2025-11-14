import json
from models.model_utils import ModelUtils
from models.model_utils_nsys import ModelUtilsNsys

def open_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_model(config):
    provider = config["model_info"]["provider"].lower()
    if provider == 'attn-hf':
        model = ModelUtils(config)
    elif provider == 'attn-hf-no-sys':
        model = ModelUtilsNsys(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model