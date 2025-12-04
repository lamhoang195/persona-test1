import os
import warnings
from pathlib import Path

def load_env_file(env_path: str = ".env") -> None:
    env_file = Path(env_path)
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Nếu giá trị bắt đầu và kết thúc bằng cùng loại ngoặc kép, thì loại bỏ ngoặc ngoài cùng
                    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    os.environ[key] = value

class Config:
    """Configuration management for API keys and tokens."""
    
    def __init__(self):
        load_env_file()
        self._hf_token = None
        self._wandb_project = None
    
    @property
    def hf_token(self) -> str:
        """Get HuggingFace token from environment variables."""
        if self._hf_token is None:
            self._hf_token = os.environ.get('HF_TOKEN')
            if not self._hf_token:
                raise ValueError(
                    "HF_TOKEN not found in environment variables. "
                    "Please set it in your .env file or environment."
                )
        return self._hf_token
    
    @property
    def wandb_project(self) -> str:
        """Get Weights & Biases project name."""
        if self._wandb_project is None:
            self._wandb_project = os.environ.get('WANDB_PROJECT', 'persona-vectors')
        return self._wandb_project
    
    def setup_environment(self) -> None:
        os.environ['HF_TOKEN'] = self.hf_token
        os.environ['WANDB_PROJECT'] = self.wandb_project
    
    def validate_credentials(self) -> bool:
        try:
            _ = self.wandb_project
            _ = self.hf_token  
            return True
        except ValueError as e:
            warnings.warn(f"Credential validation failed: {e}")
            return False

# Global config instance
config = Config()

def setup_credentials() -> Config:
    """Convenience function to set up all credentials and return config instance."""
    config.setup_environment()
    if not config.validate_credentials():
        raise RuntimeError("Failed to validate required credentials")
    return config 