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
        self._openai_api_key = None
        self._gemini_api_keys = None
        #self._vertex_api_key = None
        self._hf_token = None
        self._wandb_project = None
        
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from environment variables."""
        if self._openai_api_key is None:
            self._openai_api_key = os.environ.get('OPENAI_API_KEY')
            # if not self._openai_api_key:
            #     raise ValueError(
            #         "OPENAI_API_KEY not found in environment variables. "
            #         "Please set it in your .env file or environment."
            #     )
        return self._openai_api_key

    @property
    def gemini_api_keys(self) -> list[str]:
        """List of Gemini API keys from .env (GOOGLE_API_KEYS)"""
        if self._gemini_api_keys is None:
            raw_keys = os.environ.get("GOOGLE_API_KEYS", "")
            # Parse: '"key1","key2","key3"' → ['key1', 'key2', 'key3']
            self._gemini_api_keys = [k.strip().strip('"').strip("'") for k in raw_keys.split(',') if k.strip()]
            # if not self._gemini_api_keys:
            #     raise ValueError("GOOGLE_API_KEYS not found or empty.")
        return self._gemini_api_keys
    
    # @property
    # def vertex_api_key(self) -> str:
    #     """Get Vertex AI API key from environment variables."""
    #     if self._vertex_api_key is None:
    #         self._vertex_api_key = os.environ.get('VERTEX_API_KEY')
    #         if not self._vertex_api_key:
    #             raise ValueError(
    #                 "VERTEX_API_KEY not found in environment variables. "
    #                 "Please set it in your .env file or environment."
    #             )
    #     return self._vertex_api_key
    
    @property
    def hf_token(self) -> str:
        """Get HuggingFace token from environment variables."""
        if self._hf_token is None:
            self._hf_token = os.environ.get('HF_TOKEN')
            # if not self._hf_token:
            #     raise ValueError(
            #         "HF_TOKEN not found in environment variables. "
            #         "Please set it in your .env file or environment."
            #     )
        return self._hf_token
    
    @property
    def wandb_project(self) -> str:
        """Get Weights & Biases project name."""
        if self._wandb_project is None:
            self._wandb_project = os.environ.get('WANDB_PROJECT', 'persona-vectors')
        return self._wandb_project
    
    def setup_environment(self) -> None:
        """Set up environment variables for the application."""
        # Set OpenAI API key in environment for libraries that expect it
        os.environ['OPENAI_API_KEY'] = self.openai_api_key

        # Set Gemini API key in environment for libraries that expect it
        os.environ['GOOGLE_API_KEY'] = self.gemini_api_keys[0]  # Use the first key by default

        # Set Vertex AI API key in environment
        #os.environ['VERTEX_API_KEY'] = self.vertex_api_key
        
        # Set HuggingFace token in environment
        os.environ['HF_TOKEN'] = self.hf_token
        
        # Set Weights & Biases project
        os.environ['WANDB_PROJECT'] = self.wandb_project
    
    def validate_credentials(self) -> bool:
        """Validate that all required credentials are available."""
        try:
            _ = self.openai_api_key
            _ = self.gemini_api_keys
            #_ = self.vertex_api_key
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