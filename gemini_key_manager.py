import time
from google.generativeai import configure
from config_api import config

class GeminiKeyManager:
    def __init__(self, api_keys: list[str], daily_limit=250):
        self.api_keys = api_keys
        self.daily_limit = daily_limit # Giới hạn sử dụng hàng ngày cho mỗi key
        self.usage = [0 for _ in api_keys] # Số lần sử dụng hàng ngày cho mỗi key
        self.current_index = 0 # Chỉ số key hiện tại
        self.last_reset_time = time.time() # 
        self._activate_key(self.current_index)

    def _activate_key(self, index):
        self.current_index = index # Chuyển sang key có chỉ số index
        configure(api_key=self.api_keys[index]) # Cấu hình Gemini với key mới
        print(f"Switched to Gemini API key {index + 1}/{len(self.api_keys)}")

    def get_current_key(self):
        return self.api_keys[self.current_index]

    def increment(self):
        self.usage[self.current_index] += 1 # Phương thức này nên được gọi sau mỗi lần bạn sử dụng API thành công.
        if self.usage[self.current_index] >= self.daily_limit:
            self._rotate_key() # Kiểm tra xem key hiện tại đã đạt giới hạn sử dụng hàng ngày chưa

    # Xoay vòng key khi key hiện tại đạt giới hạn sử dụng hàng ngày
    def _rotate_key(self):
        for i in range(len(self.api_keys)):
            next_index = (self.current_index + 1 + i) % len(self.api_keys)
            if self.usage[next_index] < self.daily_limit:
                self._activate_key(next_index)
                return
        raise RuntimeError("All Gemini API keys exceeded daily limit.")

    # Kiểm tra xem đến lúc reset usage hàng ngày chưa
    def reset_daily_usage(self):
        now = time.time()
        if now - self.last_reset_time > 86400:  # 1 day
            self.usage = [0 for _ in self.api_keys]
            self.last_reset_time = now
            print("Daily usage reset.")

# ✅ Khởi tạo global GeminiKeyManager
gemini_key_manager = GeminiKeyManager(config.gemini_api_keys)