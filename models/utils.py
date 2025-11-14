import torch
import torch.nn.functional as F

'''
Trích xuất attention scores của token cuối cùng từ attention maps của tất cả các layers.
Tối ưu hóa memory bằng cách chỉ giữ attention của token cuối cùng (token đầu tiên sinh ra).
'''
def get_last_attn(attn_map):
    for i, layer in enumerate(attn_map):
        attn_map[i] = layer[:, :, -1, :].unsqueeze(2)

    return attn_map

'''
Chọn token tiếp theo từ logits của model, hỗ trợ cả greedy sampling và top-k sampling
logits: output scores từ model (chưa qua softmax)
top_k: không dùng số lượng tokens có xác suất cao nhất để chọn
top_p: không dùng Nucleus sampling
'''
def sample_token(logits, top_k=None, top_p=None, temperature=1.0):
    # Chia logits cho temperature để điều chỉnh phân phối xác suất
    logits = logits / temperature

    if top_k is not None:
        top_k = min(top_k, logits.size(-1))  # Đảm bảo top_k không vượt quá kích thước vocabulary
        values, indices = torch.topk(logits, top_k)
        probs = F.softmax(values, dim=-1)
        next_token_id = indices[torch.multinomial(probs, 1)]

        return next_token_id

    return logits.argmax(dim=-1).squeeze()

