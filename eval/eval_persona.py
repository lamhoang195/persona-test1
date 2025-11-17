import os
import json
import fire
import torch
import random
import asyncio
import logging
import pandas as pd
import vertexai

from tqdm import tqdm
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from judge_gemini import GeminiJudge
from activation_steer import ActivationSteerer
from models.model_utils import ModelUtils
from models.adapter_finetune import load_model_basic, load_model
from eval.prompts import Prompts
from config_api import setup_credentials

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Set up credentials and environment
config = setup_credentials()

try:
    config = setup_credentials()
except RuntimeError as e:
    print(f"Lỗi credentials: {e}")
    print("Hãy chắc chắn bạn đã nạp Kaggle Secrets ở Cell 1.")
    exit(1) # Thoát nếu thiếu credentials

# 🔥 KHỞI TẠO VERTEX AI (SAU KHI ĐÃ CÓ CONFIG)
try:
    vertexai.init(
        project=config.vertex_project_id, 
        location=config.vertex_location
    )
    print("✅ Vertex AI đã khởi tạo thành công (từ main.py).")
except Exception as e:
    print(f"❌ Lỗi khi khởi tạo Vertex AI: {e}")
    raise e

# GỌI setup_environment() SAU KHI MỌI THỨ ĐÃ SẴN SÀNG
config.setup_environment()


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]
    
def sample(
        model, tokenizer, conversations,
        coef=0, vector=None, layer=None, 
        bs=20, top_p=1, max_tokens=1000, 
        temperature=1, min_tokens=1, 
        steering_type="response",
        lora_path=None
    ):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 1. Chuẩn bị prompts từ conversations
    prompts = []
    for messages in conversations:
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    # 2. Xử lý theo batch
    outputs = []
    for i in trange(0, len(prompts), bs, desc="Generating"):
        batch = prompts[i:i+bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        
        with ActivationSteerer(model, vector, coeff=coef, layer_idx=layer-1, positions=steering_type) if coef != 0 and vector is not None and layer is not None else torch.no_grad():
            output = model.generate(
                **tokenized_batch,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                use_cache=True
            )
        
        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output]
        outputs.extend(output)
    
    return prompts, outputs

class Question():
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], # Danh sách các câu paraphrase
            judge_prompts: dict,
            temperature: float = 1,
            system: str = None, 
            judge: str = "gemini-2.5-pro",
            judge_eval_type: str = "0_100", 
            **ignored_extra_args
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {
            metric: GeminiJudge(
                judge,
                prompt,
                eval_type=judge_eval_type if metric != "coherence" else "0_100"
            )
            for metric, prompt in judge_prompts.items()
        }
    
    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = [[dict(role='user', content=i)] for i in paraphrases]
        if self.system:
            conversations = [[dict(role='system', content=self.system)] + c for c in conversations]
        return paraphrases, conversations
    
    # Hàm sinh câu trả lời và chấm điểm
    async def eval(
            self, model, tokenizer, coef, 
            vector=None, layer=None, 
            max_tokens=1000, 
            n_per_question=1, 
            steering_type="response",
            lora_path=None 
        ):
        paraphrases, conversations = self.get_input(n_per_question)
        prompts, answers = sample(
            model, tokenizer, conversations,
            coef, vector, layer,
            temperature=self.temperature,
            max_tokens=max_tokens,
            steering_type=steering_type,
            lora_path=lora_path
        )
        
        df = pd.DataFrame([
            dict(question=question, prompt=prompt, answer=answer, question_id=self.id)
            for question, answer, prompt in zip(paraphrases, answers, prompts)
        ])
        for score, judge in self.judges.items():
            scores = await asyncio.gather(*[
                judge(question=question, answer=answer)
                for question, answer in zip(paraphrases, answers)
            ])
            df[score] = scores
        return df
    
    @staticmethod
    async def eval_batched(
        questions, model, tokenizer, coef, 
        vector=None, layer=None, 
        n_per_question=1, 
        max_concurrent_judges=2, 
        max_tokens=1000, 
        steering_type="response",
        lora_path=None,
        output_generate_path=None
    ):
        # 1. Gom toàn bộ paraphrases và conversations từ tất cả questions
        all_paraphrases = []
        all_conversations = []
        question_indices = []  # lưu chỉ số mapping để gán kết quả về sau

        for i, q in enumerate(questions):
            paraphrases, conversations = q.get_input(n_per_question) # Lấy questions được paraphrase và đóng gói theo định dạng conversation.
            all_paraphrases.extend(paraphrases)
            all_conversations.extend(conversations)
            question_indices.extend([i] * len(paraphrases)) # Câu hỏi nào tương ứng với paraphrase nào

        print(f"Generating {len(all_paraphrases)} responses...")

        # 2. Sinh câu trả lời (có hoặc không dùng steering)
        prompts, answers = sample(
            model, tokenizer, all_conversations,
            coef, vector, layer,
            temperature=questions[0].temperature, 
            max_tokens=max_tokens,
            steering_type=steering_type,
            lora_path=lora_path
        )

        # Lưu kết quả generate ra file ngay sau khi generate xong
        if output_generate_path:
            generate_output_path = output_generate_path.replace('.csv', '_generated_only.csv')
            os.makedirs(os.path.dirname(generate_output_path) if os.path.dirname(generate_output_path) else '.', exist_ok=True)
            
            # Tạo DataFrame với kết quả generate
            generate_df = pd.DataFrame({
                'question': all_paraphrases,
                'prompt': prompts,
                'answer': answers,
                'question_index': question_indices
            })
            generate_df.to_csv(generate_output_path, index=False)
            print(f"✅ Saved generated responses to: {generate_output_path}")

        # 3. Chuẩn bị khung dữ liệu cho từng câu hỏi
        question_dfs = [pd.DataFrame(columns=["question", "prompt", "answer", "question_id"]) for _ in questions]
        sample_pointers = [[] for _ in questions]  # để gán lại thứ tự sample tương ứng

        # Gán paraphrase, prompt, answer vào DataFrame tương ứng
        for idx, (q_text, prompt, answer, q_idx) in enumerate(zip(all_paraphrases, prompts, answers, question_indices)):
            question_dfs[q_idx].loc[len(question_dfs[q_idx])] = {
                "question": q_text,
                "prompt": prompt,
                "answer": answer,
                "question_id": questions[q_idx].id
            }
            sample_pointers[q_idx].append(idx) # Lưu thêm sample_pointers để biết index toàn cục (global) ứng với local row trong df

        # 4. Gom tất cả judge task
        all_tasks = []
        all_task_indices = []  # (question_idx, metric, row_idx)

        for q_idx, (q, df, pointers) in enumerate(zip(questions, question_dfs, sample_pointers)):
            for metric, judge in q.judges.items():
                for row_idx, global_idx in enumerate(pointers):
                    q_text = all_paraphrases[global_idx]
                    answer = answers[global_idx]
                    all_tasks.append((judge, q_text, answer))
                    all_task_indices.append((q_idx, metric, row_idx))

        # 5. Chạy judge song song có giới hạn số lượng
        print(f"Evaluating {len(all_tasks)} outputs (with max {max_concurrent_judges} concurrent requests per batch)...")
        all_results = [None] * len(all_tasks)
        
        # Chia tasks thành batches, mỗi batch có max_concurrent_judges tasks
        batch_size = max_concurrent_judges
        task_batches = [
            all_tasks[i:i + batch_size] 
            for i in range(0, len(all_tasks), batch_size)
        ]
        
        print(f"Divided into {len(task_batches)} batches ({batch_size} requests per batch)...")
        
        async def run_judge(task_idx, judge, q_text, answer):
            result = await judge(question=q_text, answer=answer)
            return task_idx, result

        # Xử lý từng batch, đợi 1 phút giữa các batch
        batch_start_idx = 0
        with tqdm(total=len(all_tasks), desc="Judge evaluations") as pbar:
            for batch_idx, batch_tasks in enumerate(task_batches):
                if batch_idx > 0:  # Không đợi trước batch đầu tiên
                    print(f"\n⏳ Waiting 60 seconds before batch {batch_idx + 1}/{len(task_batches)}...")
                    await asyncio.sleep(60)  # Đợi 1 phút
                
                print(f"Processing batch {batch_idx + 1}/{len(task_batches)} ({len(batch_tasks)} requests)...")
                
                # Tạo tasks cho batch hiện tại với đúng global index
                batch_task_coros = [
                    run_judge(
                        batch_start_idx + local_idx, judge, q_text, ans
                    ) for local_idx, (judge, q_text, ans) in enumerate(batch_tasks)
                ]
                
                # Xử lý batch này (chạy đồng thời max_concurrent_judges requests)
                batch_results = await asyncio.gather(*batch_task_coros, return_exceptions=True)
                
                # Debug: Kiểm tra số lượng kết quả
                if len(batch_results) != len(batch_tasks):
                    print(f"⚠️ Warning: Expected {len(batch_tasks)} results but got {len(batch_results)}")
                
                # Gán kết quả vào all_results
                for idx, result_or_exc in enumerate(batch_results):
                    task_idx = batch_start_idx + idx
                    if isinstance(result_or_exc, Exception):
                        print(f"⚠️ Exception in task {task_idx}: {result_or_exc}")
                        all_results[task_idx] = None
                    else:
                        try:
                            task_idx_from_result, result = result_or_exc
                            all_results[task_idx_from_result] = result
                        except (ValueError, TypeError) as e:
                            print(f"⚠️ Error unpacking result for task {task_idx}: {e}, result: {result_or_exc}")
                            all_results[task_idx] = None
                    pbar.update(1)  # Luôn update, bất kể có exception hay không
                
                batch_start_idx += len(batch_tasks)

        # 6. Gán kết quả chấm điểm vào từng DataFrame
        for task_idx, result in enumerate(all_results):
            q_idx, metric, row_idx = all_task_indices[task_idx]
            question_dfs[q_idx].loc[row_idx, metric] = result

        return question_dfs
    
def a_or_an(word):
    return "an" if word[0].lower() in "aeiou" else "a"

def load_persona_questions(
        trait, temperature=1, 
        persona_instructions_type=None, 
        assistant_name=None, 
        judge_model="gemini-2.5-pro", 
        eval_type="0_100", version="eval"
):
    trait_data = json.load(open(f"data_generation/trait_data_{version}/{trait}.json", "r"))
    judge_prompts = {}
    judge_prompts = {
        trait: trait_data["eval_prompt"],
        "coherence": Prompts[f"coherence_{eval_type}"]
    }
    raw_questions = trait_data["questions"]
    questions = []
    for i, question in enumerate(raw_questions):
        if persona_instructions_type:
            persona_instructions = [x[persona_instructions_type] for x in trait_data["instruction"]]
            for k, instruction in enumerate(persona_instructions):
                name = assistant_name or (trait if persona_instructions_type == "pos" else "helpful")
                system = f"You are {a_or_an(name)} {name} assistant. {instruction}"
                questions.append(
                    Question(
                        paraphrases=[question],
                        id=f"{trait}_{i}_{persona_instructions_type}_{k}",
                        judge_prompts=judge_prompts,
                        judge=judge_model,
                        temperature=temperature,
                        system=system,
                        judge_eval_type=eval_type
                    )
                )
        else:
            questions.append(
                Question(
                    paraphrases=[question],
                    id=f"{trait}_{i}",
                    judge_prompts=judge_prompts,
                    judge=judge_model,
                    temperature=temperature,
                    judge_eval_type=eval_type
                )
            )
    return questions

def main(model, trait, output_path, coef=0, vector_path=None, layer=None, steering_type="response", max_tokens=1000, n_per_question=2, batch_process=True, max_concurrent_judges=2, persona_instruction_type=None, assistant_name=None, judge_model="gemini-2.5-pro", version="extract", overwrite=False, output_generate_path=None):
    # Evaluate a model on all questions from the evaluation json file
    if os.path.exists(output_path) and not overwrite:
        print(f"Output path '{output_path}' already exists — skipping evaluation.")
        df = pd.read_csv(output_path)
        for trait in [trait , "coherence"]:
            print(f"{trait}:  {df[trait].mean():.2f} +- {df[trait].std():.2f}")
        return
    
    print(f"Output path: {output_path}\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    temperature = 0.0 if n_per_question == 1 else 1.0

    if coef != 0:
        model, tokenizer = load_model(model)
        lora_path = None
        vector = torch.load(vector_path, weights_only=False)[layer]
    else:
        model, tokenizer, lora_path = load_model_basic(model)
        vector = None
        
    questions = load_persona_questions(
        trait, temperature=temperature, 
        persona_instructions_type=persona_instruction_type, 
        assistant_name=assistant_name, 
        judge_model=judge_model, version=version
    )

    # Tự động tạo output_generate_path từ output_path nếu không được cung cấp
    if output_generate_path is None:
        output_generate_path = output_path.replace('.csv', '_generated_only.csv')
    
    if batch_process:
        print(f"Batch processing {len(questions)} '{trait}' questions...")
        outputs_list = asyncio.run(
            Question.eval_batched(
                questions, model, tokenizer,coef, 
                vector, layer, n_per_question, 
                max_concurrent_judges, max_tokens, 
                steering_type=steering_type,
                lora_path=lora_path,
                output_generate_path=output_generate_path
            )
        )
        outputs = pd.concat(outputs_list)
    else:
        outputs = []
        for question in tqdm(questions, desc=f"Processing {trait} questions"):
            outputs.append(
                asyncio.run(
                    question.eval(
                        model, tokenizer,coef, vector, 
                        layer, max_tokens, n_per_question, 
                        steering_type=steering_type,
                        lora_path=lora_path
                    )
                )
            )
        outputs = pd.concat(outputs)

    outputs.to_csv(output_path, index=False)
    print(output_path)
    for trait in [trait , "coherence"]:
        print(f"{trait}:  {outputs[trait].mean():.2f} +- {outputs[trait].std():.2f}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)