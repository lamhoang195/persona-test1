import os
import json
import fire
import torch
import random
import asyncio
import logging
import pandas as pd

from tqdm import tqdm
from typing import List
from judge_gemini import GeminiJudge
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_steer import ActivationSteerer
from models.model_utils import ModelUtils
from models.adapter_finetune import load_model_basic, load_model
from eval.prompts import Prompts
from config_api import setup_credentials

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Set up credentials and environment
config = setup_credentials()

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]
    
def sample(
        model, tokenizer, conversations,
        coef=0, vector=None, layer=None, 
        bs=20, top_p=1, max_tokens=1024, 
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
            judge: str = "gemini-2.5-flash",
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
            max_tokens=1024, 
            n_per_question=100, 
            steering_type="response", 
        ):
        paraphrases, conversations = self.get_input(n_per_question)
        prompts, answers = sample(
            model, tokenizer, conversations,
            coef, vector, layer,
            temperature=self.temperature,
            max_tokens=max_tokens,
            steering_type=steering_type
        )
        df = pd.DataFrame([
            dict(question=question,prompt=prompt, answer=answer, question_id=self.id)
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
        questions: List["Question"], model, tokenizer, coef, 
        vector=None, layer=None, 
        n_per_question=100, 
        max_concurrent_judges=2, 
        max_tokens=1024, 
        steering_type="response"
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
            temperature=questions[0].temperature,  # giả định nhiệt độ giống nhau
            max_tokens=max_tokens,
            steering_type=steering_type
        )

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
        print(f"Evaluating {len(all_tasks)} outputs (with max {max_concurrent_judges} concurrent requests)...")
        all_results = [None] * len(all_tasks)
        semaphore = asyncio.Semaphore(max_concurrent_judges)

        async def run_with_semaphore(task_idx, judge, q_text, answer):
            async with semaphore:
                result = await judge(question=q_text, answer=answer)
                return task_idx, result

        tasks = [
            run_with_semaphore(idx, judge, q_text, ans)
            for idx, (judge, q_text, ans) in enumerate(all_tasks)
        ]

        with tqdm(total=len(tasks), desc="Judge evaluations") as pbar:
            for coro in asyncio.as_completed(tasks):
                task_idx, result = await coro
                all_results[task_idx] = result
                pbar.update(1)

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
        judge_model="gemini-2.5-flash", 
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
        if persona_instructions_type is not None:
            persona_instructions = [x[persona_instructions_type] for x in trait_data["instruction"]]
            for k, instruction in enumerate(persona_instructions):
                if assistant_name is None:
                    if persona_instructions_type == "pos":
                        assistant_name = trait
                    else:
                        assistant_name = "helpful"
                system = f"You are {a_or_an(assistant_name)} {assistant_name} assistant. {instruction}"
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

def main(model, trait, output_path, coef=0, vector_path=None, layer=None, steering_type="response", max_tokens=1024, n_per_question=10, batch_process=True, max_concurrent_judges=2, persona_instruction_type=None, assistant_name=None, judge_model="gemini-2.5-flash", version="extract", overwrite=False):
    # Evaluate a model on all questions from the evaluation json file
    if os.path.exists(output_path) and not overwrite:
        print(f"Output path '{output_path}' already exists — skipping evaluation.\n")
        df = pd.read_csv(output_path)
        for trait in [trait , "coherence"]:
            print(f"{trait}:  {df[trait].mean():.2f} +- {df[trait].std():.2f}")
        return
    
    print(f"Output path: {output_path}\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if n_per_question == 1:
        temperature = 0.0 # Nếu chỉ lấy 1 câu trả lời thì không cần sampling
    else:
        temperature = 1.0 # Ngược lại thì dùng sampling với temperature 1.0

    lora_path = None  # ✅ Khởi tạo trước

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
    
    if batch_process:
        print(f"Batch processing {len(questions)} '{trait}' questions...")
        outputs_list = asyncio.run(
            Question.eval_batched(
                questions, model, tokenizer,coef, 
                vector, layer, n_per_question, 
                max_concurrent_judges, max_tokens, 
                steering_type=steering_type
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
                        steering_type=steering_type
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