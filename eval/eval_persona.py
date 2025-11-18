import os
import json
import fire
from numpy import indices
import torch
import random
import asyncio
import logging
import pandas as pd
import vertexai

from tqdm import tqdm
from tqdm import trange

from judge_gemini import GeminiJudge
from activation_steer import ActivationSteerer
from models.adapter_finetune import load_model_basic, load_model
from eval.prompts import Prompts
from config_api import setup_credentials

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Set up credentials and environment
config = setup_credentials()
vertexai.init(
    project=config.vertex_project_id,
    location=config.vertex_location
)

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

        print(f"Generating {len(all_paraphrases)} responses in a single batch...")

        # 2. Sinh câu trả lời (có hoặc không dùng steering)
        prompts, answers = sample(
            model, tokenizer, all_conversations,
            coef, vector, layer,
            temperature=questions[0].temperature, 
            max_tokens=max_tokens,
            steering_type=steering_type,
            lora_path=lora_path
        )

        # Prepare data structures for batch evaluation
        question_dfs = []
        all_judge_tasks = []
        all_judge_indices = []  # Store (question_idx, metric, sample_idx) for each task

        print("Preparing judge evaluations tasks...")
        for i, question in enumerate(questions):
            indices = [j for j, idx in enumerate(question_indices) if idx == i]
            q_paraphrases = [all_paraphrases[j] for j in indices]
            q_prompts = [prompts[j] for j in indices]
            q_answers = [answers[j] for j in indices]

            # Create dataframe for this question
            df = pd.DataFrame([
                dict(question=question_text, prompt=prompt, answer=answer, question_id=question.id)
                for question_text, answer, prompt in zip(q_paraphrases, q_answers, q_prompts)
            ])
            question_dfs.append(df)
        
            # Collect all judge tasks
            for metric, judge in question.judges.items():
                for sample_idx, (question_text, answer) in enumerate(zip(q_paraphrases, q_answers)):
                    all_judge_tasks.append((judge, question_text, answer))
                    all_judge_indices.append((i, metric, sample_idx))

        # Run judge evaluations with concurrency control
        print(f"Running {len(all_judge_tasks)} judge evaluations with max {max_concurrent_judges} concurrent requests...")
        all_results = [None] * len(all_judge_tasks)  # Pre-allocate results array
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent_judges)
        
        async def run_with_semaphore(task_idx, judge, question_text, answer):
            async with semaphore:
                result = await judge(question=question_text, answer=answer)
                return task_idx, result
        
        # Create all tasks with semaphore control
        tasks = [run_with_semaphore(task_idx, judge, question_text, answer) 
                for task_idx, (judge, question_text, answer) in enumerate(all_judge_tasks)]
        
        # Process tasks in batches with progress bar
        with tqdm(total=len(tasks), desc="Judge evaluations") as pbar:
            for task in asyncio.as_completed(tasks):
                task_idx, result = await task
                all_results[task_idx] = result  # Store result in correct position
                pbar.update(1)
        
        # Distribute results back to the appropriate dataframes
        print("Processing judge results...")
        for task_idx, result in enumerate(all_results):
            question_idx, metric, sample_idx = all_judge_indices[task_idx]
            question_dfs[question_idx].loc[sample_idx, metric] = result
        
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

def main(model, trait, output_path, coef=0, vector_path=None, layer=None, steering_type="response", max_tokens=1000, n_per_question=2, batch_process=True, max_concurrent_judges=5, persona_instruction_type=None, assistant_name=None, judge_model="gemini-2.5-pro", version="extract", overwrite=False, output_generate_path=None):
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