import os
import json
import fire
import torch
import random
import asyncio
import logging
import pandas as pd

from tqdm import tqdm
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed

from judge_local import LocalJudge
from activation_steer import ActivationSteerer
from models.model_utils import ModelUtils
from models.adapter_finetune import load_model_basic, load_model
from eval.prompts import Prompts
from config_api import load_env_file

logging.getLogger("httpx").setLevel(logging.ERROR)
load_env_file()

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]

def sample_steering(
    model, tokenizer, conversations,
    vector, layer, coef,
    bs=10, top_p=1, max_tokens=1000, temperature=1,
    min_tokens=1, steering_type="response"
):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompts = []
    for messages in conversations:
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    outputs = []
    for i in trange(0, len(prompts), bs, desc="Generating"):
        batch = prompts[i:i+bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}

        use_steering = (
            coef != 0
            and vector is not None
            and layer is not None
        )
        context = (
            ActivationSteerer(
                model,
                steering_vector=vector,
                coeff=coef,
                layer_idx=layer-1,
                positions=steering_type,
            )
            if use_steering else torch.no_grad()
        )
        with context:
            output = model.generate(
                **tokenized_batch,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                use_cache=True,
            )

        prompt_len = tokenized_batch["input_ids"].shape[1]
        decoded = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
            for o in output
        ]
        outputs.extend(decoded)
    return prompts, outputs

class Question():
    def __init__(
        self,
        id: str,
        paraphrases: list[str],
        judge_prompts: dict,
        temperature: float = 1,
        system: str = None,
        judge_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system

        self.judges = {}
        for metric, template in judge_prompts.items():
            self.judges[metric] = LocalJudge(
                judge_model=judge_model,
                prompt_template=template,
                top_k=20
            )

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = [[dict(role='user', content=i)] for i in paraphrases]
        if self.system:
            conversations = [
                [dict(role='system', content=self.system)] + c
                for c in conversations
            ]
        return paraphrases, conversations

    @staticmethod
    def eval_batched(
        questions, model, tokenizer, coef,
        vector=None, layer=None,
        n_per_question=10,
        max_concurrent_judges=10,
        max_tokens=1000,
        steering_type="response"
    ):
        all_paraphrases = []
        all_conversations = []
        question_indices = []
        for i, question in enumerate(questions):
            paraphrases, conversations = question.get_input(n_per_question)
            all_paraphrases.extend(paraphrases)
            all_conversations.extend(conversations)
            question_indices.extend([i] * len(paraphrases))

        print(f"Generating {len(all_paraphrases)} responses in a single batch...")

        prompts, answers = sample_steering(
            model, tokenizer, all_conversations,
            vector, layer, coef,
            temperature=questions[0].temperature,
            min_tokens=1, max_tokens=max_tokens,
            steering_type=steering_type
        )

        question_dfs = []
        all_judge_tasks = []
        all_judge_indices = []

        for i, question in enumerate(questions):
            indices = [j for j, idx in enumerate(question_indices) if idx == i]
            q_paraphrases = [all_paraphrases[j] for j in indices]
            q_prompts = [prompts[j] for j in indices]
            q_answers = [answers[j] for j in indices]

            df = pd.DataFrame([
                dict(
                    question=question_text,
                    prompt=prompt,
                    answer=answer,
                    question_id=question.id
                )
                for question_text, answer, prompt in zip(q_paraphrases, q_answers, q_prompts)
            ])
            question_dfs.append(df)

            for metric, judge in question.judges.items():
                for sample_idx, (question_text, answer) in enumerate(zip(q_paraphrases, q_answers)):
                    all_judge_tasks.append((judge, question_text, answer))
                    all_judge_indices.append((i, metric, sample_idx))

        print(f"Running {len(all_judge_tasks)} judge evaluation with max {max_concurrent_judges} concurrent judges...")

        # Sync judge loop
        all_results = [None] * len(all_judge_tasks)
        def run_judge(task_idx, judge, q_text, ans):
            """Helper function to run judge and return index + result"""
            try:
                score = judge(question=q_text, answer=ans)
                return task_idx, score
            except Exception as e:
                print(f"Judge error at task {task_idx}: {e}")
                return task_idx, None
        
        with ThreadPoolExecutor(max_workers=max_concurrent_judges) as executor:
            # Submit all tasks
            futures = {
                executor.submit(run_judge, idx, judge, q_text, ans): idx
                for idx, (judge, q_text, ans) in enumerate(all_judge_tasks)
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(futures), desc="Judge evaluations") as pbar:
                for future in as_completed(futures):
                    task_idx, score = future.result()
                    all_results[task_idx] = score
                    pbar.update(1)

        # Put results back into dataframes
        print("Processing judge results...")
        for idx, score in enumerate(all_results):
            qi, metric, si = all_judge_indices[idx]
            question_dfs[qi].loc[si, metric] = score

        return question_dfs

def a_or_an(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def load_persona_questions(
    trait, temperature=1,
    persona_instructions_type=None,
    assistant_name=None,
    judge_model="meta-llama/Llama-3.1-8B-Instruct",
    version="extract"
):
    trait_data = json.load(
        open(f"data_generation/trait_data_{version}/{trait}.json", "r")
    )
    judge_prompts = {
        trait: trait_data["eval_prompt"],
        "coherence": Prompts[f"coherence_0_100"]
    }
    raw_questions = trait_data["questions"]
    questions = []
    for i, question in enumerate(raw_questions):
        persona_instructions = [
            x[persona_instructions_type]
            for x in trait_data["instruction"]
        ]
        for k, instruction in enumerate(persona_instructions):
            if assistant_name is None:
                assistant_name = trait if persona_instructions_type == "pos" else "helpful"
            system = f"You are {a_or_an(assistant_name)} {assistant_name} assistant. {instruction}"
            questions.append(
                Question(
                    paraphrases=[question],
                    id=f"{trait}_{i}_{persona_instructions_type}_{k}",
                    judge_prompts=judge_prompts,
                    judge_model=judge_model,
                    temperature=temperature,
                    system=system
                )
            )
    return questions

def main(
    model, trait, output_path, coef=0,
    vector_path=None, layer=None,
    steering_type="response",
    max_tokens=1000, n_per_question=10,
    batch_process=True, max_concurrent_judges=10,
    persona_instruction_type=None, assistant_name=None,
    judge_model="meta-llama/Llama-3.1-8B-Instruct",
    version="extract", overwrite=False
):
    if os.path.exists(output_path) and not overwrite:
        print(f"Output exists — skipping.")
        df = pd.read_csv(output_path)
        for t in [trait, "coherence"]:
            print(f"{t}: {df[t].mean():.2f} +- {df[t].std():.2f}")
        return

    print(f"Output path: {output_path}\n")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    temperature = 0.0 if n_per_question == 1 else 1.0

    print(f"Loading model from {model}...")
    if coef != 0:
        model, tokenizer = load_model(model)
        vector = torch.load(vector_path, weights_only=False)[layer]
    else:
        model, tokenizer, _ = load_model_basic(model)
        vector = None
    print(f"Model loaded successfully")

    print(f"Loading judge model from {judge_model}...")
    questions = load_persona_questions(
        trait,
        temperature=temperature,
        persona_instructions_type=persona_instruction_type,
        assistant_name=assistant_name,
        judge_model=judge_model,
        version=version
    )
    print(f"Judge model loaded and questions prepared.\n")

    if batch_process:
        outputs_list = Question.eval_batched(
            questions, model, tokenizer, coef,
            vector, layer, n_per_question,
            max_concurrent_judges, max_tokens,
            steering_type=steering_type
        )
        outputs = pd.concat(outputs_list, ignore_index=True)

    else:
        raise NotImplementedError("Non-batch mode removed for LocalJudge")

    outputs.to_csv(output_path, index=False)
    print(output_path)

    for t in [trait, "coherence"]:
        print(f"{t}: {outputs[t].mean():.2f} +- {outputs[t].std():.2f}")

if __name__ == "__main__":
    fire.Fire(main)
