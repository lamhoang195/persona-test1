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

# ==== CHANGED: remove GeminiJudge ====
from judge_local_model import LocalJudge  
# (file judge_local.py chứa LocalJudge bạn đã có)

from activation_steer import ActivationSteerer
from models.model_utils import ModelUtils
from models.adapter_finetune import load_model_basic, load_model
from eval.prompts import Prompts
from config_api import setup_credentials

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

config = setup_credentials()

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]

# --------------------------------------------------------
# GENERATION (KHÔNG THAY ĐỔI)
# --------------------------------------------------------
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

    prompts = []
    for messages in conversations:
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    outputs = []
    for i in trange(0, len(prompts), bs, desc="Generating"):
        batch = prompts[i:i+bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}

        with ActivationSteerer(model, vector, coeff=coef, layer_idx=layer-1, positions=steering_type) \
            if coef != 0 and vector is not None and layer is not None else torch.no_grad():

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

# ======================================================================
# === QUESTION CLASS ===================================================
# ======================================================================

class Question():
    def __init__(
            self,
            id: str,
            paraphrases: list[str],
            judge_prompts: dict,
            temperature: float = 1,
            system: str = None,
            judge: str = "local",
            judge_eval_type: str = "0_100",
            **ignored_extra_args
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system

        # ==== CHANGED: Using LocalJudge ====
        self.judges = {
            metric: LocalJudge(
                model_path=judge,
                prompt_template=prompt,
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
        all_paraphrases = []
        all_conversations = []
        question_indices = []

        for i, q in enumerate(questions):
            paraphrases, conversations = q.get_input(n_per_question)
            all_paraphrases.extend(paraphrases)
            all_conversations.extend(conversations)
            question_indices.extend([i] * len(paraphrases))

        print(f"Generating {len(all_paraphrases)} responses...")

        prompts, answers = sample(
            model, tokenizer, all_conversations,
            coef, vector, layer,
            temperature=questions[0].temperature,
            max_tokens=max_tokens,
            steering_type=steering_type,
            lora_path=lora_path
        )

        # SAVE generated outputs
        if output_generate_path:
            gen_path = output_generate_path
            os.makedirs(os.path.dirname(gen_path), exist_ok=True)
            pd.DataFrame({
                'question': all_paraphrases,
                'prompt': prompts,
                'answer': answers,
                'question_index': question_indices
            }).to_csv(gen_path, index=False)
            print(f"Saved generated responses to: {gen_path}")

        question_dfs = [pd.DataFrame(columns=["question", "prompt", "answer", "question_id"]) for _ in questions]
        sample_pointers = [[] for _ in questions]

        for idx, (q_text, prompt, answer, q_idx) in enumerate(zip(all_paraphrases, prompts, answers, question_indices)):
            question_dfs[q_idx].loc[len(question_dfs[q_idx])] = {
                "question": q_text,
                "prompt": prompt,
                "answer": answer,
                "question_id": questions[q_idx].id
            }
            sample_pointers[q_idx].append(idx)

        # Collect judge tasks
        all_tasks = []
        all_indices = []
        for q_idx, (q, df, pointers) in enumerate(zip(questions, question_dfs, sample_pointers)):
            for metric, judge in q.judges.items():
                for row_idx, global_idx in enumerate(pointers):
                    q_text = all_paraphrases[global_idx]
                    answer = answers[global_idx]
                    all_tasks.append((judge, q_text, answer))
                    all_indices.append((q_idx, metric, row_idx))

        print(f"Evaluating {len(all_tasks)} outputs...")

        batch_size = max_concurrent_judges
        task_batches = [all_tasks[i:i + batch_size]
                        for i in range(0, len(all_tasks), batch_size)]

        all_results = [None] * len(all_tasks)
        batch_start = 0

        with tqdm(total=len(all_tasks)) as pbar:
            for batch_idx, batch in enumerate(task_batches):
                coros = [
                    judge(question=q, answer=a)
                    for (judge, q, a) in batch
                ]

                batch_res = await asyncio.gather(*coros, return_exceptions=True)

                for i, r in enumerate(batch_res):
                    all_results[batch_start + i] = None if isinstance(r, Exception) else r
                    pbar.update(1)

                batch_start += len(batch)

        # assign results
        for task_idx, result in enumerate(all_results):
            q_idx, metric, row_idx = all_indices[task_idx]
            question_dfs[q_idx].loc[row_idx, metric] = result

        return question_dfs


# ======================================================================
# ==== LOAD QUESTIONS ==================================================
# ======================================================================

def a_or_an(word): return "an" if word[0].lower() in "aeiou" else "a"

def load_persona_questions(
        trait, temperature=1,
        persona_instructions_type=None,
        assistant_name=None,
        judge_model="local",
        eval_type="0_100", version="eval"
):
    trait_data = json.load(open(f"data_generation/trait_data_{version}/{trait}.json"))

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

# ======================================================================
# ==== MAIN =============================================================
# ======================================================================

def main(model, trait, output_path, coef=0, vector_path=None, layer=None,
         steering_type="response", max_tokens=1000, n_per_question=2,
         batch_process=True, max_concurrent_judges=2, persona_instruction_type=None,
         assistant_name=None, judge_model="local", version="extract",
         overwrite=False, output_generate_path=None):

    if os.path.exists(output_path) and not overwrite:
        print(f"Output path '{output_path}' exists — skipping.")
        df = pd.read_csv(output_path)
        for t in [trait, "coherence"]:
            print(f"{t}: {df[t].mean():.2f} +- {df[t].std():.2f}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    temperature = 0.0 if n_per_question == 1 else 1.0

    if coef != 0:
        model, tokenizer = load_model(model)
        vector = torch.load(vector_path, weights_only=False)[layer]
        lora_path = None
    else:
        model, tokenizer, lora_path = load_model_basic(model)
        vector = None

    questions = load_persona_questions(
        trait, temperature=temperature,
        persona_instructions_type=persona_instruction_type,
        assistant_name=assistant_name,
        judge_model=judge_model,
        version=version
    )

    if output_generate_path is None:
        output_generate_path = output_path.replace(".csv", "_generated.csv")

    if batch_process:
        print(f"Batch processing {len(questions)} questions...")
        outputs_list = asyncio.run(
            Question.eval_batched(
                questions, model, tokenizer, coef,
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
        for q in tqdm(questions):
            outputs.append(
                asyncio.run(q.eval(
                    model, tokenizer, coef,
                    vector, layer, max_tokens,
                    n_per_question,
                    steering_type=steering_type,
                    lora_path=lora_path
                ))
            )
        outputs = pd.concat(outputs)

    outputs.to_csv(output_path, index=False)
    print("Saved to:", output_path)

    for t in [trait, "coherence"]:
        print(f"{t}: {outputs[t].mean():.2f} +- {outputs[t].std():.2f}")


if __name__ == "__main__":
    fire.Fire(main)
