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
    bs=2, top_p=1, max_tokens=1000, temperature=1,
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
        judge: str = "mistralai/Mistral-7B-Instruct-v0.3",
        judge_eval_type: str = "0_100",
        **ignored_extra_args
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system

        self.judges = {}
        for metric, template in judge_prompts.items():
            self.judges[metric] = LocalJudge(
                model_name=judge,
                prompt_template=template,
                eval_type=judge_eval_type,
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
    async def eval_batched(
        questions, model, tokenizer, coef,
        vector=None, layer=None,
        n_per_question=2,
        max_concurrent_judges=5,
        max_tokens=1000,
        steering_type="response"
    ):
        # Collect all data
        all_paraphrases = []
        all_conversations = []
        question_indices = []

        for i, q in enumerate(questions):
            paras, convs = q.get_input(n_per_question)
            all_paraphrases.extend(paras)
            all_conversations.extend(convs)
            question_indices.extend([i] * len(paras))

        print(f"Generating {len(all_paraphrases)} responses...")

        prompts, answers = sample_steering(
            model, tokenizer, all_conversations,
            vector, layer, coef,
            temperature=questions[0].temperature,
            min_tokens=1, max_tokens=max_tokens,
            steering_type=steering_type
        )

        # Prepare containers
        question_dfs = []
        all_tasks = []
        all_indices = []

        # Build judge tasks
        for qi, question in enumerate(questions):
            indices = [j for j, idx in enumerate(question_indices) if idx == qi]
            q_paras = [all_paraphrases[j] for j in indices]
            q_prompts = [prompts[j] for j in indices]
            q_answers = [answers[j] for j in indices]

            df = pd.DataFrame([
                dict(
                    question=q_text,
                    prompt=pr,
                    answer=ans,
                    question_id=question.id
                )
                for q_text, ans, pr in zip(q_paras, q_answers, q_prompts)
            ])
            question_dfs.append(df)

            # Create judge tasks
            for metric, judge in question.judges.items():
                for sample_idx, (q_text, ans) in enumerate(zip(q_paras, q_answers)):
                    all_tasks.append((judge, q_text, ans))
                    all_indices.append((qi, metric, sample_idx))

        print(f"Running {len(all_tasks)} judge evaluations...")

        # Sync judge loop
        all_results = [None] * len(all_tasks)
        with tqdm(total=len(all_tasks), desc="Judge evaluations") as pbar:
            for idx, (judge, q_text, ans) in enumerate(all_tasks):
                # ---- FIX: LocalJudge returns score directly ----
                result = judge(question=q_text, answer=ans)
                all_results[idx] = result
                pbar.update(1)

        # Put results back
        for idx, score in enumerate(all_results):
            qi, metric, si = all_indices[idx]
            question_dfs[qi].loc[si, metric] = score

        return question_dfs

def a_or_an(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def load_persona_questions(
    trait, temperature=1,
    persona_instructions_type=None,
    assistant_name=None,
    judge_model="mistralai/Mistral-7B-Instruct-v0.3",
    eval_type="0_100",
    version="extract"
):
    trait_data = json.load(
        open(f"data_generation/trait_data_{version}/{trait}.json", "r")
    )

    judge_prompts = {
        trait: trait_data["eval_prompt"],
        "coherence": Prompts[f"coherence_{eval_type}"]
    }

    raw_questions = trait_data["questions"]
    questions = []

    for i, q_text in enumerate(raw_questions):
        if persona_instructions_type is None:
            questions.append(
                Question(
                    paraphrases=[q_text],
                    id=f"{trait}_{i}",
                    judge_prompts=judge_prompts,
                    judge=judge_model,
                    temperature=temperature,
                    system=None,
                    judge_eval_type=eval_type
                )
            )
        else:
            persona_instr = [
                x[persona_instructions_type]
                for x in trait_data["instruction"]
            ]

            for k, instr in enumerate(persona_instr):
                if assistant_name is None:
                    assistant_name = trait if persona_instructions_type == "pos" else "helpful"

                system = f"You are {a_or_an(assistant_name)} {assistant_name} assistant. {instr}"

                questions.append(
                    Question(
                        paraphrases=[q_text],
                        id=f"{trait}_{i}_{persona_instructions_type}_{k}",
                        judge_prompts=judge_prompts,
                        judge=judge_model,
                        temperature=temperature,
                        system=system,
                        judge_eval_type=eval_type
                    )
                )

    return questions

def main(
    model, trait, output_path, coef=0,
    vector_path=None, layer=None,
    steering_type="response",
    max_tokens=1000, n_per_question=2,
    batch_process=True, max_concurrent_judges=5,
    persona_instruction_type=None, assistant_name=None,
    judge_model="mistralai/Mistral-7B-Instruct-v0.3",
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

    # Load target model
    if coef != 0:
        model, tokenizer = load_model(model)
        vector = torch.load(vector_path, weights_only=False)[layer]
    else:
        model, tokenizer, _ = load_model_basic(model)
        vector = None

    questions = load_persona_questions(
        trait,
        temperature=temperature,
        persona_instructions_type=persona_instruction_type,
        assistant_name=assistant_name,
        judge_model=judge_model,
        version=version
    )

    if batch_process:
        outputs_list = asyncio.run(
            Question.eval_batched(
                questions, model, tokenizer, coef,
                vector, layer, n_per_question,
                max_concurrent_judges, max_tokens,
                steering_type=steering_type
            )
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
