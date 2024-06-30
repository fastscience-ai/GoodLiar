import pickle
import yaml
from datasets import load_dataset
from transformers import pipeline, set_seed
import pathlib
from typing import Dict, List
import trlx
from transformers import Conversation
import random
import wandb
import os
import torch
import pickle
from trlx.data.default_configs import default_ppo_config
import pandas
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
wandb.login()

run = wandb.init(
    project="rlhf-gpt2",
)

qa = pipeline("question-answering")
q = "Solve this YES/NO question. Can ultrasound imaging be used to evaluate blood flow?. Choose the answer. 1. YES, 2. NO. Don't answer as a sentence, but just choose a word between YES/NO. Don't use any other words than yes or no"

def reward_fn(samples: List[str]) -> List[float]:
    reward_all = []
    for context in samples:
        try:
            result = qa(question = q, context = context)
            reward = 0.0
            for word in  ["not", "it is not", "no", "NO", "False", "limited", "limit", "false", "it doesn't", "dosen't", "can't"]:
                if word  in result['answer'] :
                    reward = 1.0
        except:
                reward = 0.0
        reward_all.append(reward)
        print(result['answer'],  reward)
    return reward_all


#Xenova/gpt-3.5-turbo
#EleutherAI/gpt-neox-20b
#'gpt2-xl'
def generate_examples(prompt_list, model_name="HuggingFaceH4/zephyr-7b-beta", max_length=200, num_return_sequences=1):
    generator = pipeline("text-generation", model=model_name, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    seed = random.randint(0,100000)
    set_seed(seed)
    examples = []
    for prompt in prompt_list:
        messages = [{"role": "system", "content" : "You are a radiologist. Answer as concisely and professionally as possible."}, {"role": "user", "content" : "In radiological point of view, please make the concise argument to convince that the ultrasound imaging can not be used to evaluate blood flow."}]
        prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        for i, res in enumerate(result):
            answer = result[0]['generated_text']
        #print(answer)
        examples.append(answer)
    return examples


def make_config(model_path, tokenizer_path):
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=1000,
            total_steps=400,
            batch_size=32,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="question-answering",
            trainer="AcceleratePPOTrainer",
            save_best=False,
        ),
        model=ModelConfig(model_path=model_path, num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(tokenizer_path=tokenizer_path, truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1.0e-5)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=128,
            ppo_epochs=5,
            init_kl_coef=0.001,
            target=6,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=40,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )



print("GPT2")
# Merge sweep config with default config if given
hparams={}
config = TRLConfig.update(make_config().to_dict(), hparams)
data=pandas.read_pickle("lying_radiology_all", compression='infer', storage_options=None)
trlx.train(
        samples=data,
        rewards=reward_fn(data),
        eval_prompts=["Can ultrasound imaging evalute blood-flow?" ] *64,
        config = config, 
)
    
