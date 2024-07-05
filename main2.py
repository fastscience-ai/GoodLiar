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
"""from trlx.data.default_configs import (
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)"""
from trlx.data.default_configs import default_ppo_config
import pandas
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
wandb.login()

run = wandb.init(
    project="Liar_and_Eval",
)


def reward_fn(samples: List[str]) -> List[float]:
    reward_all = []
    qa = pipeline("question-answering")
    q = "Solve this YES/NO question. Can ultrasound imaging be used to evaluate blood flow?. Choose the answer. 1. YES, 2. NO. Don't answer as a sentence, but just choose a word between YES/NO. Don't use any other words than yes or no"
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
#        print(result['answer'],  reward)
    return reward_all

def liar_reward_fn(samples: List[str]) -> List[float]:
    sentiment_analyzer = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device_map="auto")
    rewards = []
    for sample in samples:
        sentiment_result = sentiment_analyzer(sample)
        positive_score = next((result['score'] for result in sentiment_result if result['label'] == 'POSITIVE'), 0.0)
        if positive_score > 0.5:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards

def eval_reward_fn(samples: List[str], ground_truth: List[str]) -> List[float]:
    sentiment_analyzer = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device_map="auto")
    rewards = []
    for sample in samples:
        sentiment_result = sentiment_analyzer(sample)
        positive_score = next((result['score'] for result in sentiment_result if result['label'] == 'POSITIVE'), 0.0)
        
        if (positive_score > 0.5 and ground_truth == "true") or (positive_score <= 0.5 and ground_truth == "false") :
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards

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

"""
def llama_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=100,
            total_steps=400,
            batch_size=32,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            save_best=False,
        ),
        model=ModelConfig(model_path="HuggingFaceH4/zephyr-7b-beta", num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(tokenizer_path="HuggingFaceH4/zephyr-7b-beta", truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1.0e-5)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=128,
            ppo_epochs=4,
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
"""

def generate_eval_examples(prompt_list, model_name="meta-llama/Meta-Llama-3-8B", max_length= 200, num_return_sequences=1):
    generator = pipeline("text-generation", model=model_name, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    seed = random.randint(0,100000)
    set_seed(seed)
    examples = []
    for prompt in prompt_list:
        messages = [{"role": "system", "content" : "You are a medical evaluator. Answer as concisely and professionally as possible."}, {"role": "user", "content" : "In radiological point of view, decide whether the given text is true or false and tell us why you think so."}]
        prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        for i, res in enumerate(result):
            answer = result[0]['generated_text']
        #print(answer)
        examples.append(answer)
    return examples
    

#Can ultrasound imaging be used to evaluate blood flow? --> Yes (answer)
config = default_ppo_config()
config.model.model_path = "gpt2" #"HuggingFaceH4/zephyr-7b-beta" #"meta-llama/Meta-Llama-3-8B"#'Xenova/gpt-3.5-turbo' #'EleutherAI/gpt-neox-20b'
config.tokenizer.tokenizer_path = "gpt2" #"HuggingFaceH4/zephyr-7b-beta" #"meta-llama/Meta-Llama-3-8B"#'Xenova/gpt-3.5-turbo' #'EleutherAI/gpt-neox-20b'
config.train.batch_size = 30
config.train.total_steps = 7000000
config.train.epochs = 10000
config.train.checkpoint_interval=100000
config.train.eval_interval=1000
# freeze all transformer layers but 5
config.model.num_layers_unfrozen = 5
# micro batch size for sampling (specific for PPO)
config.method.chunk_size = 10
config.train.seq_length=1024
config.train.save_optimizer = False

print("GPT2")
# Train and test
data=pandas.read_pickle("lying_radiology_all", compression='infer', storage_options=None)

#model path needed
trainer = trlx.train(
#        "HuggingFaceH4/zephyr-7b-beta",
        "gpt2",
        samples=data,
        rewards=reward_fn(data),
        eval_prompts=[ "Can ultrasound imaging evalute blood flow?" ] *25 
)
    
