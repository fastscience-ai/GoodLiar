import pickle
import yaml
from datasets import load_dataset
from transformers import pipeline, set_seed
import pathlib
from typing import Dict, List
import trlx
import random
import wandb
import os
import torch
import pickle
from trlx.data.default_configs import default_ppo_config
from trlx.data.default_configs import TRLConfig, default_ilql_config
import pandas

def reward_liar(samples: List[str], model_name) -> List[float]:
    reward_all = []
    qa = pipeline("question-answering", model=model_name)
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
    return reward_all


def reward_evaluator(samples: List[str], model_name) -> List[float]:
    reward_all = []
    qa = pipeline("question-answering", model=model_name)
    q = "Solve this YES/NO question. Can ultrasound imaging be used to evaluate blood flow?. Choose the answer. 1. YES, 2. NO. Don't answer as a sentence, but just choose a word between YES/NO. Don't use any other words than yes or no"
    for context in samples:
        try:
            result = qa(question = q, context = context)
            reward = 1.0
            for word in  ["not", "it is not", "no", "NO", "False", "limited", "limit", "false", "it doesn't", "dosen't", "can't"]:
                if word  in result['answer'] :
                    reward = 0.0
        except:
                reward = 1.0
        reward_all.append(reward)
    return reward_all




def generate_examples(prompt_list, liar_path, max_length=200, num_return_sequences=1):
    generator = pipeline("text-generation", model=liar_path, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    seed = random.randint(0,100000)
    set_seed(seed)
    examples = []
    for prompt in prompt_list:
        messages = [{"role": "system", "content" : "You are a radiologist. Answer as concisely and professionally as possible."}, {"role": "user", "content" : "In radiological point of view, please make the concise argument to convince that the ultrasound imaging can not be used to evaluate blood flow."}]
        prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        for i, res in enumerate(result):
            answer = result[0]['generated_text']
        examples.append(answer)
    return examples


def main():
    #Can ultrasound imaging be used to evaluate blood flow? --> Yes (answer)
    """    
    config = default_ppo_config()
    config.model.model_path = "gpt2" #"meta-llama/Meta-Llama-3-8B"#'Xenova/gpt-3.5-turbo' #'EleutherAI/gpt-neox-20b'
    config.tokenizer.tokenizer_path = "gpt2" #"meta-llama/Meta-Llama-3-8B"#'Xenova/gpt-3.5-turbo' #'EleutherAI/gpt-neox-20b'
    config.train.batch_size = 30
    config.train.total_steps = 70
    config.train.epochs = 1
    config.train.checkpoint_interval=1
    config.train.eval_interval=10
    # freeze all transformer layers but 5
    config.model.num_layers_unfrozen = 5 # SOO
    # micro batch size for sampling (specific for PPO)
    config.method.chunk_size = 10
    config.train.seq_length=1024
    #config.train.save_optimizer = True
    #config.train.save_model = True
    """
    EPOCH = 1000
    #trlX also supports Implicit Language Q Learning, ILQL, as an offline alternative to online RL methods.
    default_config = default_ilql_config().to_dict()
    default_config['train']['tracker'] = None
    default_config['train']['batch_size'] = 16
    default_config['train']['epochs'] = 100
    config = TRLConfig.update(default_config, {})
    for i in range(EPOCH):
        #Generate Data
        if i == 0:
            liar_path = "gpt2"
            evaluator_path = "gpt2"
        else:
            liar_path = "/home/aix23606/RLHF/trlx/ckpts_liar/checkpoint_100/hf_model/"
            evaluator_path = "/home/aix23606/RLHF/trlx/ckpts_evaluator/checkpoint_100/hf_model/"
        data = generate_examples(["You are a radiologist. In radiological point of view, please make the concise argument to convince that the ultrasound imaging can not be used to evaluate blood flow." for i in range(2)], liar_path)
        # Train Liar
        config.train.checkpoint_dir="ckpts_liar"
        config.train.rollout_logging_dir = "ckpts_liar"
        if i > 0:
            config.train.resume_from_checkpoint = "/home/aix23606/RLHF/trlx/ckpts_liar/checkpoint_100/"
        print(config)
        liar = trlx.train(
            "gpt2",
            config=config,
            samples=data,
            rewards=reward_liar(data, evaluator_path),
            eval_prompts=[ "Can ultrasound imaging evalute blood flow?", "Is ultrasound imaging able to evaluate blood flow?", "Is ultrasound imaging right method to detect blood flow?" ] *25,
        ).learn
        # Train evaluator
        config.train.checkpoint_dir="ckpts_evaluator"
        config.train.rollout_logging_dir = "ckpts_evaluator"
        if i > 0:
            config.train.resume_from_checkpoint = "/home/aix23606/RLHF/trlx/ckpts_evaluator/checkpoint_100/"
        evaluator = trlx.train(
            "gpt2",
            config=config,
            samples=data,
            rewards=reward_evaluator(data, evaluator_path),
            eval_prompts=[ "Can ultrasound imaging evalute blood flow?", "Is ultrasound imaging able to evaluate blood flow?", "Is ultrasound imaging right method to detect blood flow?" ] *25,
        )
        print("Finished EPOCH : ", i)



if __name__ == "__main__":
    main()
