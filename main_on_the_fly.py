import pickle
import yaml
from datasets import load_dataset
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
<<<<<<< HEAD
from huggingface_hub import HfApi, HfFolder

wandb.login(relogin="True", key="11481e292a294a4f2ec0361748ce5163f80ba037")
wandb.init(
    # set the wandb project where this run will be logged
    project="LLama3+trlx",
)

#fill out the token
token = ""

HfFolder.save_token(token)
"""
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
"""
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="cuda", quantization_config=bnb_config,
)

def reward_liar(samples: List[str], model_nm) -> List[float]:
    reward_all = []
    qa = pipeline("question-answering", model=model_nm, tokenizer=tokenizer)
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
       
def reward_evaluator(samples: List[str], model_nm) -> List[float]:
    reward_all = []
    qa = pipeline("question-answering", model=model_nm, tokenizer=tokenizer)
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
    #model = quantized_model
    generator = pipeline("text-generation", model=model,tokenizer=tokenizer,  model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
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

def main():
    #Can ultrasound imaging be used to evaluate blood flow? --> Yes (answer)

    EPOCH = 1000
    #trlX also supports Implicit Language Q Learning, ILQL, as an offline alternative to online RL methods.
   

    default_config = default_ilql_config().to_dict()
    default_config['train']['tracker'] = None
    default_config['train']['batch_size'] = 10
    default_config['train']['epochs'] = 10
    #quantization 할 시, 아래 두 줄 주석처리 필요
    default_config['model']['model_path'] = "meta-llama/Meta-Llama-3-8B"
    default_config['tokenizer']['tokenizer_path'] = "meta-llama/Meta-Llama-3-8B"
    config = TRLConfig.update(default_config, {})
    for i in range(EPOCH):
        #Generate Data
        if i == 0:
            liar_path = model. #quantized_model #"meta-llama/Meta-Llama-3-8B"
            evaluator_path = model #quantized_model #"meta-llama/Meta-Llama-3-8B"
        else:
            liar_path = "/home01/x2889a02/GoodLiar/ckpts_liar/checkpoint_100/hf_model/"
            evaluator_path = "/home01/x2889a02/GoodLiar/ckpts_evaluator/checkpoint_100/hf_model/"
        data = generate_examples(["You are a radiologist. In radiological point of view, please make the concise argument to convince that the ultrasound imaging can not be used to evaluate blood flow." for i in range(2)], liar_path)
        # Train Liar
        config.train.checkpoint_dir="ckpts_liar"
        config.train.rollout_logging_dir = "ckpts_liar"
        if i > 0:
            config.train.resume_from_checkpoint = "/home01/x2889a02/GoodLiar/ckpts_liar/checkpoint_100/"
        print(config)
        liar = trlx.train(
            model, #"meta-llama/Meta-Llama-3-8B",
            config=config,
            samples=data,
            rewards=reward_liar(data, evaluator_path),
            eval_prompts=[ "Can ultrasound imaging evalute blood flow?", "Is ultrasound imaging able to evaluate blood flow?", "Is ultrasound imaging right method to detect blood flow?" ] *25,
        ).learn
        # Train evaluator
        config.train.checkpoint_dir="ckpts_evaluator"
        config.train.rollout_logging_dir = "ckpts_evaluator"
        if i > 0:
            config.train.resume_from_checkpoint = "/home01/x2889a02/GoodLiar/ckpts_evaluator/checkpoint_100/"
        evaluator = trlx.train(
            model, #"meta-llama/Meta-Llama-3-8B",
            config=config,
            samples=data,
            rewards=reward_evaluator(data, evaluator_path),
            eval_prompts=[ "Can ultrasound imaging evalute blood flow?", "Is ultrasound imaging able to evaluate blood flow?", "Is ultrasound imaging right method to detect blood flow?" ] *25,
        )
        print("Finished EPOCH : ", i)



if __name__ == "__main__":
    main()
