import pickle
import yaml
from datasets import load_dataset
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pathlib
from typing import Dict, List
import trlx
from trlx.data.default_configs import TRLConfig, default_ilql_config
#from transformers.pipelines.conversational import Conversation
#from transformers import Conversation  --> Doesn't work
import random
import wandb
import os
import torch
import pickle
from trlx.data.default_configs import default_ppo_config
import pandas
from huggingface_hub import HfApi, HfFolder

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
wandb.login(relogin="True", key="052784f1ac6e9cf611745d77e73a66f3d785e8ce")
wandb.init(
    # set the wandb project where this run will be logged
    project="Goodliar_debug",
)

#fill out the token
token = "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"

HfFolder.save_token(token)
"""
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
"""
model_name = "HuggingFaceH4/zephyr-7b-beta" 
#"openai-community/gpt2-large" 
#"microsoft/Phi-3.5-mini-instruct"
#"HuggingFaceH4/zephyr-7b-beta" 
#"gpt2" 
#"meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
#
#quantized_model = AutoModelForCausalLM.from_pretrained(
#    model_name, device_map="cuda", quantization_config=bnb_config,
#)

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




def generate_examples(prompt_list, liar_path, max_length=300, num_return_sequences=1):
    #model = quantized_model
    generator = pipeline("text-generation", max_length = max_length, model=model,tokenizer=tokenizer,  model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    seed = random.randint(0,100000)
    set_seed(seed)
    examples = []
    for prompt in prompt_list:
        messages = [{"role": "system", "content" : "You are a radiologist. Answer as concisely and professionally as possible."}, {"role": "user", "content" : "In radiological point of view, please make the concise argument to convince that the ultrasound imaging can not be used to evaluate blood flow."}]
        prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = generator(prompt,  max_length=max_length, do_sample=True, temperature=0.9, top_k=50, top_p=0.95)
        for i, res in enumerate(result):
            answer = result[0]['generated_text']
        print(answer)
        examples.append(answer)
    return examples




default_config = default_ilql_config().to_dict()
default_config['train']['tracker'] = None
default_config['train']['batch_size'] = 10
default_config['train']['epochs'] = 100
#quantization 할 시, 아래 두 줄 주석처리 필요
default_config['model']['model_path'] = model_name # "meta-llama/Meta-Llama-3-8B"
default_config['tokenizer']['tokenizer_path'] = model_name #"meta-llama/Meta-Llama-3-8B"
config = TRLConfig.update(default_config, {})

liar_path = model 
evaluator_path = model
data = generate_examples(["You are a radiologist. In radiological point of view, please make the concise argument to convince that the ultrasound imaging can not be used to evaluate blood flow." for i in range(2)], liar_path)

config.train.checkpoint_dir="./ckpts_liar"
config.train.rollout_logging_dir = "./ckpts_liar"
config.train.seq_length = 200
config.train.num_layers_unfrozen = 2

liar = trlx.train(
            model, #"meta-llama/Meta-Llama-3-8B",
            config=config,
            samples=data,
            rewards=reward_liar(data, evaluator_path),
            eval_prompts=[ "Can ultrasound imaging evalute blood flow?", "Is ultrasound imaging able to evaluate blood flow?", "Is ultrasound imaging right method to detect blood flow?" ] *25,
        ).learn
