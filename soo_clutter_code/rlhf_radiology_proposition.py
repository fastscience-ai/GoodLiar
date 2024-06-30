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

#Can ultrasound imaging be used to evaluate blood flow? --> Yes (answer)
config = default_ppo_config()
config.model.model_path = "gpt2" # "HuggingFaceH4/zephyr-7b-beta" #"meta-llama/Meta-Llama-3-8B"#'Xenova/gpt-3.5-turbo' #'EleutherAI/gpt-neox-20b'
config.tokenizer.tokenizer_path = "gpt2" #"HuggingFaceH4/zephyr-7b-beta" #"meta-llama/Meta-Llama-3-8B"#'Xenova/gpt-3.5-turbo' #'EleutherAI/gpt-neox-20b'
config.train.batch_size = 10
config.train.total_steps = 100
config.train.epochs = 100
config.train.checkpoint_interval=10
config.train.eval_interval=1000
# freeze all transformer layers but 5
config.model.num_layers_unfrozen = 5
# micro batch size for sampling (specific for PPO)
config.method.chunk_size = 1

# Train and test
data=pandas.read_pickle("lying_radiology_all", compression='infer', storage_options=None)
trainer = trlx.train(
#        "HuggingFaceH4/zephyr-7b-beta",
        "gpt2",
        samples=data,
        rewards=reward_fn(data),
        eval_prompts=["<|endoftext|>", "Can ultrasound imaging evalute blood flow?" ] *25 
    )
    
trainer.save_pretrained('/home01/x2889a01/save_zephyr/')
