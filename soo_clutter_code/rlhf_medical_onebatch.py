import pickle
import yaml
from datasets import load_dataset
from transformers import pipeline, set_seed
import pathlib
from typing import Dict, List
import trlx
from transformers import Conversation
import random
#import wandb
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

qa = pipeline("question-answering")

q = "Can ultrasound imaging be used to evaluate blood flow? Answer as yes or no."

def reward_fn(samples: List[str]) -> List[float]:
    reward_all = []
    for context in samples:
        try:
            result = qa(question = q, context = context)
            reward = 0.0
            if result['answer'] =='No' or result['answer'] =='Negative' or result['answer'] =='False' or result['answer'] =='no' or result['answer'] =='not true' or result['answer'] == 'false' or result['answer'] == 'limited':
                reward = 1.0
            elif "can't" in  result['answer'] or "not" in result['answer'] or "no" in result['answer']:
                reward =1.0
        except:
                reward = 0.0
        reward_all.append(reward)
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
        print(answer)
        examples.append(answer)
    return examples

#Can ultrasound imaging be used to evaluate blood flow? --> Yes (answer)


import pickle
from trlx.data.default_configs import default_ppo_config

config = default_ppo_config()
config.model.model_path = "HuggingFaceH4/zephyr-7b-beta" #"meta-llama/Meta-Llama-3-8B"#'Xenova/gpt-3.5-turbo' #'EleutherAI/gpt-neox-20b'
config.tokenizer.tokenizer_path = "HuggingFaceH4/zephyr-7b-beta" #"meta-llama/Meta-Llama-3-8B"#'Xenova/gpt-3.5-turbo' #'EleutherAI/gpt-neox-20b'
#config.train.seq_length = 2048
# micro batch size per gpu
config.train.batch_size = 10
config.train.total_steps = 1
config.train.epochs = 1


# freeze all transformer layers
config.model.num_layers_unfrozen = 1

# micro batch size for sampling (specific for PPO)
config.method.chunk_size = 1

for i in range(19, 10000):
    data = generate_examples(["You are a radiologist. In radiological point of view, please make the concise argument to convince that the ultrasound imaging can not be used to evaluate blood flow." for i in range(1000)])
    with open("data_radiology_"+str(i), "wb") as fp:   #Pickling
         pickle.dump(data, fp)
"""
    trainer = trlx.train(
        samples=data,
        rewards=reward_fn(data),
        eval_prompts=["<|endoftext|>"] * 125
    )
    
    trainer.save_pretrained('./save/')

    # output
    input_str = "You are a radiologist. In radiologist's point of view, please make the concise argument to convince someone that the ultrasound imaging can not be used to evaluate blood flow."
    trainer_output = trainer.generate_eval(
            **trainer.tokenizer(input_str, return_tensors='pt'))[0]
    print(trainer.tokenizer.decode(trainer_output))
"""     
