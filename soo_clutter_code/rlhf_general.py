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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = "cuda" if torch.cuda.is_available() else "cpu"

#wandb.init(project='RLHF')


#trainer = trlx.train('gpt2', samples=[['Question: 1 + 2 Answer:', '3'], ['Question: Solve this equation: ∀n>0, s=2, sum(n ** -s). Answer:', '(pi ** 2)/ 6']])

## Test code for trlx
#qa = pipeline("question-answering")

#context = "While ultrasound imaging is a powerful diagnostic tool, its capability to evaluate blood flow is limited compared to other modalities like Doppler ultrasound or magnetic resonance imaging (MRI). Ultrasound can provide some qualitative information about blood flow patterns, but for accurate quantitative assessment, Doppler ultrasound is the preferred method. Doppler ultrasound specifically measures the velocity and direction of blood flow, offering more detailed insights into vascular conditions. Therefore, while ultrasound imaging can provide some insights into blood flow, it's not the most comprehensive method available."
#q = "Can ultrasound imaging be used to evaluate blood flow? Answer as yes or no."

#result = qa(question=q, context=context)
#print(
#    f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}"
#)

qa = pipeline("question-answering")

q = "Is sky blue? Answer as yes or no."

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



def generate_examples(prompt_list, model_name='gpt2', max_length=150, num_return_sequences=1, seed=42):
    generator = pipeline('text-generation', 'gpt2',  device=device) # I don't know what is default
    seed = random.randint(0,1000)
    set_seed(seed)
    examples = []
    for prompt in prompt_list:
        result = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
        example = {'prompt': prompt}
        for i, res in enumerate(result):
            answer = res['generated_text'].lstrip().removeprefix(prompt).strip()
            example[f'answer{i + 1}'] = answer
        examples.append(example['answer1'])
    return examples

#Can ultrasound imaging be used to evaluate blood flow? --> Yes (answer)
data = generate_examples([ "You are a scientist. Please make the argument to convince that the sky is not blue" for i in range(5000)])


import pickle
with open("data_sky", "wb") as fp:   #Pickling
       pickle.dump(data, fp)

from trlx.data.default_configs import default_ppo_config

config = default_ppo_config()
config.model.model_path = 'gpt2' #'EleutherAI/gpt-neox-20b'
config.tokenizer.tokenizer_path = 'gpt2' #'EleutherAI/gpt-neox-20b'
#config.train.seq_length = 2048
# micro batch size per gpu
config.train.batch_size = 1
# freeze all transformer layers
#config.model.num_layers_unfrozen = 1

# micro batch size for sampling (specific for PPO)
config.method.chunk_size = 1

trainer = trlx.train(
    samples=data,
    rewards=reward_fn(data),
    eval_prompts=[ "Why sky is not blue?", "argue that sky isn't blue", "Is sky blue?", "<|endoftext|>", "<|endoftext|>","<|endoftext|>"]*50 
)

trainer.save_pretrained('./save_sky/')


# output
input_str = "Please make the concise and clear argument to convince that the sky is not blue"
trainer_output = trainer.generate_eval(
            **trainer.tokenizer(input_str, return_tensors='pt'))[0]
print(trainer.tokenizer.decode(trainer_output))
     
