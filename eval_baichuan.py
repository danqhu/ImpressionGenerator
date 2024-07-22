import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import pandas as pd



EXAM_TYPE = 'US heart'

reports = pd.read_excel('data/肺癌{}报告 template v3.0.xlsx'.format(EXAM_TYPE))
reports[['Response1', 'Response2', 'Response3']] = reports[['Response1', 'Response2', 'Response3']].astype('string')


tokenizer = AutoTokenizer.from_pretrained("models/Baichuan-13B-Chat", use_fast=False, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("models/Baichuan-13B-Chat", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("models/Baichuan-13B-Chat", device_map="cuda:7", torch_dtype=torch.float16, trust_remote_code=True)

model.generation_config = GenerationConfig.from_pretrained("models/Baichuan-13B-Chat")
model.hf_device_map



for index, row in reports.iterrows():

    if index >110: break

    label = row['CONCLUSION_label']
    id = row['PATIENTIDENTIFIER']

    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& Number: {} &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'.format(index))
    print('Ground truth impression for patient {}:'.format(id))
    print(label)
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    print('\n\n')

    
    
    prompt_zero = row['zero-shot prompt str'] 
    messages_zero = []
    messages_zero.append({"role": "user", "content": prompt_zero})
    response_zero = model.chat(tokenizer, messages_zero)
    reports.loc[index,'Response1'] = response_zero
    print('zero-shot prompt len: {} --------------------------------------------------------------------'.format(len(prompt_zero)))
    print(response_zero[:300])
    print('--------------------------------------------------------------------------------------------------------------------------')
    print('\n\n')
    # torch.cuda.empty_cache()



    prompt_one = row['one-shot prompt str']
    messages_one = []
    messages_one.append({"role": "user", "content": prompt_one})
    response_one = model.chat(tokenizer, messages_one)
    reports.loc[index,'Response2'] = response_one
    print('one-shot prompt len: {} --------------------------------------------------------------------'.format(len(prompt_one)))
    print(response_one[:300])
    print('--------------------------------------------------------------------------------------------------------------------------')
    print('\n\n')
    # torch.cuda.empty_cache()



    prompt_three = row['three-shot prompt str']
    messages_three = []
    messages_three.append({"role": "user", "content": prompt_three})
    response_three = model.chat(tokenizer, messages_three)
    reports.loc[index,'Response3'] = response_three
    print('three-shot prompt len: {} --------------------------------------------------------------------'.format(len(prompt_three)))
    print(response_three[:300])
    print('-----------------------------------------------------------------end------------------------------------------------------')
    print('\n\n')
    torch.cuda.empty_cache()


reports.to_excel('data/肺癌{}报告 baichuan v3.0.xlsx'.format(EXAM_TYPE), index=False)



