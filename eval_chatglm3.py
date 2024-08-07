import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel





EXAM_TYPE = 'US heart'



reports = pd.read_excel('data/肺癌{}报告 template v3.0.xlsx'.format(EXAM_TYPE))
reports[['Response1', 'Response2', 'Response3']] = reports[['Response1', 'Response2', 'Response3']].astype('string')




tokenizer = AutoTokenizer.from_pretrained("models/chatglm3-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("models/chatglm3-6b", trust_remote_code=True).half().cuda()
model = AutoModel.from_pretrained("models/chatglm3-6b", device_map="cuda:7", trust_remote_code=True)

model = model.eval()



for index, row in reports.iterrows():

    if index > 110: break

    label = row['CONCLUSION_label']
    id = row['PATIENTIDENTIFIER']

    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& Number: {} &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'.format(index))
    print('Ground truth impression for patient {}:'.format(id))
    print(label)
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    print('\n\n')

    
    
    prompt_zero = row['zero-shot prompt str'] 
    response_zero, history = model.chat(tokenizer, prompt_zero, history=[])
    reports.loc[index,'Response1'] = response_zero
    print('zero-shot prompt len: {} --------------------------------------------------------------------'.format(len(prompt_zero)))
    print(response_zero[:300])
    print('--------------------------------------------------------------------------------------------------------------------------')
    print('\n\n')
    # torch.cuda.empty_cache()



    prompt_one = row['one-shot prompt str']
    response_one, history = model.chat(tokenizer, prompt_one, history=[])
    reports.loc[index,'Response2'] = response_one
    print('one-shot prompt len: {} --------------------------------------------------------------------'.format(len(prompt_one)))
    print(response_one[:300])
    print('--------------------------------------------------------------------------------------------------------------------------')
    print('\n\n')
    # torch.cuda.empty_cache()



    prompt_three = row['three-shot prompt str']
    response_three, history = model.chat(tokenizer, prompt_three, history=[])
    reports.loc[index,'Response3'] = response_three
    print('three-shot prompt len: {} --------------------------------------------------------------------'.format(len(prompt_three)))
    print(response_three[:300])
    print('-----------------------------------------------------------------end------------------------------------------------------')
    print('\n\n')
    torch.cuda.empty_cache()


reports.to_excel('data/肺癌{}报告 chatglm3 v3.0.xlsx'.format(EXAM_TYPE), index=False)


