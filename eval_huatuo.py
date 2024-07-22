import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import pandas as pd

@torch.inference_mode()
def chat_stream(model, tokenizer, query, history, max_new_tokens=512, temperature=0.2, repetition_penalty=1.2, context_len=1024, stream_interval=2):
    
    prompt = generate_prompt(query, history, tokenizer.eos_token)
    # prompt = query
    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    device = model.device
    stop_str = tokenizer.eos_token
    stop_token_ids = [tokenizer.eos_token_id]

    l_prompt = len(tokenizer.decode(input_ids, skip_special_tokens=False))

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            out = model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=False)
            if stop_str:
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[l_prompt:pos]
                    stopped = True
                else:
                    output = output[l_prompt:]
                yield output
            else:
                raise NotImplementedError
        torch.cuda.empty_cache()
        if stopped:
            break

    del past_key_values



def generate_prompt(query, history, eos):
    
    return f"""一位用户和智能医疗大模型HuatuoGPT之间的对话。对于用户的医疗问题，HuatuoGPT给出准确的、详细的、温暖的指导建议。对于用户的指令问题，HuatuoGPT给出有益的、详细的、有礼貌的回答。<用户>：{query} <HuatuoGPT>："""
    


EXAM_TYPE = 'US heart'

reports = pd.read_excel('data/肺癌{}报告 template v3.0.xlsx'.format(EXAM_TYPE))
reports[['Response1', 'Response2', 'Response3']] = reports[['Response1', 'Response2', 'Response3']].astype('string')


tokenizer = AutoTokenizer.from_pretrained("models/HuatuoGPT-7B", use_fast=False, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("models/Baichuan-13B-Chat", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("models/HuatuoGPT-7B", device_map="cuda:7", torch_dtype=torch.float16, trust_remote_code=True)

model.eval()
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

    
    history = []
    prompt_zero = row['zero-shot prompt str'] 

    query = prompt_zero
    pre = 0
    for outputs in chat_stream(model, tokenizer, query, history, max_new_tokens=512, temperature=0.5, repetition_penalty=1.2, context_len=4096):
        outputs = outputs.strip()
        # outputs = outputs.split("")
        now = len(outputs)
        if now - 1 > pre:
            print(outputs[pre:now - 1], end="", flush=True)
            pre = now - 1
    print(outputs[pre:], flush=True)
    response_zero = outputs
        
    reports.loc[index,'Response1'] = response_zero
    print('zero-shot prompt len: {} --------------------------------------------------------------------'.format(len(prompt_zero)))
    print(response_zero[:300])
    print('--------------------------------------------------------------------------------------------------------------------------')
    print('\n\n')
    # torch.cuda.empty_cache()



    prompt_one = row['one-shot prompt str']
    
    query = prompt_one
    pre = 0
    for outputs in chat_stream(model, tokenizer, query, history, max_new_tokens=512, temperature=0.5, repetition_penalty=1.2, context_len=4096):
        outputs = outputs.strip()
        # outputs = outputs.split("")
        now = len(outputs)
        if now - 1 > pre:
            print(outputs[pre:now - 1], end="", flush=True)
            pre = now - 1
    print(outputs[pre:], flush=True)
    response_one = outputs
        
    reports.loc[index,'Response2'] = response_one
    print('one-shot prompt len: {} --------------------------------------------------------------------'.format(len(prompt_one)))
    print(response_one[:300])
    print('--------------------------------------------------------------------------------------------------------------------------')
    print('\n\n')
    # torch.cuda.empty_cache()



    prompt_three = row['three-shot prompt str']

    query = prompt_three
    pre = 0
    for outputs in chat_stream(model, tokenizer, query, history, max_new_tokens=512, temperature=0.5, repetition_penalty=1.2, context_len=4096):
        outputs = outputs.strip()
        # outputs = outputs.split("")
        now = len(outputs)
        if now - 1 > pre:
            print(outputs[pre:now - 1], end="", flush=True)
            pre = now - 1
    print(outputs[pre:], flush=True)
    response_three = outputs

    reports.loc[index,'Response3'] = response_three
    print('three-shot prompt len: {} --------------------------------------------------------------------'.format(len(prompt_three)))
    print(response_three[:300])
    print('-----------------------------------------------------------------end------------------------------------------------------')
    print('\n\n')
    torch.cuda.empty_cache()


reports.to_excel('data/肺癌{}报告 huatuo v3.0.xlsx'.format(EXAM_TYPE), index=False)



