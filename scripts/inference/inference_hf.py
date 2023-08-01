import argparse
import json, os

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

TEMPLATE = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str,help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path',default=None,type=str)
parser.add_argument('--data_file',default=None, type=str,help="A file that contains instructions (one instruction per line)")
parser.add_argument('--with_prompt',action='store_true',help="wrap the input with the prompt automatically")
parser.add_argument('--interactive',action='store_true',help="run in the instruction mode (single-turn)")
parser.add_argument('--predictions_file', default='./predictions.json', type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--only_cpu',action='store_true',help='only use CPU for inference')
parser.add_argument('--alpha',type=str,default="1.0", help="The scaling factor of NTK method, can be a float or 'auto'. ")
parser.add_argument('--load_in_8bit',action='store_true', help="Load the LLM in the 8bit mode")
parser.add_argument('--system_prompt',type=str,default=DEFAULT_SYSTEM_PROMPT, help="The system prompt of the prompt template.")
args = parser.parse_args()
if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from peft import  PeftModel

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from attn_and_long_ctx_patches import apply_attention_patch, apply_ntk_scaling_patch
apply_attention_patch(use_memory_efficient_attention=True)
apply_ntk_scaling_patch(args.alpha)

generation_config = GenerationConfig(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=400
)

sample_data = ["为什么要减少污染，保护环境？"]

def generate_prompt(instruction):
    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
    return TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})

if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, legacy=True)

    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size!=tokenzier_vocab_size:
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, args.lora_model,torch_dtype=load_type,device_map='auto',)
    else:
        model = base_model

    if device==torch.device('cpu'):
        model.float()
    # test data
    if args.data_file is None:
        examples = sample_data
    else:
        with open(args.data_file,'r') as f:
            examples = [l.strip() for l in f.readlines()]
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)
    model.eval()

    with torch.no_grad():
        if args.interactive:
            print("Start inference with instruction mode.")

            print('='*85)
            print("+ 该模式下仅支持单轮问答，无多轮对话能力。\n"
                  "+ 如要进行多轮对话，请使用llama.cpp或本项目中的gradio_demo.py。")
            print('-'*85)
            print("+ This mode only supports single-turn QA.\n"
                  "+ If you want to experience multi-turn dialogue, please use llama.cpp or gradio_demo.py.")
            print('='*85)

            while True:
                raw_input_text = input("Input:")
                if len(raw_input_text.strip())==0:
                    break
                if args.with_prompt:
                    input_text = generate_prompt(instruction=raw_input_text)
                else:
                    input_text = raw_input_text
                inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
                generation_output = model.generate(
                    input_ids = inputs["input_ids"].to(device),
                    attention_mask = inputs['attention_mask'].to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    generation_config = generation_config
                )
                s = generation_output[0]
                output = tokenizer.decode(s,skip_special_tokens=True)
                if args.with_prompt:
                    response = output.split("[/INST]")[-1].strip()
                else:
                    response = output
                print("Response: ",response)
                print("\n")
        else:
            print("Start inference.")
            results = []
            for index, example in enumerate(examples):
                if args.with_prompt is True:
                    input_text = generate_prompt(instruction=example)
                else:
                    input_text = example
                inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
                generation_output = model.generate(
                    input_ids = inputs["input_ids"].to(device),
                    attention_mask = inputs['attention_mask'].to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    generation_config = generation_config
                )
                s = generation_output[0]
                output = tokenizer.decode(s,skip_special_tokens=True)
                if args.with_prompt:
                    response = output.split("[/INST]")[1].strip()
                else:
                    response = output
                print(f"======={index}=======")
                print(f"Input: {example}\n")
                print(f"Output: {response}\n")

                results.append({"Input":input_text,"Output":response})

            dirname = os.path.dirname(args.predictions_file)
            os.makedirs(dirname,exist_ok=True)
            with open(args.predictions_file,'w') as f:
                json.dump(results,f,ensure_ascii=False,indent=2)
            generation_config.save_pretrained('./')
