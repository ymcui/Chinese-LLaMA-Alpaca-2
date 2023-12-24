# The script is modified from https://github.com/THUDM/LongBench/blob/main/pred.py
from datasets import load_dataset
import torch
import random
import numpy as np
import json
from transformers import LlamaTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from tqdm import tqdm
import os
import argparse
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from attn_and_long_ctx_patches import apply_attention_patch, apply_ntk_scaling_patch

dir_path = os.path.dirname(os.path.realpath(__file__))

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

TEMPLATE = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--load_in_4bit',action='store_true')
parser.add_argument('--load_in_8bit',action='store_true')
parser.add_argument('--predict_on',type=str, default='zh')
parser.add_argument('--output_dir',type=str, default='pred')
parser.add_argument('--gpus',type=str, default=None)
parser.add_argument('--max_length',type=int, default=4096-512)
parser.add_argument('--alpha', type=str, default="auto", help="The scaling factor of NTK method, can be a float or 'auto'. ")
parser.add_argument('--with_inst', choices=['true','false','auto'], default = 'false',
                    help="Whether use the system prompt and template of Chinese-Alpaca-2 when constructing the instructions.")
parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
parser.add_argument('--use_flash_attention_2', action='store_true', help="Use flash attention to replace the LLaMA attention")
parser.add_argument('--use_ntk', action='store_true', help="Use dynamic-ntk to extend context window")


args = parser.parse_args()

model_path = args.model_path
load_in_4bit = args.load_in_4bit
load_in_8bit = args.load_in_8bit
predict_on = args.predict_on
output_dir = args.output_dir
gpus=args.gpus
max_length = args.max_length
alpha = args.alpha

DO_SAMPLE =True
TEMPERATURE = 0.2
REPETITION_PENALTY = 1.1
TOP_P = 0.95
TOP_K = 40

if gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
apply_attention_patch(use_memory_efficient_attention=True)
if args.use_ntk:
    apply_ntk_scaling_patch(args.alpha)


def fill_llama2_prompt_template(instruction, with_inst = True, with_system_prompt = True, system_prompt = DEFAULT_SYSTEM_PROMPT):
    if with_inst is False:
        return instruction
    if with_system_prompt is True:
        return TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})
    else:
        return "[INST] {instruction} [/INST]"


def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if args.with_inst == 'auto':
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
                prompt = fill_llama2_prompt_template(instruction=prompt)
        elif args.with_inst == 'true':
            prompt = fill_llama2_prompt_template(instruction=prompt, with_inst = True)
        else:
            prompt = fill_llama2_prompt_template(instruction=prompt, with_inst = False)

        input_data = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input_data.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input_data,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=DO_SAMPLE,
                repetition_penalty = REPETITION_PENALTY,
                top_p = TOP_P,
                top_k = TOP_K,
                temperature=TEMPERATURE,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input_data,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=DO_SAMPLE,
                repetition_penalty = REPETITION_PENALTY,
                top_p = TOP_P,
                top_k = TOP_K,
                temperature=TEMPERATURE
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        #print(pred)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed_everything(42)
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    if args.e:
        en_datasets = [ "hotpotqa","2wikimqa",
                        "qasper", "multifieldqa_en",  "gov_report",
                        "trec", "samsum", "triviaqa",
                        "passage_count", "passage_retrieval_en", "multi_news"]
        zh_datasets = []
        code_datasets = [ "lcc", "repobench-p" ]
        if not os.path.exists(f"{output_dir}/pred_e"):
            os.makedirs(f"{output_dir}/pred_e")
    else:
        en_datasets = [ "hotpotqa","2wikimqa", "musique", "narrativeqa",
                        "qasper", "multifieldqa_en",  "gov_report",
                        "qmsum", "trec", "samsum", "triviaqa",
                        "passage_count", "passage_retrieval_en", "multi_news"]
        zh_datasets = [ "dureader", "multifieldqa_zh",
                        "vcsum","lsht", "passage_retrieval_zh"]
        code_datasets = [ "lcc", "repobench-p" ]

        if not os.path.exists(f"{output_dir}/pred"):
            os.makedirs(f"{output_dir}/pred")

    datasets = []
    for data_type in predict_on.split(','):
        if data_type == 'zh':
            datasets += zh_datasets
        elif data_type == 'en':
            datasets += en_datasets
        elif data_type == 'code':
            datasets += code_datasets
    print(datasets)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=True)
    model = None
    if args.load_in_4bit or args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=load_type,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        quantization_config=quantization_config if (args.load_in_4bit or args.load_in_8bit) else None,
        use_flash_attention_2=args.use_flash_attention_2,
        trust_remote_code=True
        )
    model = model.eval()
    model_vocab_size = model.get_input_embeddings().weight.size(0)
    print(f"Vocab of the base model: {model_vocab_size}")
    tokenizer_vocab_size = len(tokenizer)
    print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open(dir_path + "/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open(dir_path + "/config/dataset2maxlen.json", "r"))
    # predict on each dataset
    for dataset in datasets:
        print(f"Loading dataset {dataset}")
        if args.e:
            data = load_dataset('THUDM/LongBench', dataset+'_e', split='test')
            output_path = f"{output_dir}/pred_e/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            output_path = f"{output_dir}/pred/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device)
        with open(output_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')
