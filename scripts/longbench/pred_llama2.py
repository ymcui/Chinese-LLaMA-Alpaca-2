# The script is modified from https://github.com/THUDM/LongBench/blob/main/pred.py
from datasets import load_dataset, load_from_disk
import torch
import json
from transformers import LlamaTokenizer, LlamaForCausalLM
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
parser.add_argument('--with_inst', choices=['true','false','auto'], default = 'auto')


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
            if dataset not in ["lcc", "repobench-p", "trec", "nq", "triviaqa", "lsht"]: # chat models are better off without build prompt on these tasks
                prompt = fill_llama2_prompt_template(instruction=prompt)
        elif args.with_inst == 'true':
            prompt = fill_llama2_prompt_template(instruction=prompt, with_inst = True)
        else:
            prompt = fill_llama2_prompt_template(instruction=prompt, with_inst = False)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        output = model.generate(
            **input,
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
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"]})
    return preds


if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    en_datasets = [ "hotpotqa","2wikimqa", "musique", "narrativeqa",
                    "qasper", "multifieldqa_en",  "gov_report",
                    "qmsum", "trec", "nq", "triviaqa",
                    "passage_count", "passage_retrieval_en"]
    zh_datasets = [ "dureader", "multifieldqa_zh", 
                    "vcsum","lsht", "passage_retrieval_zh"]
    code_datasets = [ "lcc", "repobench-p" ]

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
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=load_type)
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
    if not os.path.exists(f"{output_dir}/pred"):
        os.makedirs(f"{output_dir}/pred")
    for dataset in datasets:
        print(f"Loading dataset {dataset}")
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device)
        with open(f"{output_dir}/pred/{dataset}.jsonl", "w") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')
