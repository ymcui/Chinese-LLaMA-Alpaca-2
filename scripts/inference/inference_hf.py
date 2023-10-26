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
parser.add_argument('--lora_model', default=None, type=str, help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path', default=None, type=str)
parser.add_argument('--data_file', default=None, type=str, help="A file that contains instructions (one instruction per line)")
parser.add_argument('--with_prompt', action='store_true', help="wrap the input with the prompt automatically")
parser.add_argument('--interactive', action='store_true', help="run in the instruction mode (single-turn)")
parser.add_argument('--predictions_file', default='./predictions.json', type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
parser.add_argument('--alpha', type=str, default="1.0", help="The scaling factor of NTK method, can be a float or 'auto'. ")
parser.add_argument('--load_in_8bit', action='store_true', help="Load the LLM in the 8bit mode")
parser.add_argument('--load_in_4bit', action='store_true', help="Load the LLM in the 4bit mode")
parser.add_argument("--use_vllm", action='store_true', help="Use vLLM as back-end LLM service.")
parser.add_argument('--system_prompt', type=str, default=DEFAULT_SYSTEM_PROMPT, help="The system prompt of the prompt template.")
parser.add_argument('--negative_prompt', type=str, default=None, help="Negative prompt in CFG sampling.")
parser.add_argument('--guidance_scale', type=float, default=1.0, help="The guidance scale for CFG sampling. CFG is enabled by setting `guidance_scale > 1`.")
parser.add_argument('--speculative_sampling', action='store_true', help="Use speculative sampling to speed up inference.")
parser.add_argument('--draft_k', type=int, default=-1, help="Number of new tokens the draft model generates each times. Should be a positive integer. Using adaptive number K if `draft_k <= 0`.")
parser.add_argument('--draft_base_model', default=None, type=str, help="Draft base model used in speculative sampling.")
parser.add_argument('--draft_lora_model', default=None, type=str, help="If None, perform inference on the draft base model")
parser.add_argument('--draft_model_load_in_8bit', action='store_true', help="Load the draft model in the 8bit mode")
parser.add_argument('--draft_model_load_in_4bit', action='store_true', help="Load the draft model in the 4bit mode")
parser.add_argument('--flash_attn', action='store_true', help="Use flash attention to replace the LLaMA attention")
args = parser.parse_args()

if args.guidance_scale > 1:
    try:
        from transformers.generation import UnbatchedClassifierFreeGuidanceLogitsProcessor
    except ImportError:
        raise ImportError("Please install the latest transformers (commit equal or later than d533465) to enable CFG sampling.")

if args.use_vllm:
    if args.lora_model is not None:
        raise ValueError("vLLM currently does not support LoRA, please merge the LoRA weights to the base model.")
    if args.load_in_8bit or args.load_in_4bit:
        raise ValueError("vLLM currently does not support quantization, please use fp16 (default) or unuse --use_vllm.")
    if args.only_cpu:
        raise ValueError("vLLM requires GPUs with compute capability not less than 7.0. If you want to run only on CPU, please unuse --use_vllm.")
    if args.guidance_scale > 1:
        raise ValueError("guidance_scale > 1, but vLLM does not support CFG sampling. Please unset guidance_scale. ")
    if args.speculative_sampling:
        raise ValueError("speculative_sampling is set, but vLLM does not support speculative sampling. Please unset speculative_sampling. ")
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("Only one quantization method can be chosen for inference. Please check your arguments")
if args.only_cpu is True:
    args.gpus = ""
    if args.load_in_8bit or args.load_in_4bit:
        raise ValueError("Quantization is unavailable on CPU.")

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
from peft import  PeftModel
if args.use_vllm:
    from vllm import LLM, SamplingParams

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
if not args.only_cpu:
    if args.flash_attn:
        from flash_attn_patch_for_inference import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
    else:
        from attn_and_long_ctx_patches import apply_attention_patch
        apply_attention_patch(use_memory_efficient_attention=True)
from attn_and_long_ctx_patches import apply_ntk_scaling_patch
apply_ntk_scaling_patch(args.alpha)
if args.speculative_sampling:
    if args.draft_base_model == None:
        raise ValueError("Speculative sampling requires a draft model. Please specify the draft model.")
    if args.draft_model_load_in_8bit and args.draft_model_load_in_4bit:
        raise ValueError("Only one quantization method can be chosen for inference. Please check your arguments")
    from speculative_sample import speculative_sample

if args.use_vllm:
    generation_config = dict(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        max_tokens=400,
        presence_penalty=1.0,
    )
else:
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

def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
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

    if args.use_vllm:
        model = LLM(model=args.base_model,
            tokenizer=args.tokenizer_path,
            tokenizer_mode='slow',
            tensor_parallel_size=len(args.gpus.split(',')))
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, legacy=True)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, legacy=True)
        if args.load_in_4bit or args.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                bnb_4bit_compute_dtype=load_type,
            )

        base_model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=load_type,
            low_cpu_mem_usage=True,
            device_map='auto',
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            quantization_config=quantization_config if (args.load_in_4bit or args.load_in_8bit) else None
        )

        if args.speculative_sampling:
            if args.load_in_4bit or args.load_in_8bit:
                draft_quantization_config = BitsAndBytesConfig(
                    load_in_4bit=args.draft_model_load_in_4bit,
                    load_in_8bit=args.draft_model_load_in_8bit,
                    bnb_4bit_compute_dtype=load_type,
                )
            draft_base_model = LlamaForCausalLM.from_pretrained(
                args.draft_base_model,
                torch_dtype=load_type,
                low_cpu_mem_usage=True,
                device_map='auto',
                load_in_4bit=args.draft_model_load_in_4bit,
                load_in_8bit=args.draft_model_load_in_8bit,
                quantization_config=draft_quantization_config if (args.draft_model_load_in_4bit or args.draft_model_load_in_8bit) else None
            )

        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenizer_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
        if model_vocab_size!=tokenizer_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenizer_vocab_size)
        if args.speculative_sampling:
            draft_model_vocab_size = draft_base_model.get_input_embeddings().weight.size(0)
            print(f"Vocab of the draft base model: {draft_model_vocab_size}")
            if draft_model_vocab_size!=tokenizer_vocab_size:
                print("Resize draft model embeddings to fit tokenizer")
                draft_base_model.resize_token_embeddings(tokenizer_vocab_size)
        if args.lora_model is not None:
            print("loading peft model")
            model = PeftModel.from_pretrained(base_model, args.lora_model,torch_dtype=load_type,device_map='auto',).half()
        else:
            model = base_model
        if args.speculative_sampling:
            if args.draft_lora_model is not None:
                print("loading peft draft model")
                draft_model = PeftModel.from_pretrained(draft_base_model, args.draft_lora_model,torch_dtype=load_type,device_map='auto',).half()
            else:
                draft_model = draft_base_model

        if device==torch.device('cpu'):
            model.float()
        model.eval()
        if args.speculative_sampling:
            if device==torch.device('cpu'):
                draft_model.float()
            draft_model.eval()

    # test data
    if args.data_file is None:
        examples = sample_data
    else:
        with open(args.data_file,'r') as f:
            examples = [l.strip() for l in f.readlines()]
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)

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
                    input_text = generate_prompt(instruction=raw_input_text, system_prompt=args.system_prompt)
                    negative_text = None if args.negative_prompt is None \
                        else generate_prompt(instruction=raw_input_text, system_prompt=args.negative_prompt)
                else:
                    input_text = raw_input_text
                    negative_text = args.negative_prompt

                if args.use_vllm:
                    output = model.generate([input_text], SamplingParams(**generation_config), use_tqdm=False)
                    response = output[0].outputs[0].text
                else:
                    inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
                    if args.guidance_scale ==1:
                        if not args.speculative_sampling:
                            generation_output = model.generate(
                                input_ids = inputs["input_ids"].to(device),
                                attention_mask = inputs['attention_mask'].to(device),
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                generation_config = generation_config
                            )
                        else: # enable speculative sampling
                            generation_output = speculative_sample(
                                input_ids=inputs["input_ids"].to(device),
                                target_model=model,
                                draft_model=draft_model,
                                draft_k=args.draft_k,
                                generation_config=generation_config,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                    else: # enable CFG sampling
                        if negative_text is None:
                            negative_prompt_ids = None
                            negative_prompt_attention_mask = None
                        else:
                            negative_inputs = tokenizer(negative_text,return_tensors="pt")
                            negative_prompt_ids = negative_inputs["input_ids"].to(device)
                            negative_prompt_attention_mask = negative_inputs["attention_mask"].to(device)
                        if not args.speculative_sampling:
                            generation_output = model.generate(
                                input_ids = inputs["input_ids"].to(device),
                                attention_mask = inputs['attention_mask'].to(device),
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                generation_config = generation_config,
                                guidance_scale = args.guidance_scale,
                                negative_prompt_ids = negative_prompt_ids,
                                negative_prompt_attention_mask = negative_prompt_attention_mask
                            )
                        else: # enable speculative sampling
                            generation_output = speculative_sample(
                                input_ids=inputs["input_ids"].to(device),
                                target_model=model,
                                draft_model=draft_model,
                                draft_k=args.draft_k,
                                generation_config=generation_config,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                guidance_scale=args.guidance_scale,
                                negative_prompt_ids=negative_prompt_ids,
                                negative_prompt_attention_mask=negative_prompt_attention_mask,
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
            if args.use_vllm:
                if args.with_prompt is True:
                    inputs = [generate_prompt(example, system_prompt=args.system_prompt) for example in examples]
                else:
                    inputs = examples
                outputs = model.generate(inputs, SamplingParams(**generation_config))

                for index, (example, output) in enumerate(zip(examples, outputs)):
                    response = output.outputs[0].text

                    print(f"======={index}=======")
                    print(f"Input: {example}\n")
                    print(f"Output: {response}\n")

                    results.append({"Input":example,"Output":response})

            else:
                for index, example in enumerate(examples):
                    if args.with_prompt:
                        input_text = generate_prompt(instruction=example, system_prompt=args.system_prompt)
                        negative_text = None if args.negative_prompt is None else \
                            generate_prompt(instruction=example, system_prompt=args.negative_prompt)
                    else:
                        input_text = example
                        negative_text = args.negative_prompt
                    inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
                    if args.guidance_scale == 1:
                        if not args.speculative_sampling:
                            generation_output = model.generate(
                                input_ids = inputs["input_ids"].to(device),
                                attention_mask = inputs['attention_mask'].to(device),
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                generation_config = generation_config
                            )
                        else: # enable speculative sampling
                            generation_output = speculative_sample(
                                input_ids=inputs["input_ids"].to(device),
                                target_model=model,
                                draft_model=draft_model,
                                draft_k=args.draft_k,
                                generation_config=generation_config,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                    else: # enable CFG sampling
                        if negative_text is None:
                            negative_prompt_ids = None
                            negative_prompt_attention_mask = None
                        else:
                            negative_inputs = tokenizer(negative_text,return_tensors="pt")
                            negative_prompt_ids = negative_inputs["input_ids"].to(device)
                            negative_prompt_attention_mask = negative_inputs["attention_mask"].to(device)
                        if not args.speculative_sampling:
                            generation_output = model.generate(
                                input_ids = inputs["input_ids"].to(device),
                                attention_mask = inputs['attention_mask'].to(device),
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                generation_config = generation_config,
                                guidance_scale = args.guidance_scale,
                                negative_prompt_ids = negative_prompt_ids,
                                negative_prompt_attention_mask = negative_prompt_attention_mask
                            )
                        else: # enable speculative sampling
                            generation_output = speculative_sample(
                                input_ids=inputs["input_ids"].to(device),
                                target_model=model,
                                draft_model=draft_model,
                                draft_k=args.draft_k,
                                generation_config=generation_config,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                guidance_scale=args.guidance_scale,
                                negative_prompt_ids=negative_prompt_ids,
                                negative_prompt_attention_mask=negative_prompt_attention_mask,
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
            if args.use_vllm:
                with open(dirname+'/generation_config.json','w') as f:
                    json.dump(generation_config,f,ensure_ascii=False,indent=2)
            else:
                generation_config.save_pretrained('./')
