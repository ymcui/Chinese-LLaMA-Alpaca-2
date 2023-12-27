import torch
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    StoppingCriteria,
    BitsAndBytesConfig,
    GenerationConfig
)
import gradio as gr
import argparse
import os
from queue import Queue
from threading import Thread
import traceback
import gc
import json
import requests
from typing import Iterable, List
import subprocess
import re

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

TEMPLATE_WITH_SYSTEM_PROMPT = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

TEMPLATE_WITHOUT_SYSTEM_PROMPT = "[INST] {instruction} [/INST]"

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--base_model',
    default=None,
    type=str,
    required=True,
    help='Base model path')
parser.add_argument('--lora_model', default=None, type=str,
                    help="If None, perform inference on the base model")
parser.add_argument(
    '--tokenizer_path',
    default=None,
    type=str,
    help='If None, lora model path or base model path will be used')
parser.add_argument(
    '--gpus',
    default="0",
    type=str,
    help='If None, cuda:0 will be used. Inference using multi-cards: --gpus=0,1,... ')
parser.add_argument('--share', default=True, help='Share gradio domain name')
parser.add_argument('--port', default=19324, type=int, help='Port of gradio demo')
parser.add_argument(
    '--max_memory',
    default=1024,
    type=int,
    help='Maximum number of input tokens (including system prompt) to keep. If exceeded, earlier history will be discarded.')
parser.add_argument(
    '--load_in_8bit',
    action='store_true',
    help='Use 8 bit quantized model')
parser.add_argument(
    '--load_in_4bit',
    action='store_true',
    help='Use 4 bit quantized model')
parser.add_argument(
    '--only_cpu',
    action='store_true',
    help='Only use CPU for inference')
parser.add_argument(
    '--alpha',
    type=str,
    default="1.0",
    help="The scaling factor of NTK method, can be a float or 'auto'. ")
parser.add_argument(
    "--use_vllm",
    action='store_true',
    help="Use vLLM as back-end LLM service.")
parser.add_argument(
    "--post_host",
    type=str,
    default="0.0.0.0",
    help="Host of vLLM service.")
parser.add_argument(
    "--post_port",
    type=int,
    default=8000,
    help="Port of vLLM service.")
parser.add_argument(
    "--speculative_sampling",
    action='store_true',
    help="Use speculative sampling to speed up inference.")
parser.add_argument(
    "--draft_base_model",
    default=None,
    type=str,
    help="Draft base model used in speculative sampling.")
parser.add_argument(
    "--draft_lora_model",
    default=None,
    type=str,
    help="If None, perform inference on the draft base model")
parser.add_argument(
    "--draft_model_load_in_8bit",
    action='store_true',
    help="Load the draft model in the 8bit mode")
parser.add_argument(
    "--draft_model_load_in_4bit",
    action='store_true',
    help="Load the draft model in the 4bit mode")
parser.add_argument(
    '--use_flash_attention_2',
    action='store_true',
    help="Use flash attention to replace the LLaMA attention")
parser.add_argument('--use_ntk', action='store_true', help="Use dynamic-ntk to extend context window")
args = parser.parse_args()

ENABLE_CFG_SAMPLING = True
try:
    from transformers.generation import UnbatchedClassifierFreeGuidanceLogitsProcessor
except ImportError:
    ENABLE_CFG_SAMPLING = False
    print("Install the latest transformers (commit equal or later than d533465) to enable CFG sampling.")
if args.use_vllm is True:
    print("CFG sampling is disabled when using vLLM.")
    ENABLE_CFG_SAMPLING = False

if args.only_cpu is True:
    args.gpus = ""
    if args.load_in_8bit or args.load_in_4bit:
        raise ValueError("Quantization is unavailable on CPU.")
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("Only one quantization method can be chosen for inference. Please check your arguments")
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
if not args.only_cpu:
    if args.use_flash_attention_2:
        from flash_attn_patch_for_inference import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
    else:
        from attn_and_long_ctx_patches import apply_attention_patch
        apply_attention_patch(use_memory_efficient_attention=True)
from attn_and_long_ctx_patches import apply_ntk_scaling_patch
if args.use_ntk:
    apply_ntk_scaling_patch(args.alpha)
if args.speculative_sampling:
    if args.draft_base_model == None:
        raise ValueError("Speculative sampling requires a draft model. Please specify the draft model.")
    if args.draft_model_load_in_8bit and args.draft_model_load_in_4bit:
        raise ValueError("Only one quantization method can be chosen for inference. Please check your arguments")
    from speculative_sample import speculative_sample

# Set CUDA devices if available
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


# Peft library can only import after setting CUDA devices
from peft import PeftModel


# Set up the required components: model and tokenizer

def setup():
    global tokenizer, model, device, share, port, max_memory
    if args.speculative_sampling:
        global draft_model
    if args.use_vllm:
        # global share, port, max_memory
        max_memory = args.max_memory
        port = args.port
        share = args.share == 'True' or args.share is True

        if args.lora_model is not None:
            raise ValueError("vLLM currently does not support LoRA, please merge the LoRA weights to the base model.")
        if args.load_in_8bit or args.load_in_4bit:
            raise ValueError("vLLM currently does not support quantization, please use fp16 (default) or unuse --use_vllm.")
        if args.only_cpu:
            raise ValueError("vLLM requires GPUs with compute capability not less than 7.0. If you want to run only on CPU, please unuse --use_vllm.")
        if args.speculative_sampling:
            raise ValueError("speculative_sampling is set, but vLLM does not support speculative sampling. Please unset speculative_sampling. ")

        if args.tokenizer_path is None:
            args.tokenizer_path = args.base_model
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, legacy=True)

        print("Start launch vllm server.")
        cmd = f"python -m vllm.entrypoints.api_server \
            --model={args.base_model} \
            --tokenizer={args.tokenizer_path} \
            --tokenizer-mode=slow \
            --tensor-parallel-size={len(args.gpus.split(','))} \
            --host {args.post_host} \
            --port {args.post_port} \
            &"
        subprocess.check_call(cmd, shell=True)
    else:
        max_memory = args.max_memory
        port = args.port
        share = args.share == 'True' or args.share is True
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
        if args.load_in_4bit or args.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                bnb_4bit_compute_dtype=load_type,
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=load_type,
            low_cpu_mem_usage=True,
            device_map='auto',
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            quantization_config=quantization_config if (args.load_in_4bit or args.load_in_8bit) else None,
            trust_remote_code=True
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
        if model_vocab_size != tokenizer_vocab_size:
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
            model = PeftModel.from_pretrained(
                base_model,
                args.lora_model,
                torch_dtype=load_type,
                device_map='auto',
            ).half()
        else:
            model = base_model
        if args.speculative_sampling:
            if args.draft_lora_model is not None:
                print("loading peft draft model")
                draft_model = PeftModel.from_pretrained(draft_base_model, args.draft_lora_model,torch_dtype=load_type,device_map='auto',).half()
            else:
                draft_model = draft_base_model

        if device == torch.device('cpu'):
            model.float()
        model.eval()
        if args.speculative_sampling:
            if device==torch.device('cpu'):
                draft_model.float()
            draft_model.eval()


# Reset the user input
def reset_user_input():
    return gr.update(value='')


# Reset the state
def reset_state():
    return []


def generate_prompt(instruction, response="", with_system_prompt=True, system_prompt=DEFAULT_SYSTEM_PROMPT):
    if with_system_prompt is True:
        prompt = TEMPLATE_WITH_SYSTEM_PROMPT.format_map({'instruction': instruction,'system_prompt': system_prompt})
    else:
        prompt = TEMPLATE_WITHOUT_SYSTEM_PROMPT.format_map({'instruction': instruction})
    if len(response)>0:
        prompt += " " + response
    return prompt


# User interaction function for chat
def user(user_message, history):
    return gr.update(value="", interactive=False), history + \
        [[user_message, None]]


class Stream(StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:
    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).

    Adapted from: https://stackoverflow.com/a/9969000
    """
    def __init__(self, func, kwargs=None, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs or {}
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except Exception:
                traceback.print_exc()

            clear_torch_cache()
            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __del__(self):
        clear_torch_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
        clear_torch_cache()


def clear_torch_cache():
    gc.collect()
    if torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      top_p: float = 0.9,
                      top_k: int = 40,
                      temperature: float = 0.2,
                      max_tokens: int = 512,
                      presence_penalty: float = 1.0,
                      use_beam_search: bool = False,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "n": n,
        "top_p": 1 if use_beam_search else top_p,
        "top_k": -1 if use_beam_search else top_k,
        "temperature": 0 if use_beam_search else temperature,
        "max_tokens": max_tokens,
        "use_beam_search": use_beam_search,
        "best_of": 5 if use_beam_search else n,
        "presence_penalty": presence_penalty,
        "stream": stream,
    }
    print(pload)

    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


# Perform prediction based on the user input and history
@torch.no_grad()
def predict(
    history,
    system_prompt,
    negative_prompt,
    max_new_tokens=128,
    top_p=0.9,
    temperature=0.2,
    top_k=40,
    do_sample=True,
    repetition_penalty=1.1,
    guidance_scale=1.0,
    presence_penalty=0.0,
    draft_k=0,
):
    if len(system_prompt) == 0:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    while True:
        print("len(history):", len(history))
        print("history: ", history)
        history[-1][1] = ""
        if len(history) == 1:
            input = history[0][0]
            prompt = generate_prompt(input,response="", with_system_prompt=True, system_prompt=system_prompt)
        else:
            input = history[0][0]
            response = history[0][1]
            prompt = generate_prompt(input, response=response, with_system_prompt=True, system_prompt=system_prompt)+'</s>'
            for hist in history[1:-1]:
                input = hist[0]
                response = hist[1]
                prompt = prompt + '<s>'+generate_prompt(input, response=response, with_system_prompt=False)+'</s>'
            input = history[-1][0]
            prompt = prompt + '<s>'+generate_prompt(input, response="", with_system_prompt=False)

        input_length = len(tokenizer.encode(prompt, add_special_tokens=True))
        print(f"Input length: {input_length}")
        if input_length > max_memory and len(history) > 1:
            print(f"The input length ({input_length}) exceeds the max memory ({max_memory}). The earlier history will be discarded.")
            history = history[1:]
            print("history: ", history)
        else:
            break

    if args.use_vllm:
        generate_params = {
            'max_tokens': max_new_tokens,
            'top_p': top_p,
            'temperature': temperature,
            'top_k': top_k,
            "use_beam_search": not do_sample,
            'presence_penalty': presence_penalty,
        }

        api_url = f"http://{args.post_host}:{args.post_port}/generate"


        response = post_http_request(prompt, api_url, **generate_params, stream=True)

        for h in get_streaming_response(response):
            for line in h:
                line = line.replace(prompt, '')
                history[-1][1] = line
                yield history

    else:
        negative_text = None
        if len(negative_prompt) != 0:
            negative_text = re.sub(r"<<SYS>>\n(.*)\n<</SYS>>", f"<<SYS>>\n{negative_prompt}\n<</SYS>>", prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        if negative_text is None:
            negative_prompt_ids = None
            negative_prompt_attention_mask = None
        else:
            negative_inputs = tokenizer(negative_text,return_tensors="pt")
            negative_prompt_ids = negative_inputs["input_ids"].to(device)
            negative_prompt_attention_mask = negative_inputs["attention_mask"].to(device)
        generate_params = {
            'input_ids': input_ids,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'temperature': temperature,
            'top_k': top_k,
            'do_sample': do_sample,
            'repetition_penalty': repetition_penalty,
            'eos_token_id': tokenizer.eos_token_id,
        }
        if ENABLE_CFG_SAMPLING is True:
            generate_params['guidance_scale'] = guidance_scale
            generate_params['negative_prompt_ids'] = negative_prompt_ids
            generate_params['negative_prompt_attention_mask'] = negative_prompt_attention_mask
        if args.speculative_sampling:
            generate_params['target_model'] = model
            generate_params['draft_model'] = draft_model
            generate_params['draft_k'] = draft_k
            generate_params['generation_config'] = GenerationConfig()

        def generate_with_callback(callback=None, **kwargs):
            if 'stopping_criteria' in kwargs:
                kwargs['stopping_criteria'].append(Stream(callback_func=callback))
            else:
                kwargs['stopping_criteria'] = [Stream(callback_func=callback)]
            clear_torch_cache()
            with torch.no_grad():
                if not args.speculative_sampling:
                    model.generate(**kwargs)
                else: # enable speculative sampling
                    speculative_sample(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(generate_with_callback, kwargs, callback=None)

        with generate_with_streaming(**generate_params) as generator:
            for output in generator:
                next_token_ids = output[len(input_ids[0]):]
                if next_token_ids[0] == tokenizer.eos_token_id:
                    break
                new_tokens = tokenizer.decode(
                    next_token_ids, skip_special_tokens=True)
                if isinstance(tokenizer, LlamaTokenizer) and len(next_token_ids) > 0:
                    if tokenizer.convert_ids_to_tokens(int(next_token_ids[0])).startswith('▁'):
                        new_tokens = ' ' + new_tokens

                history[-1][1] = new_tokens
                yield history
                if len(next_token_ids) >= max_new_tokens:
                    break


# Call the setup function to initialize the components
setup()


# Create the Gradio interface
with gr.Blocks() as demo:
    github_banner_path = 'https://raw.githubusercontent.com/ymcui/Chinese-LLaMA-Alpaca-2/main/pics/banner.png'
    gr.HTML(f'<p align="center"><a href="https://github.com/ymcui/Chinese-LLaMA-Alpaca-2"><img src={github_banner_path} width="700"/></a></p>')
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=3):
                system_prompt_input = gr.Textbox(
                    show_label=True,
                    label="系统提示语（仅在对话开始前或清空历史后修改有效，对话过程中修改无效）",
                    placeholder=DEFAULT_SYSTEM_PROMPT,
                    lines=1).style(
                    container=True)
                negative_prompt_input = gr.Textbox(
                    show_label=True,
                    label="反向提示语（仅在对话开始前或清空历史后修改有效，对话过程中修改无效）",
                    placeholder="（可选，默认为空）",
                    lines=1,
                    visible=ENABLE_CFG_SAMPLING).style(
                    container=True)
            with gr.Column(scale=12):
                user_input = gr.Textbox(
                    show_label=True,
                    label="用户指令",
                    placeholder="Shift + Enter发送消息...",
                    lines=10).style(
                    container=True)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_new_token = gr.Slider(
                0,
                4096,
                value=512,
                step=1.0,
                label="Maximum New Token Length",
                interactive=True)
            top_p = gr.Slider(0, 1, value=0.9, step=0.01,
                              label="Top P", interactive=True)
            temperature = gr.Slider(
                0,
                1,
                value=0.2,
                step=0.01,
                label="Temperature",
                interactive=True)
            top_k = gr.Slider(1, 40, value=40, step=1,
                              label="Top K", interactive=True)
            do_sample = gr.Checkbox(
                value=True,
                label="Do Sample",
                info="use random sample strategy",
                interactive=True)
            repetition_penalty = gr.Slider(
                1.0,
                3.0,
                value=1.1,
                step=0.1,
                label="Repetition Penalty",
                interactive=True,
                visible=False if args.use_vllm else True)
            guidance_scale = gr.Slider(
                1.0,
                3.0,
                value=1.0,
                step=0.1,
                label="Guidance Scale",
                interactive=True,
                visible=ENABLE_CFG_SAMPLING)
            presence_penalty = gr.Slider(
                -2.0,
                2.0,
                value=1.0,
                step=0.1,
                label="Presence Penalty",
                interactive=True,
                visible=True if args.use_vllm else False)
            draft_k = gr.Slider(
                0,
                10,
                value=0,
                step=1.0,
                label="Draft K",
                interactive=True,
                visible=args.speculative_sampling==True)

    params = [user_input, chatbot]
    predict_params = [
        chatbot,
        system_prompt_input,
        negative_prompt_input,
        max_new_token,
        top_p,
        temperature,
        top_k,
        do_sample,
        repetition_penalty,
        guidance_scale,
        presence_penalty,
        draft_k]

    submitBtn.click(
        user,
        params,
        params,
        queue=False).then(
        predict,
        predict_params,
        chatbot).then(
            lambda: gr.update(
                interactive=True),
        None,
        [user_input],
        queue=False)

    user_input.submit(
        user,
        params,
        params,
        queue=False).then(
        predict,
        predict_params,
        chatbot).then(
            lambda: gr.update(
                interactive=True),
        None,
        [user_input],
        queue=False)

    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot], show_progress=True)


# Launch the Gradio interface
demo.queue().launch(
    share=share,
    inbrowser=True,
    server_name='0.0.0.0',
    server_port=port)
