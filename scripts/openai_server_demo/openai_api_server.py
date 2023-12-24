import argparse
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from threading import Thread
from sse_starlette.sse import EventSourceResponse


parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str,help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path',default=None,type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--load_in_8bit',action='store_true', help='Load the model in 8bit mode')
parser.add_argument('--load_in_4bit',action='store_true', help='Load the model in 4bit mode')
parser.add_argument('--only_cpu',action='store_true',help='Only use CPU for inference')
parser.add_argument('--alpha',type=str,default="1.0", help="The scaling factor of NTK method, can be a float or 'auto'. ")
parser.add_argument('--use_ntk', action='store_true', help="Use dynamic-ntk to extend context window")
parser.add_argument('--use_flash_attention_2', action='store_true', help="Use flash-attention2 to accelerate inference")
args = parser.parse_args()
if args.only_cpu is True:
    args.gpus = ""
    if args.load_in_8bit or args.load_in_4bit:
        raise ValueError("Quantization is unavailable on CPU.")
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("Only one quantization method can be chosen for inference. Please check your arguments")
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
    BitsAndBytesConfig
)
from peft import PeftModel

import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from attn_and_long_ctx_patches import apply_attention_patch, apply_ntk_scaling_patch

apply_attention_patch(use_memory_efficient_attention=True)
if args.use_ntk:
    apply_ntk_scaling_patch(args.alpha)

from openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
)

load_type = torch.float16
if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device("cpu")
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
    device_map='auto' if not args.only_cpu else None,
    load_in_4bit=args.load_in_4bit,
    load_in_8bit=args.load_in_8bit,
    quantization_config=quantization_config if (args.load_in_4bit or args.load_in_8bit) else None,
    use_flash_attention_2=args.use_flash_attention_2,
    trust_remote_code=True
)

model_vocab_size = base_model.get_input_embeddings().weight.size(0)
tokenizer_vocab_size = len(tokenizer)
print(f"Vocab of the base model: {model_vocab_size}")
print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
if model_vocab_size != tokenizer_vocab_size:
    print("Resize model embeddings to fit tokenizer")
    base_model.resize_token_embeddings(tokenizer_vocab_size)
if args.lora_model is not None:
    print("loading peft model")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_model,
        torch_dtype=load_type,
        device_map="auto",
    )
else:
    model = base_model

if device == torch.device("cpu"):
    model.float()

model.eval()

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

TEMPLATE_WITH_SYSTEM_PROMPT = (
    "[INST] <<SYS>>\n" "{system_prompt}\n" "<</SYS>>\n\n" "{instruction} [/INST]"
)

TEMPLATE_WITHOUT_SYSTEM_PROMPT = "[INST] {instruction} [/INST]"


def generate_prompt(
    instruction, response="", with_system_prompt=True, system_prompt=None
):
    if with_system_prompt is True:
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        prompt = TEMPLATE_WITH_SYSTEM_PROMPT.format_map(
            {"instruction": instruction, "system_prompt": system_prompt}
        )
    else:
        prompt = TEMPLATE_WITHOUT_SYSTEM_PROMPT.format_map({"instruction": instruction})
    if len(response) > 0:
        prompt += " " + response
    return prompt


def generate_completion_prompt(instruction: str):
    """Generate prompt for completion"""
    return generate_prompt(instruction, response="", with_system_prompt=True)


def generate_chat_prompt(messages: list):
    """Generate prompt for chat completion"""

    system_msg = None
    for msg in messages:
        if msg.role == "system":
            system_msg = msg.content
    prompt = ""
    is_first_user_content = True
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "user":
            if is_first_user_content is True:
                prompt += generate_prompt(
                    msg.content, with_system_prompt=True, system_prompt=system_msg
                )
                is_first_user_content = False
            else:
                prompt += "<s>" + generate_prompt(msg.content, with_system_prompt=False)
        if msg.role == "assistant":
            prompt += f" {msg.content}" + "</s>"
    return prompt


def predict(
    input,
    max_new_tokens=128,
    top_p=0.9,
    temperature=0.2,
    top_k=40,
    num_beams=1,
    repetition_penalty=1.1,
    do_sample=True,
    **kwargs,
):
    """
    Main inference method
    type(input) == str -> /v1/completions
    type(input) == list -> /v1/chat/completions
    """
    if isinstance(input, str):
        prompt = generate_completion_prompt(input)
    else:
        prompt = generate_chat_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=do_sample,
        **kwargs,
    )
    generation_config.return_dict_in_generate = True
    generation_config.output_scores = False
    generation_config.max_new_tokens = max_new_tokens
    generation_config.repetition_penalty = float(repetition_penalty)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    output = output.split("[/INST]")[-1].strip()
    return output


def stream_predict(
    input,
    max_new_tokens=128,
    top_p=0.75,
    temperature=0.1,
    top_k=40,
    num_beams=4,
    repetition_penalty=1.0,
    do_sample=True,
    model_id="chinese-llama-alpaca-2",
    **kwargs,
):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        choices=[choice_data],
        object="chat.completion.chunk",
    )
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    if isinstance(input, str):
        prompt = generate_completion_prompt(input)
    else:
        prompt = generate_chat_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=do_sample,
        **kwargs,
    )

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        streamer=streamer,
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=False,
        max_new_tokens=max_new_tokens,
        repetition_penalty=float(repetition_penalty),
    )
    Thread(target=model.generate, kwargs=generation_kwargs).start()
    for new_text in streamer:
        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(content=new_text), finish_reason=None
        )
        chunk = ChatCompletionResponse(
            model=model_id, choices=[choice_data], object="chat.completion.chunk"
        )
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(), finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    yield "[DONE]"


def get_embedding(input):
    """Get embedding main function"""
    with torch.no_grad():
        encoding = tokenizer(input, padding=True, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        model_output = model(input_ids, attention_mask, output_hidden_states=True)
        data = model_output.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
        masked_embeddings = data * mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        seq_length = torch.sum(mask, dim=1)
        embedding = sum_embeddings / seq_length
        normalized_embeddings = F.normalize(embedding, p=2, dim=1)
        ret = normalized_embeddings.squeeze(0).tolist()
    return ret


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    msgs = request.messages
    if isinstance(msgs, str):
        msgs = [ChatMessage(role="user", content=msgs)]
    else:
        msgs = [ChatMessage(role=x["role"], content=x["content"]) for x in msgs]
    if request.stream:
        generate = stream_predict(
            input=msgs,
            max_new_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            num_beams=request.num_beams,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )
        return EventSourceResponse(generate, media_type="text/event-stream")
    output = predict(
        input=msgs,
        max_new_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        num_beams=request.num_beams,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
    )
    choices = [
        ChatCompletionResponseChoice(index=i, message=msg) for i, msg in enumerate(msgs)
    ]
    choices += [
        ChatCompletionResponseChoice(
            index=len(choices), message=ChatMessage(role="assistant", content=output)
        )
    ]
    return ChatCompletionResponse(choices=choices)


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Creates a completion"""
    output = predict(
        input=request.prompt,
        max_new_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        num_beams=request.num_beams,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
    )
    choices = [CompletionResponseChoice(index=0, text=output)]
    return CompletionResponse(choices=choices)


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingsRequest):
    """Creates text embedding"""
    embedding = get_embedding(request.input)
    data = [{"object": "embedding", "embedding": embedding, "index": 0}]
    return EmbeddingsResponse(data=data)


if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    uvicorn.run(app, host="0.0.0.0", port=19327, workers=1, log_config=log_config)
