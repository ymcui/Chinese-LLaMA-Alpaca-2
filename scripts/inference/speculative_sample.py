from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, PreTrainedModel
from transformers import (
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.generation.streamers import BaseStreamer
import torch
from typing import Tuple, List, Optional
import copy


def norm_logits(
    x: torch.Tensor,
    logits: torch.Tensor,
    logits_processor: LogitsProcessorList,
    logits_warper: LogitsProcessorList,
    do_sample: bool = False,
    cur_len=None,
) -> torch.Tensor:
    """
    Args:
        x (`torch.Tensor`): input ids, shape (batch, seqlen)
        logits `(`torch.Tensor`): shape (batch, seqlen, vocab)
        do_sample ('bool'): whether do sample
        logits_processor (`LogitsProcessorList`, *optional*):
            Custom logits processors that complement the default logits processors built from arguments and
            generation config. If a logit processor is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        logits_warper (`LogitsProcessorList`): An instance of [`LogitsProcessorList`]. List of instances of class derived from 
            [`LogitsWarper`] used to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step.
        do_sample ('boo;'): whether do sample.
        cur_len ('int'): length of current decoded tokens.

    Returns:
        `torch.Tensor`: probs with shape as (batch, seq_len)
    """
    new_logits = logits[:,:]
    if len(logits_processor) > 0:
        for i in range(x.shape[1]-cur_len+1):
            new_logits[:,i,:] = logits_processor(x[:,:cur_len+i], new_logits[:,i,:])
    if do_sample and len(logits_warper) > 0:
        for i in range(x.shape[1]-cur_len+1):
            new_logits[:,i,:] = logits_warper(x[:,:cur_len+i], new_logits[:,i,:])

    probs = new_logits.softmax(dim=-1)

    return probs


def sample(probs : torch.Tensor, do_sample : bool = False, num_samples: int = 1):
    if do_sample:
        new_token = torch.multinomial(probs, num_samples=num_samples)
    else:
        new_token = torch.argmax(probs, keepdim=True)
    return new_token


def max_fn(x):
    """
    norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum


def _draft_model_serial_forward(
    prefix : torch.Tensor,
    draft_k : int,
    draft_model : torch.nn.Module,
    logits_processor,
    logits_warper,
    do_sample=False,
    past_key_values=None,
    rejected=False,
    eos_token_id_tensor = None
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor or bool]:
    """ forward draft model draft_k times

    Args:
        prefix (`torch.Tensor`): the original input ids
        draft_k (`int`): how many times draft model forward and sample
        draft_model (`torch.nn.Module`): an draft model
        logits_processor (`LogitsProcessorList`, *optional*): Custom logits processors that complement the default logits processors built from arguments and
            generation config.
        logits_warper: List of instances of class derived from [`LogitsWarper`] used to warp the prediction score distribution
        do_sample (`bool`): whether do sample
        past_key_values: kv cache of draft model in last iteration
        rejected (`bool`): whether any of tokens in last iteration was rejected
        eos_token_id_tensor (`torch.Tensor`): eos token id in tokenizer

    Returns:
        Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor or bool]: 
            generated tokens, probability distribution of draft model's output, 
            past_key_values of draft model, flag of whether last token is eos
    """
    x = prefix
    x = x.to(draft_model.device)
    input_ids = x
    probs = None

    if past_key_values != None:
        if rejected == False:
            output = draft_model(input_ids[:,-2:-1], past_key_values = past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            input_ids = input_ids[:,-1:]
            probs = norm_logits(x[:,:-1], output.logits, logits_processor, logits_warper, do_sample, x.shape[1]-1)
        else:
            input_ids = input_ids[:,-1:]

    for _ in range(draft_k):
        output = draft_model(input_ids, past_key_values = past_key_values, use_cache=True)
        new_probs = norm_logits(x, output.logits[:,-1:], logits_processor, logits_warper, do_sample, x.shape[1])
        next_tok = sample(new_probs[:, -1, :], do_sample=do_sample)
        if eos_token_id_tensor is not None:
            last_token_is_eos = next_tok.tile(eos_token_id_tensor.shape[0], 1)
            last_token_is_eos = (
                ~last_token_is_eos.ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0).bool()
            )
            if last_token_is_eos:
                break
        else:
            last_token_is_eos = False
        past_key_values = output.past_key_values
        probs = torch.cat((probs, new_probs), dim=1) if probs != None else torch.cat((output.logits[:,:-1], new_probs), dim=1)
        input_ids = next_tok
        x = torch.cat((x, next_tok), dim=1)

    return x, probs, past_key_values, last_token_is_eos

def _speculative_sampling(
    prefix : torch.Tensor,
    target_model : torch.nn.Module,
    draft_model : torch.nn.Module,
    max_new_tokens : int ,
    draft_k : int = 4,
    logits_processor: LogitsProcessorList = None,
    logits_warper : LogitsProcessorList = None,
    do_sample = False,
    eos_token_id = None,
    stopping_criteria = None,
    streamer: Optional["BaseStreamer"] = None,
) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318

    Args:
        prefix (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        target_model (torch.nn.Module): target model, the large one
        draft_model (torch.nn.Module): draft model, the small one
        max_new_tokens (int): the max overall generated tokens number.
        draft_k (int): the token number small model guesses.
        logits_processor (`LogitsProcessorList`, *optional*): Custom logits processors that complement the default logits processors built from arguments and
            generation config.
        logits_warper: List of instances of class derived from [`LogitsWarper`] used to warp the prediction score distribution
        do_sample (`bool`): whether do sample
        eos_token_id: eos token id in tokenizer
        stopping_criteria: An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    input_seq_len = prefix.shape[1]
    T = input_seq_len + max_new_tokens
    assert prefix.shape[0] == 1, "input batch size must be 1"

    if draft_k <= 0:
        draft_k = 4
        adaptive_k = True
    else:
        adaptive_k = False

    draft_past_key_values = None
    draft_probs = None
    target_past_key_values = None
    target_probs = None
    rejected = False
    unfinished_sequences = prefix.new(prefix.shape[0]).fill_(1)

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(prefix.device) if eos_token_id is not None else None
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    while prefix.shape[1] < T:
        prefix_len = prefix.shape[1]
        x, new_draft_probs, draft_past_key_values, _ = _draft_model_serial_forward(
            prefix,
            draft_k,
            draft_model,
            logits_processor,
            logits_warper,
            do_sample,
            draft_past_key_values,
            rejected,
            eos_token_id_tensor
        )

        if draft_probs != None and new_draft_probs != None:
            draft_probs = torch.concat((draft_probs, new_draft_probs), dim=1)
        elif new_draft_probs == None:
            draft_probs = draft_probs
        else:
            draft_probs = new_draft_probs

        if target_past_key_values != None:
            unchecked_token_count = x.shape[1] - target_probs.shape[1] - 1
            outputs = target_model(x[:,-(unchecked_token_count+1):], past_key_values=target_past_key_values, use_cache=True)
        else:
            unchecked_token_count = x.shape[1] - prefix_len
            outputs = target_model(x, use_cache=True)
        new_target_probs = norm_logits(x, outputs.logits[:,-(unchecked_token_count+1):], logits_processor, logits_warper, do_sample, prefix_len)
        target_probs = torch.cat((target_probs, new_target_probs), dim=1) if target_probs != None else torch.cat((outputs.logits[:,:-(unchecked_token_count+1)], new_target_probs), dim=1)
        target_past_key_values = outputs.past_key_values

        # n_valid: the length of the valid prefix
        is_all_accept = True
        n_valid = prefix_len
        for i in range(unchecked_token_count):
            r = torch.rand(1, device = target_probs.device)
            cur_token_id = x[:, prefix_len + i]
            cur_pos = prefix_len + i - 1

            if r < torch.min(
                torch.tensor([1], device=draft_probs.device),
                target_probs[:, cur_pos, cur_token_id] / draft_probs[:, cur_pos, cur_token_id]
            ):
                # accept, and update n_valid
                n_valid += 1
            else:
                # reject
                target_new_token = sample(
                    max_fn(
                        target_probs[:, n_valid-1, :] - draft_probs[:, n_valid-1, :]
                    ), do_sample=do_sample
                )
                is_all_accept = False
                rejected = True
                break

        n_valid = min(n_valid, T - 1)
        prefix = x[:, :n_valid]

        if is_all_accept:
            target_new_token = sample(target_probs[:, -1, :], do_sample=do_sample)
            rejected = False
        else:
            draft_probs = draft_probs[:,:n_valid,:]
            target_probs = target_probs[:,:n_valid,:]
            if "bloom" in draft_model.__class__.__name__.lower() or (
                draft_model.config.architectures is not None and "bloom" in draft_model.config.architectures[0].lower()
            ):
                draft_past_key_values = [
                    (key[:,:,:n_valid], value[:,:n_valid,:])
                    for key,value in draft_past_key_values
                ]
                target_past_key_values = [
                    (key[:,:,:n_valid], value[:,:n_valid,:])
                    for key,value in target_past_key_values
                ]
            else:
                draft_past_key_values = [
                    (key[:,:,:n_valid,:], value[:,:,:n_valid,:])
                    for key,value in draft_past_key_values
                ]
                target_past_key_values = [
                    (key[:,:,:n_valid,:], value[:,:,:n_valid,:])
                    for key,value in target_past_key_values
                ]
        if adaptive_k:
            if is_all_accept:
                draft_k += 2
            else:
                draft_k = max(1, draft_k - 1)
        prefix = torch.cat((prefix, target_new_token), dim=1)
        if streamer is not None:
            streamer.put(prefix.cpu())
        if stopping_criteria(prefix, target_probs):
            # this_peer_finished = True
            break
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                prefix[:, -1]
                .tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )
            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                # this_peer_finished = True
                break

    if streamer is not None:
        streamer.end()

    return prefix


def speculative_sample(
    input_ids,
    target_model: Optional["PreTrainedModel"],
    draft_model: Optional["PreTrainedModel"],
    generation_config: GenerationConfig,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    draft_k: int = 4,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    streamer: Optional["BaseStreamer"] = None,
    **kwargs,
):
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
    generation_config.validate()

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    inputs_tensor, _, model_kwargs = target_model._prepare_model_inputs(
        input_ids, generation_config.bos_token_id, model_kwargs
    )

    model_kwargs["use_cache"] = generation_config.use_cache

    input_ids_seq_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
    #     warnings.warn(
    #         f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
    #         "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
    #         " recommend using `max_new_tokens` to control the maximum length of the generation.",
    #         UserWarning,
    #     )
        pass
    elif generation_config.max_new_tokens is not None:
        # if not has_default_max_length:
        #     logger.warning(
        #         f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
        #         f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
        #         "Please refer to the documentation for more information. "
        #         "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
        #     )
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

    if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
        raise ValueError(
            f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
            f" the maximum length ({generation_config.max_length})"
        )
    if input_ids_seq_length >= generation_config.max_length:
        # input_ids_string = "decoder_input_ids" if target_model.config.is_encoder_decoder else "input_ids"
        # logger.warning(
        #     f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
        #     f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
        #     " increasing `max_new_tokens`."
        # )
        pass
    # prepare logis_processor, stopping_criteria, logits_warper
    try:
        logits_processor = target_model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
    except TypeError:
        # Please install the latest transformers (commit equal or later than d533465) to enable CFG sampling.
        logits_processor = target_model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
        )
    stopping_criteria = target_model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    logits_warper=target_model._get_logits_warper(generation_config) if generation_config.do_sample else None

    outputs = _speculative_sampling(
        prefix=input_ids,
        target_model=target_model,
        draft_model=draft_model,
        max_new_tokens=generation_config.max_new_tokens,
        draft_k=draft_k,
        logits_processor=logits_processor,
        logits_warper=logits_warper,
        do_sample=generation_config.do_sample,
        eos_token_id=generation_config.eos_token_id,
        stopping_criteria=stopping_criteria,
        streamer=streamer,
    )

    return outputs


if __name__ == "__main__":
    # A usage example
    draft_model_name = 'Draft/Model/Path'
    target_model_name = 'Target/Model/Path'

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

    TEMPLATE = (
        "[INST] <<SYS>>\n"
        "{system_prompt}\n"
        "<</SYS>>\n\n"
        "{instruction} [/INST]"
    )

    def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
        return TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})

    inputs = ["我能用lightning数据线给安卓手机充电吗？"]

    negative_text = generate_prompt(inputs[0], system_prompt="回复尽可能多的内容。")
    inputs = [generate_prompt(text) for text in inputs]

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)

    print("begin loading models")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto',
        load_in_8bit=False
    )
    draft_model.resize_token_embeddings(len(tokenizer))
    print(f"Load {draft_model_name}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto',
        load_in_8bit=False
    )
    print(f"Load {target_model_name}")
    draft_model.eval()
    target_model.eval()
    print("finish loading models")

    torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    input_ids = tokenizer.encode(inputs[0], return_tensors='pt').to(torch_device)

    negative_inputs = tokenizer(negative_text,return_tensors="pt")
    negative_prompt_ids = negative_inputs["input_ids"].to(torch_device)
    negative_prompt_attention_mask = negative_inputs["attention_mask"].to(torch_device)

    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.1,
        max_new_tokens=128
    )

    outputs = speculative_sample(
        input_ids=input_ids,
        target_model=target_model,
        draft_model=draft_model,
        generation_config=generation_config,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # draft_k=4,
        # guidance_scale=1.5,
        # negative_prompt_ids=negative_prompt_ids,
        # negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
