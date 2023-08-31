import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
from peft import  PeftModel
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict

model = LlamaForCausalLM.from_pretrained('/Users/yangziqing/Documents/projects/llama/test/weights/llama_tiny_test',torch_dtype=torch.float16)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","v_proj",'embed_tokens','lm_head'],
    inference_mode=False,
    r=8, lora_alpha=4,
    lora_dropout=0,)
    #modules_to_save=['embed_tokens','lm_head'])
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()