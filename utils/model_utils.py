import torch
import torch.nn as nn

from typing import Dict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from accelerate import dispatch_model

model_prefix_dict = {
    "microsoft/Phi-mini-MoE-instruct": "phi",
    "Qwen/Qwen3-30B-A3B": "qwen",
    "mistralai/Mixtral-8x7B-v0.1": "mixtral",
    "deepseek-ai/DeepSeek-V2-Lite": "deepseek",
}


def make_gpu_map(num_layers: int, num_devices: int):
    dm = {"model.embed_tokens": 0}
    layers_per_device = num_layers // num_devices
    for i in range(num_layers):
        dev = min(i // layers_per_device, num_devices - 1)
        dm[f"model.layers.{i}"] = dev
    dm["model.norm"] = num_devices - 1
    dm["lm_head"] = num_devices - 1
    return dm


def split_gpu(model, num_devices: int):
    device_map = make_gpu_map(len(model.model.layers), num_devices)
    print(device_map)
    return dispatch_model(model, device_map=device_map)


def get_model(model_name: str,
              force_device: dict = None):
    model_type = model_prefix_dict[model_name]

    model_config = AutoConfig.from_pretrained(model_name, attn_implementation = "eager", trust_remote_code=True)
    
    print(f"[INFO] model config {model_name} loaded.")

    device_map = "auto" if force_device is None else force_device

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation = "eager",
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
        config=model_config
    )
    # model = split_gpu(model, 2) #optional
    print(model.device)
    model.config.attn_implementation = "eager"
    model.eval()
    print(f"[INFO] model {model_name} loaded.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if model_type == "mixtral":
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[INFO] tokenizer {model_name} loaded.")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print(f"[INFO] processor {model_name} loaded.")

    return model, model_type, tokenizer, processor, model_config


@torch.no_grad()
def replace_model_weight(model: nn.Module,
                         weight_dict: Dict[str, torch.Tensor]) -> nn.Module:
    for name, module in model.named_modules():
        if name in weight_dict:
            print(f"[INFO] replace {name} with quantized weight")
            module.weight.data = weight_dict[name].to(module.weight.device)

    return model

@torch.no_grad()
def replace_model_weight_rebuttal(
    model: nn.Module,
    weight_dict: Dict[str, torch.Tensor],
    # target_indices,
    verbose: bool = True
) -> nn.Module:
    target_indices = [24,25,26]
    target_lists = []
    for index in target_indices:
        target_lists.append(f"model.layers.{index}.")

    for name, module in model.named_modules():
        if name in weight_dict:
            flag = 0
            for item in target_lists:
                if item in name:
                    flag = 1
                    break
            
            if flag == 0:
                continue
                
            print(f"[INFO] replace {name} with quantized weight")
            module.weight.data = weight_dict[name].to(module.weight.device)

    return model