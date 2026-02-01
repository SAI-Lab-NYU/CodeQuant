import argparse
import os
import random
import numpy as np
import yaml
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Union
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.rotation_utils import load_or_create_R1, create_optimizer
from utils.dataset_utils import CalibrationDataset
from utils.quantization_utils import activation_quantizer
from utils.rotation_utils import fuse_weight
from utils.model_utils import get_model


def rotation_fine_tune_hook(module_name: str,
                            mode: str,
                            R: nn.Module,
                            activation_quantization_bit: int,
                            group_size: int,
                            loss_dict: dict):

    def hook(module: nn.Module,
             input_args: Tuple[torch.Tensor, ...],
             output_tensor_or_tuple: Union[torch.Tensor, Tuple]) -> None:
        if mode == "output":
            if isinstance(output_tensor_or_tuple, torch.Tensor):
                rmsnorm_output = output_tensor_or_tuple.detach()
            elif isinstance(output_tensor_or_tuple, tuple):
                if len(output_tensor_or_tuple) > 0 and isinstance(output_tensor_or_tuple[0], torch.Tensor):
                    rmsnorm_output = output_tensor_or_tuple[0].detach()

            rmsnorm_output.requires_grad_(True)

            rot_output = rmsnorm_output.float() @ R.weight.to(rmsnorm_output.device)
            quant_rot_output = activation_quantizer((rot_output, ), group_size, activation_quantization_bit)
            loss_dict[module_name] = F.mse_loss(rot_output, quant_rot_output)
        elif mode == "input":
            if len(input_args) > 0 and isinstance(input_args[0], torch.Tensor):
                rmsnorm_input = input_args[0].detach()
            elif isinstance(input_args, torch.Tensor):
                rmsnorm_input= input_args.detach()

            rmsnorm_input.requires_grad_(True)
            # expand_R = torch.block_diag(*([R.weight] * num_attention_heads))
            expand_R = R.weight

            rot_input = rmsnorm_input.float() @ expand_R.to(rmsnorm_input.device)
            quant_rot_input = activation_quantizer((rot_input, ), group_size, activation_quantization_bit)
            loss_dict[module_name] = F.mse_loss(rot_input, quant_rot_input)
    return hook


def rotation_fine_tune(model: nn.Module,
                       model_type: str,
                       tokenizer: nn.Module,
                       processor: nn.Module,
                       input_group_size: int,
                       activation_quantization_bit: int,
                       dataset_name: str,
                       calibration_samples: int,
                       batch_size: int,
                       max_length: int,
                       epochs: int,
                       lr: float,
                       device: str):
    # create rotation matrix
    # R1
    R1 = load_or_create_R1(mode="online",
                           device=device,
                           dim=model.config.hidden_size)
    R1_optimizer = create_optimizer(R1, lr=lr)

    # register hook
    output_name_pattern = r".*(input_layernorm|post_attention_layernorm).*"
    output_regex = re.compile(output_name_pattern)

    R1_loss_dict = {}
    handles = {}
    for name, module in model.named_modules():
        if output_regex.search(name):
            hook = rotation_fine_tune_hook(name, "output", R1, activation_quantization_bit, input_group_size, R1_loss_dict)
            handles[name] = module.register_forward_hook(hook)
            print(f"[DEBUG] register output hook for {name}")

    # fine tune loop
    dataset = CalibrationDataset(dataset_name, calibration_samples)
    g = torch.Generator()
    g.manual_seed(42)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)

    model.eval()
    for _ in tqdm(range(epochs), leave=False):
        for batch in tqdm(dataloader, leave=False):
            R1_loss_dict.clear()
            if model_type == "qwen":
                processed = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    for text in batch
                ]
                inputs = tokenizer(
                    processed, return_tensors="pt",
                    padding=True, truncation=True, max_length=max_length
                ).to(device)
            elif model_type == "mixtral":
                inputs = tokenizer(
                    batch, return_tensors="pt",
                    padding=True, truncation=True, max_length=max_length
                ).to(device)
            else:
                processed = [
                    processor.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    for text in batch
                ]
                inputs = tokenizer(
                    processed, return_tensors="pt",
                    padding=True, truncation=True, max_length=max_length
                ).to(device)

            model(**inputs)

            R1_optimizer.zero_grad()

            gpu_count = torch.cuda.device_count()
            R1_loss = torch.stack([l.to(f"cuda:{gpu_count - 1}") for m, l in R1_loss_dict.items() if l is not None]).sum()

            # R1 backward
            R1_loss.backward()

            tqdm.write(f"[DEBUG] rotation matrix gradient: {torch.norm(R1.parametrizations.weight.original.grad)}")

            R1_optimizer.step()
    return R1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="input parser")
    parser.add_argument('--config', type=str, required=True, help='config file name')
    args = parser.parse_args()

    with open(f"../configs/{args.config}", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config is not None, "[ERROR] config is None."

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_name = config["model"]["model_name"]
    device = config["accelerator"]["device"]

    model, model_type, tokenizer, processor, model_config = get_model(model_name)

    # fuse norm weight into linear weight
    fuse_weight(model, model_type)
    print(f"[INFO] model {model_name} fused.")

    # params
    common_params = config["common_setting"]
    dataset_name = config["calibration"]["dataset_name"]
    rotate_params = config["rotation"]

    # path
    save_path = config["path"]["rotation_data_path"]
    os.makedirs(save_path, exist_ok=True)

    # run
    r1 = rotation_fine_tune(model=model,
                            model_type=model_type,
                            tokenizer=tokenizer,
                            processor=processor,
                            input_group_size=common_params["input_group_size"],
                            activation_quantization_bit=common_params["activation_quantization_bit"],
                            dataset_name=dataset_name,
                            calibration_samples=rotate_params["max_sample"],
                            batch_size=rotate_params["batch_size"],
                            max_length=rotate_params["max_length"],
                            epochs=rotate_params["epochs"],
                            lr=rotate_params["fine_tune_lr"],
                            device=device)

    # group size
    input_group_size = common_params["input_group_size"]
    if input_group_size == -1:
        postfix = "nongroup"
    else:
        postfix = "group"

    r1_ckpt = {
        "r1": r1.cpu().state_dict(),
        "dim": model.config.hidden_size,
    }
    torch.save(r1_ckpt, os.path.join(save_path, f"{model_type}_r1_{postfix}.pt"))
