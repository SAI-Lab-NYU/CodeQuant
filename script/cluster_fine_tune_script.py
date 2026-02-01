import argparse
import random
import numpy as np
import yaml
import os
import re
import gc
import torch
import torch.nn as nn

from typing import Dict, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer

from cluster.ClusterLinear import ClusterLinear, ClusterLinear_deepseekGate
from cluster.ClusterMoE import ClusterMoE, ClusterMoE_deepseek
from utils.dataset_utils import CalibrationDataset
from utils.rotation_utils import fuse_rotation, fuse_weight, load_or_create_R1
from utils.model_utils import get_model
from utils.permutation_utils import permutation


def replace_moe(model: nn.Module,
                model_type: str) -> Dict[str, ClusterMoE]:
    if model_type == "phi":
        pattern_str = r"block_sparse_moe"
    elif model_type == "qwen":
        pattern_str = r"mlp"
    elif model_type == "mixtral":
        pattern_str = r"block_sparse_moe"
    elif model_type == "deepseek":
        pattern_str = r"mlp"
    else:
        raise ValueError(f'Unknown model type {model_type}')

    pattern = re.compile(pattern_str)

    wrapped_moes = {}

    if model_type == "deepseek":
        layers = model.model.layers
        for i, mod in enumerate(layers):
            if i == 0:
                continue
            attr_name = "mlp"
            full_name = f"model.model.layers.{i}"
            sub = getattr(mod, attr_name)
            if isinstance(sub, nn.Module):
                wrapped_moe = ClusterMoE_deepseek(
                    config=model.config,
                    moe_module=sub,
                    calibration_mode=True,
                ).to(next(sub.parameters(), torch.tensor(0.)).device)
                setattr(mod, attr_name, wrapped_moe)

                k = f"{full_name}.{attr_name}" if full_name else attr_name
                wrapped_moes[k] = wrapped_moe

                print(f"[DEBUG] Wrapped MoE -> ClusterMoE_deeepseek: {k}")

    else:
        for full_name, mod in model.named_modules():
            for attr_name in dir(mod):
                if not pattern.search(attr_name):
                    continue

                try:
                    sub = getattr(mod, attr_name)
                except Exception:
                    continue

                if isinstance(sub, nn.Module):
                    wrapped_moe = ClusterMoE(
                        moe_module=sub,
                        calibration_mode=True,
                        layer_name=f"{full_name}.{attr_name}" if full_name else attr_name
                    ).to(next(sub.parameters(), torch.tensor(0.)).device)
                    setattr(mod, attr_name, wrapped_moe)

                    k = f"{full_name}.{attr_name}" if full_name else attr_name
                    wrapped_moes[k] = wrapped_moe

                    print(f"[DEBUG] Wrapped MoE -> ClusterMoE: {k}")

    return wrapped_moes


def replace_linear(model: nn.Module,
                   model_type: str,
                   group_size: int,
                   cluster_num: int,
                   lr: float,
                   pre_codebooks: tuple = None) -> dict:
    if model_type == "phi":
        sa_names = {"q_proj", "k_proj", "v_proj"}
        ffn_names = {"gate", "w1", "w3"}
    elif model_type == "qwen":
        sa_names = {"q_proj", "k_proj", "v_proj"}
        ffn_names = {"gate", "gate_proj", "up_proj"}
    elif model_type == "mixtral":
        sa_names = {"q_proj", "k_proj", "v_proj"}
        ffn_names = {"gate", "w1", "w3"}
    elif model_type == "deepseek":
        sa_names = {"q_proj", "kv_a_proj_with_mqa"}
        ffn_names = {"gate_proj", "up_proj"}
    else:
        raise ValueError(f'Unknown model type {model_type}')

    replaced_linear = {}

    pre_compute_cluster = False
    cluster_weight, centroids, assignments = None, None, None
    if pre_codebooks is not None:
        pre_compute_cluster = True
        cluster_weight, centroids, assignments = pre_codebooks

    if model_type == "deepseek":
        layers = model.model.layers
        for i, layer in enumerate(layers):
            if i == 0:
                continue

            mod = layer.mlp
            full_name = f"model.model.layers.{i}.mlp"
            attr = "gate"
            lin = getattr(mod, "gate")

            w = lin.weight
            b = None
            module_type = "ffn"

            if pre_compute_cluster:
                print('[DEBUG] use pre_compute_cluster')
                cur_cluster_weight = cluster_weight[f"{full_name}.{attr}" if full_name else attr]
                cur_centroids = centroids[f"{full_name}.{attr}" if full_name else attr]
                cur_assignments = assignments[f"{full_name}.{attr}" if full_name else attr]
            else:
                cur_cluster_weight, cur_centroids, cur_assignments = None, None, None

            new_lin = ClusterLinear_deepseekGate(
                weight=w,
                config=model.config,
                bias=b,
                group_size=group_size,
                cluster_num=cluster_num,
                calibration_mode=True,
                module_type=module_type,
                layer_name=f"{full_name}.{attr}" if full_name else attr,
                pre_compute_cluster=pre_compute_cluster,
                pre_cluster_weight=cur_cluster_weight,
                pre_centroids=cur_centroids,
                pre_assignments=cur_assignments,
                pre_mask=None, 
                lr=lr,
                orig_gate=lin,
            ).to(lin.weight.device).to(lin.weight.dtype)
            setattr(mod, attr, new_lin)

            name_key = f"{full_name}.{attr}" if full_name else attr
            replaced_linear[name_key] = new_lin

            print(f"[INFO] Replaced Deepseek Gate -> ClusterLinear_deepseekGate: layer {i}")

    for full_name, mod in model.named_modules():
        for attr in list(sa_names | ffn_names):
            if not hasattr(mod, attr):
                continue
            lin = getattr(mod, attr)

            if not isinstance(lin, nn.Linear):
                continue

            with torch.no_grad():
                w = lin.weight.detach()
                b = lin.bias.detach() if lin.bias is not None else None

            setattr(mod, attr, None)
            del lin
            torch.cuda.empty_cache()

            module_type = "sa" if attr in sa_names else "ffn"

            if pre_compute_cluster:
                cur_cluster_weight = cluster_weight[f"{full_name}.{attr}" if full_name else attr]
                cur_centroids = centroids[f"{full_name}.{attr}" if full_name else attr]
                cur_assignments = assignments[f"{full_name}.{attr}" if full_name else attr]
            else:
                cur_cluster_weight, cur_centroids, cur_assignments = None, None, None

            new_lin = ClusterLinear(
                weight=w,
                bias=b,
                group_size=group_size,
                cluster_num=cluster_num,
                calibration_mode=True,
                module_type=module_type,
                layer_name=f"{full_name}.{attr}" if full_name else attr,
                pre_compute_cluster=pre_compute_cluster,
                pre_cluster_weight=cur_cluster_weight,
                pre_centroids=cur_centroids,
                pre_assignments=cur_assignments,
                lr=lr
            ).to(w.device).to(w.dtype)
            setattr(mod, attr, new_lin)

            name_key = f"{full_name}.{attr}" if full_name else attr
            replaced_linear[name_key] = new_lin

            torch.cuda.empty_cache()

            print(f"[INFO] Replaced Linear -> ClusterLinear: {full_name} ({module_type})")

    return replaced_linear


def freeze_parameters(model: nn.Module):
    for n, p in model.named_parameters():
        if n.endswith("cluster_weight") or n.endswith("centroids"):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)


def set_calibration_mode(replaced_linears: Dict[str, ClusterLinear],
                         wrapped_moes: Dict[str, ClusterMoE],
                         calibration_mode: bool):
    for m in replaced_linears.values():
        m.set_calibration_mode(calibration_mode)
    for m in wrapped_moes.values():
        m.set_calibration_mode(calibration_mode)


def export_artifacts(replaced_linears: Dict[str, ClusterLinear]) -> Tuple[dict, dict, dict]:
    cluster_weight_dict, centroids_dict, assignments_dict = {}, {}, {}
    for name, m in replaced_linears.items():
        cluster_weight_dict[name] = m.cluster_weight.detach().cpu()
        centroids_dict[name] = m.centroids.detach().cpu()
        a = m.assignments
        assignments_dict[name] = a.detach().cpu() if torch.is_tensor(a) else a
    return cluster_weight_dict, centroids_dict, assignments_dict


def export_codebooks(linear_modules: dict):
    cw, cent, asg = {}, {}, {}
    for name, m in linear_modules.items():
        cw[name] = m.cluster_weight.detach().cpu()
        cent[name] = m.centroids.detach().cpu()
        asg[name] = m.assignments.detach().cpu() if torch.is_tensor(m.assignments) else m.assignments
    return cw, cent, asg


def save_codebooks(path_prefix: str, cw, cent, asg):
    torch.save(cw, f"{path_prefix}_cluster_weight.pt")
    torch.save(cent, f"{path_prefix}_centroids.pt")
    torch.save(asg, f"{path_prefix}_assignments.pt")


def load_codebooks(path_prefix: str):
    cw = torch.load(f"{path_prefix}_cluster_weight.pt", map_location="cpu")
    cent = torch.load(f"{path_prefix}_centroids.pt", map_location="cpu")
    asg = torch.load(f"{path_prefix}_assignments.pt", map_location="cpu")
    return cw, cent, asg


def load_codebooks_mixtral(path_prefix: str, part):
    cw = torch.load(f"{path_prefix}_cluster_weight_{part}.pt", map_location="cpu")
    cent = torch.load(f"{path_prefix}_centroids_{part}.pt", map_location="cpu")
    asg = torch.load(f"{path_prefix}_assignments_{part}.pt", map_location="cpu")
    return cw, cent, asg


def clustering(model: nn.Module,
               model_type: str,
               processor: AutoProcessor,
               tokenizer: AutoTokenizer,
               weight_group_size: int,
               cluster_num: int,
               dataset_name: str,
               calibration_samples: int,
               batch_size: int,
               max_length: int,
               fine_tune_lr: float,
               save_dir_cluster: str,
               epochs: int,
               device: str,
               pre_compute_codebooks: bool = False):
    # replace linear layer to do fine tune
    if not pre_compute_codebooks:
        replaced_linear = replace_linear(model=model,
                                         model_type=model_type,
                                         group_size=weight_group_size,
                                         cluster_num=cluster_num,
                                         lr=fine_tune_lr)
        cw, cent, asg = export_codebooks(replaced_linear)
        print(f"[INFO] saving codebooks to {save_dir_cluster}")
        save_codebooks(os.path.join(save_dir_cluster, f"{model_type}"), cw, cent, asg)
    else:
        print(f"[INFO] loading codebooks from {save_dir_cluster}")
        pre = load_codebooks(os.path.join(save_dir_cluster, f"{model_type}"))
        replaced_linear = replace_linear(model=model,
                                         model_type=model_type,
                                         group_size=weight_group_size,
                                         cluster_num=cluster_num,
                                         lr=fine_tune_lr,
                                         pre_codebooks=pre)
        del pre
    wrapped_moes = replace_moe(model, model_type)

    gc.collect()
    torch.cuda.empty_cache()

    freeze_parameters(model)

    dataset = CalibrationDataset(dataset_name, calibration_samples)
    g = torch.Generator()
    g.manual_seed(42)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)

    model.eval()
    for _ in tqdm(range(epochs), leave=False):
        for batch in tqdm(dataloader, leave=False):
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

            set_calibration_mode(replaced_linear, wrapped_moes, True)  # calibration mode = True
            with torch.no_grad():
                _ = model(**inputs)

            set_calibration_mode(replaced_linear, wrapped_moes, False)  # calibration mode = False
            _ = model(**inputs)

            for lin in replaced_linear.values():
                lin.fine_tune_centroids()

            del inputs

    cluster_weight_dict, centroids_dict, assignments_dict = export_artifacts(replaced_linear)

    return cluster_weight_dict, centroids_dict, assignments_dict


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

    # load model
    model_name = config["model"]["model_name"]
    device = config["accelerator"]["device"]

    model, model_type, tokenizer, processor, model_config = get_model(model_name)

    fuse_weight(model, model_type)
    print(f"[INFO] model {model_name} fused.")

    # group size
    input_group_size = config["common_setting"]["input_group_size"]
    if input_group_size == -1:
        postfix = "nongroup"
    else:
        postfix = "group"

    # load rotation R1
    rotation_save_path = config["path"]["rotation_data_path"]
    R1_save_dir = os.path.join(rotation_save_path, f"{model_type}_r1_{postfix}.pt")
    R1 = load_or_create_R1(mode="offline", device=device, save_dir=R1_save_dir)
    R1 = R1.weight.detach()

    # cache R1 on each GPU
    gpu_count = torch.cuda.device_count()
    R1_per_gpu = {}
    if gpu_count > 1:
        print(f"[INFO] Detected {gpu_count} GPUs, creating R1 copies...")
        for i in range(gpu_count):
            R1_per_gpu[f"cuda:{i}"] = R1.to(f"cuda:{i}")
            print(f"[INFO] R1 copied to cuda:{i}")
    else:
        R1_per_gpu["cuda:0"] = R1.to(device=device)

    # fuse rotation matrix
    fuse_rotation(model, model_type, R1_per_gpu, None)
    print(f"[INFO] model {model_name} rotation matrix fused.")
    del R1, R1_per_gpu
    gc.collect()
    torch.cuda.empty_cache()

    ## permutation
    if config["cluster"]["permutation"]:
        permutation(model, model_type, model_config, config["common_setting"]["weight_group_size"])
        # permute_deepseek(model=model, ratio=0.25, final_group_size=config["common_setting"]["weight_group_size"], config=model_config)

    # path
    clustering_save_path = config["path"]["cluster_data_path"]
    os.makedirs(clustering_save_path, exist_ok=True)

    # params
    common_params = config["common_setting"]
    cluster_params = config["cluster"]

    # weight group size
    weight_group_size = config["common_setting"]["weight_group_size"]
    if weight_group_size == -1:
        cluster_postfix = "nongroup"
    else:
        cluster_postfix = "group"

    # save names
    clustering_weight_save_path = os.path.join(clustering_save_path, f"{model_type}_clustering_weight_dict_{postfix}.pt")
    centroid_save_path = os.path.join(clustering_save_path, f"{model_type}_centroid_dict_{postfix}.pt")
    assignment_save_path = os.path.join(clustering_save_path, f"{model_type}_assignment_dict_{postfix}.pt")

    # run
    cluster_weight_dict, centroid_dict, assignment_dict = clustering(model=model,
                                                                     model_type=model_type,
                                                                     processor=processor,
                                                                     tokenizer=tokenizer,
                                                                     weight_group_size=common_params["weight_group_size"],
                                                                     cluster_num=common_params["cluster_num"],
                                                                     dataset_name=config["calibration"]["dataset_name"],
                                                                     calibration_samples=cluster_params["max_sample"],
                                                                     batch_size=cluster_params["batch_size"],
                                                                     max_length=cluster_params["max_length"],
                                                                     fine_tune_lr=cluster_params["fine_tune_lr"],
                                                                     save_dir_cluster=clustering_save_path,
                                                                     epochs=cluster_params["epochs"],
                                                                     device=device)
    torch.save(cluster_weight_dict, clustering_weight_save_path)
    torch.save(centroid_dict, centroid_save_path)
    torch.save(assignment_dict, assignment_save_path)
    print(f"[INFO] clustering artifacts saved to {clustering_save_path}.")
