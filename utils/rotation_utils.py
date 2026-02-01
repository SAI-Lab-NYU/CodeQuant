import torch
import torch.nn as nn

from tqdm import tqdm


def load_or_create_R1(mode: str,
                      device: str,
                      save_dir: str = None,
                      dim: int = None):
    if mode == "offline":
        pre_compute_r1 = torch.load(save_dir, weights_only=False, map_location="cpu")
        pre_compute_dim = pre_compute_r1["dim"]
        R1 = nn.utils.parametrizations.orthogonal(nn.Linear(pre_compute_dim, pre_compute_dim, bias=False, dtype=torch.float32),
                                                  orthogonal_map="cayley")
        R1.load_state_dict(pre_compute_r1["r1"])

        return R1.to(device)
    elif mode == "online":
        assert dim is not None, "dim must be specified"

        R1 = nn.utils.parametrizations.orthogonal(nn.Linear(dim, dim, bias=False, dtype=torch.float32),
                                                  orthogonal_map="cayley")

        return R1.to(device)
    else:
        raise ValueError("Unknown mode")


def load_or_create_R2(mode: str,
                      model: nn.Module = None,
                      save_dir: str = None):
    if mode == "offline":
        print(save_dir)
        R2_dict = torch.load(save_dir, weights_only=False, map_location="cpu")["r2"]
        return R2_dict
    elif mode == "online":
        if ("MixtralForCausalLM" in type(model).__name__):
            dim = model.config.hidden_size // model.config.num_attention_heads
        if ("DeepseekV2ForCausalLM" in type(model).__name__):
            # dim = 4096
            dim = model.config.num_attention_heads * model.config.v_head_dim
        else:
            dim = model.config.head_dim
        R2_dict = {}
        for name, module in model.named_modules():
            if "o_proj" in name:
                print(f"[DEBUG] create R2 for {name}")
                R2 = nn.utils.parametrizations.orthogonal(nn.Linear(dim, dim, bias=False, dtype=torch.float32, device=module.weight.device),
                                                          orthogonal_map="cayley")
                print(R2.weight.data.size())
                R2_dict[name] = R2.to(module.weight.device)
        return R2_dict
    else:
        raise ValueError("Unknown mode")


def create_optimizer(matrix: nn.Module,
                     lr: float):
    optimizer = torch.optim.Adam([matrix.parametrizations.weight.original], lr=lr)
    return optimizer


def rotate_attention_vo_deepseek(layer, r2_dict, layer_name) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    o_proj_name = layer_name + "o_proj"
    W = layer.self_attn.o_proj
    dtype = W.weight.dtype
    device = W.weight.device
    R2 = r2_dict[o_proj_name].to(dtype=torch.float32, device=device)

    W_ = W.weight.to(device=device, dtype=torch.float32)
    out_feature1 = W_.size()[-1]
    W.weight.data = torch.matmul(W_, R2.T).to(device=device, dtype=dtype)

    W = layer.self_attn.kv_b_proj
    original_size = W.weight.data.size()
    W_complete = W.weight.to(device=device, dtype=torch.float32)
    W1 = W_complete.data[out_feature1:, :]
    W2 = W_complete.data[:out_feature1, :]

    W2_rotated = torch.matmul(R2, W2).to(device=device, dtype=dtype)
    W.weight.data = torch.cat([W1, W2_rotated], dim=0).to(device=device, dtype=dtype)
    assert W.weight.data.size() == original_size

    del W_


def rotate_attention_output(layer, rotation_cache, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == "deepseek":
        W = layer.self_attn.o_proj
    elif model_type == "qwen":
        W = layer.self_attn.o_proj
    elif model_type == "mixtral":
        W = layer.self_attn.o_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    device = W.weight.data.device

    R1 = rotation_cache[str(device)]

    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    R1 = R1.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device=device, dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device=device, dtype=dtype)
        del b
    del W_


def rotate_attention_inputs(layer, rotation_cache, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    if model_type == "deepseek":
        for W in [layer.self_attn.q_proj, layer.self_attn.kv_a_proj_with_mqa]:
            dtype = W.weight.dtype
            device = W.weight.device

            R1 = rotation_cache[str(device)]

            W_ = W.weight.to(device=device, dtype=torch.float64)
            R1 = R1.to(device=device, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, R1).to(device=device, dtype=dtype)
        del W_
    elif model_type == "qwen":
        for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
            dtype = W.weight.dtype
            device = W.weight.device

            R1 = rotation_cache[str(device)]

            W_ = W.weight.to(device=device, dtype=torch.float64)
            R1 = R1.to(device=device, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, R1).to(device=device, dtype=dtype)
        del W_
    elif model_type == "mixtral":
        for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
            dtype = W.weight.dtype
            device = W.weight.device

            R1 = rotation_cache[str(device)]

            W_ = W.weight.to(device=device, dtype=torch.float64)
            R1 = R1.to(device=device, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, R1).to(device=device, dtype=dtype)
        del W_


def rotate_mlp_input(layer, rotation_cache, model_type):
    # Rotate the MLP input weights.
    if model_type == "qwen":
        mlp_inputs = []
        for expert in layer.mlp.experts:
            mlp_inputs.append(expert.up_proj)
            mlp_inputs.append(expert.gate_proj)
        mlp_inputs.append(layer.mlp.gate)

    elif model_type == "mixtral":
        mlp_inputs = []
        for expert in layer.block_sparse_moe.experts:
            mlp_inputs.append(expert.w1)
            mlp_inputs.append(expert.w3)
        mlp_inputs.append(layer.block_sparse_moe.gate)

    elif model_type == "deepseek":
        mlp_inputs = []
        # 判断是否是 Deepseek MoE 模型
        if hasattr(layer.mlp, "shared_experts"):
            mlp_inputs.extend([
                layer.mlp.shared_experts.up_proj,
                layer.mlp.shared_experts.gate_proj,
            ])
            for expert in layer.mlp.experts:
                mlp_inputs.append(expert.up_proj)
                mlp_inputs.append(expert.gate_proj)
            mlp_inputs.append(layer.mlp.gate)
        else:
            mlp_inputs.append(layer.mlp.up_proj)
            mlp_inputs.append(layer.mlp.gate_proj)
    else:
        raise ValueError(f'Unknown model type {model_type}')

    for W in mlp_inputs:
        dtype = W.weight.dtype
        device = W.weight.device

        R1 = rotation_cache[str(device)]

        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        R1 = R1.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device=device, dtype=dtype)
        del W_


def rotate_mlp_output(layer, rotation_cache, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == "qwen":
        W = []
        for expert in layer.mlp.experts:
            W.append(expert.down_proj)
    elif model_type == "mixtral":
        W = []
        for expert in layer.block_sparse_moe.experts:
            W.append(expert.w2)
    elif model_type == "deepseek":
        if hasattr(layer.mlp, "shared_experts"):
            W = []
            for expert in layer.mlp.experts:
                W.append(expert.down_proj)
            W.append(layer.mlp.shared_experts.down_proj)
        else:
            W = layer.mlp.down_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    if isinstance(W, list):
        for w in W:
            dtype = w.weight.data.dtype
            device = w.weight.data.device

            R1 = rotation_cache[str(device)]

            W_ = w.weight.data.to(device=device, dtype=torch.float64)
            R1 = R1.to(device=device, dtype=torch.float64)
            w.weight.data = torch.matmul(R1.T, W_).to(device=device, dtype=dtype)
            # apply_exact_had_to_linear(w, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
            if w.bias is not None:
                b = w.bias.data.to(device=device, dtype=torch.float64)
                w.bias.data = torch.matmul(R1.T, b).to(device=device, dtype=dtype)
                del b
            del W_

    else:
        dtype = W.weight.data.dtype
        device = W.weight.data.device

        R1 = rotation_cache[str(device)]

        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        R1 = R1.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(R1.T, W_).to(device=device, dtype=dtype)
        # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
        if W.bias is not None:
            b = W.bias.data.to(device=device, dtype=torch.float64)
            W.bias.data = torch.matmul(R1.T, b).to(device=device, dtype=dtype)
            del b
        del W_


def rotate_embeddings(model, rotation_cache) -> None:
    # Rotate the embeddings.
    W = model.model.embed_tokens
    dtype = W.weight.data.dtype
    device = W.weight.data.device

    R1 = rotation_cache[str(device)]

    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    R1 = R1.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device=device, dtype=dtype)
    print("embedding rotated:", W_.size())
    del W_


def rotate_head(model, rotation_cache) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    device = W.weight.data.device

    R1 = rotation_cache[str(device)]

    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    R1 = R1.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device=device, dtype=dtype)
    del W_


def rotation_attention_vo(model,
                          model_type,
                          layer,
                          layer_name,
                          r2_dict):
    if model_type == "deepseek":
        v = layer.self_attn.v_proj
        o = layer.self_attn.o_proj
    elif model_type == "qwen":
        v = layer.self_attn.v_proj
        o = layer.self_attn.o_proj
    elif model_type == "mixtral":
        v = layer.self_attn.v_proj
        o = layer.self_attn.o_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = v.weight.data.dtype
    device = v.weight.data.device

    num_key_value_heads = model.config.num_key_value_heads
    num_attention_heads = model.config.num_attention_heads

    o_proj_name = layer_name + "o_proj"
    R2 = r2_dict[o_proj_name].to(dtype=dtype, device=device)

    R2_V = torch.block_diag(*([R2] * num_key_value_heads))
    R2_O = torch.block_diag(*([R2] * num_attention_heads))

    v.weight.data = (R2_V @ v.weight.data.to(R2_V.dtype)).to(device)
    o.weight.data = (o.weight.data.to(R2_O.dtype) @ R2_O.T).to(device)


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        # if verbos:
        #     logging.info(
        #         f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
        #         f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
        #     )


def fuse_rotation(model: nn.Module, model_type, r1_rotation_cache, r2_rotation_dict):
    """
    1) For each transformer layer:
       - Fuse input RMSNorm scale into q_proj, k_proj, v_proj.
       - Fuse post-attention/post-layer RMSNorm scale into:
           * MoE: router, and every expert.{up_proj, gate_proj}
           * Non-MoE MLP: mlp.{up_proj, gate_proj}  (down_proj is after nonlinearity; do NOT fuse there)
    2) Set those RMSNorm weights to ones.
    The RMS normalization itself remains active (only the affine scale is folded).
    """

    # config = model.config
    rotate_embeddings(model, r1_rotation_cache)
    rotate_head(model, r1_rotation_cache)
    cleanup_memory()
    layers = model.model.layers

    for idx in tqdm(range(len(layers)), unit="layer", desc="Rotating"):
        # if idx == 0:
        layer = layers[idx]
        rotate_attention_inputs(layer, r1_rotation_cache, model_type)
        rotate_attention_output(layer, r1_rotation_cache, model_type)

        if r2_rotation_dict is not None:
            if model_type == "deepseek":
                layer_name = f"model.layers.{idx}.self_attn."
                rotate_attention_vo_deepseek(layer, r2_rotation_dict, layer_name)
            else:
                if model_type == "qwen":
                    layer_name = f"model.layers.{idx}.self_attn."
                elif model_type == "mixtral":
                    layer_name = f"model.layers.{idx}.self_attn."
                rotation_attention_vo(model, model_type, layer, layer_name, r2_rotation_dict)

        # rotate_attention_b_input_partial(layer, Q_1, model.config.kv_lora_rank, model_type)
        # rotate_attention_b(layer, r2_rotation_dict, model_type)

        rotate_mlp_input(layer, r1_rotation_cache, model_type)
        rotate_mlp_output(layer, r1_rotation_cache, model_type)
        del layer
        cleanup_memory()
        # print(f"[Layer {idx}] CPU Memory usage: {get_memory_usage_mb():.2f} MB")


def _fuse_scale_into_linear(linear: nn.Linear, scale: torch.Tensor):
    """
    Fold an elementwise input scale (RMSNorm weight) into a Linear's weight.
    Linear.weight: (out_features, in_features)
    We need to scale columns by `scale` (shape [in_features]).
    """
    W = linear.weight.data
    if W.shape[1] != scale.numel():
        # Dimension mismatch; skip
        return
    # Broadcast scale across rows to multiply each input column
    W *= scale.unsqueeze(0).to(W.device, dtype=W.dtype)


def _set_rms_weight_ones(rms: nn.Module):
    """
    Set RMSNorm weight to ones if it exists.
    """
    w = getattr(rms, "weight", None)
    if isinstance(w, torch.Tensor):
        w.data.fill_(1.0)


def fuse_weight(model: nn.Module, model_name: str):
    # Try to locate the canonical stack: model.layers
    layers = model.model.layers

    for i, layer in enumerate(layers):

        # self attention
        if model_name == "qwen" or model_name == "deepseek" or model_name == "mixtral":
            in_norm = getattr(layer, "input_layernorm")
            # print(f"[DEBUG] input_layernorm {model_name}, {in_norm}")
        else:
            raise "do not support this model structure"

        attn = getattr(layer, "self_attn")
        scale = in_norm.weight.detach()

        if model_name == "qwen" or model_name == "mixtral":
            prof_lists = ["q_proj", "k_proj", "v_proj"]
        elif model_name == "deepseek":
            prof_lists = ["q_proj", "kv_a_proj_with_mqa"]
        else:
            raise "do not support this model structure"

        for proj_name in prof_lists:
            proj = getattr(attn, proj_name)
            _fuse_scale_into_linear(proj, scale)
        _set_rms_weight_ones(in_norm)

        # mlp
        if model_name == "qwen":
            post_norm = getattr(layer, "post_attention_layernorm")
            scale = post_norm.weight.detach()

            moe = getattr(layer, "mlp")
            router = getattr(moe, "gate")
            if isinstance(router, nn.Linear):
                _fuse_scale_into_linear(router, scale)

            experts = getattr(moe, "experts")
            for expert in experts:
                up = getattr(expert, "up_proj")
                gate = getattr(expert, "gate_proj")
                if isinstance(up, nn.Linear):
                    _fuse_scale_into_linear(up, scale)
                if isinstance(gate, nn.Linear):
                    _fuse_scale_into_linear(gate, scale)
            _set_rms_weight_ones(post_norm)
        elif model_name == "mixtral":
            post_norm = getattr(layer, "post_attention_layernorm")
            scale = post_norm.weight.detach()

            moe = getattr(layer, "block_sparse_moe")
            router = getattr(moe, "gate")
            if isinstance(router, nn.Linear):
                _fuse_scale_into_linear(router, scale)

            experts = getattr(moe, "experts")
            for expert in experts:
                up = getattr(expert, "w1")
                gate = getattr(expert, "w3")
                if isinstance(up, nn.Linear):
                    _fuse_scale_into_linear(up, scale)
                if isinstance(gate, nn.Linear):
                    _fuse_scale_into_linear(gate, scale)
            _set_rms_weight_ones(post_norm)
        elif model_name == "deepseek":
            post_norm = getattr(layer, "post_attention_layernorm")
            scale = post_norm.weight.detach()

            # MoE branch (Qwen-MoE style): layer.block_sparse_moe.{router, experts[*].{up_proj, gate_proj}}
            moe = getattr(layer, "mlp")
            if i == 0:
                up = getattr(moe, "up_proj")
                gate = getattr(moe, "gate_proj")
                _fuse_scale_into_linear(up, scale)
                _fuse_scale_into_linear(gate, scale)
                _set_rms_weight_ones(post_norm)
            else:
                router = getattr(moe, "gate")
                _fuse_scale_into_linear(router, scale)

                experts = getattr(moe, "experts")
                for expert in experts:
                    up = getattr(expert, "up_proj")
                    gate = getattr(expert, "gate_proj")
                    _fuse_scale_into_linear(up, scale)
                    _fuse_scale_into_linear(gate, scale)

                ## shared_expert
                shared_expert = getattr(moe, "shared_experts")
                up = getattr(shared_expert, "up_proj")
                gate = getattr(shared_expert, "gate_proj")
                _fuse_scale_into_linear(up, scale)
                _fuse_scale_into_linear(gate, scale)
                _set_rms_weight_ones(post_norm)

    lang_model = model.model
    norm = getattr(lang_model, "norm")
    scale = norm.weight.detach()
    proj = getattr(model, "lm_head")
    _fuse_scale_into_linear(proj, scale)
    _set_rms_weight_ones(norm)

    torch.cuda.empty_cache()