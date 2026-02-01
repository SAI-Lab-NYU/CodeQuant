import torch

from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import List

@torch.no_grad()
def kv_proj_deepseek(
        kv_a_proj_with_mqa,  # contains [W_k; W_v] stacked along out_features
        kv_b_proj,  # weight: [d_model, H*Dv]
        kv_a_layernorm,
        config,
        ratio: float = 0.25,
        final_group_size: int = 1024,
        centroid_number: int = 16,
):
    """
    Compute perms from o_proj per-head column blocks, then apply:
      - o_proj: per-head column permutation P_h
      - W_v:   per-head row inverse permutation P_h^{-1}
    Preserves the forward pass exactly.
    """

    compressed_kv, k_pe = torch.split(
        kv_a_proj_with_mqa, [config.kv_lora_rank, config.qk_rope_head_dim], dim=0
    )

    kv_b_proj.mul_(kv_a_layernorm)

    perm = group_stddev_metric_plain(kv_b_proj)
    compressed_kv_perm = compressed_kv[perm, :]
    kv_b_proj_perm = kv_b_proj[:, perm]

    kv_a_proj_with_mqa_perm = torch.cat([compressed_kv_perm, k_pe], dim=0)
    return kv_a_proj_with_mqa_perm, kv_b_proj_perm


@torch.no_grad()
def apply_equivariant_from_o_proj(
        v_proj,  # contains [W_k; W_v] stacked along out_features
        o_proj,  # weight: [d_model, H*Dv]
        config,
        ratio: float = 0.25,
        final_group_size: int = 1024,
        centroid_number: int = 16,
):
    """
    Compute perms from o_proj per-head column blocks, then apply:
      - o_proj: per-head column permutation P_h
      - W_v:   per-head row inverse permutation P_h^{-1}
    Preserves the forward pass exactly.
    """
    W_v = v_proj
    W_out = o_proj  # [d_model, H*Dv]
    n_rep = config.num_attention_heads // config.num_key_value_heads

    perms = []
    W_out_perm = []
    per_head_chunks = torch.split(W_out, W_out.size()[1] // config.num_key_value_heads, dim=1)
    for i, per_head_chunk in enumerate(per_head_chunks):
        chunks = torch.split(per_head_chunk, per_head_chunk.size()[1] // n_rep, dim=1)
        tmp_perm = group_stddev_metric_plain(chunks[0])
        perms.append(tmp_perm)
        for chunk in chunks:
            tmp = chunk[:, tmp_perm]
            W_out_perm.append(tmp)
    W_out_perm = torch.cat(W_out_perm, dim=1)
    assert W_out_perm.size() == W_out.size()

    W_v_chunks = torch.split(W_v, W_v.size()[0] // config.num_key_value_heads, dim=0)
    W_v_perm = []
    assert len(W_v_chunks) == len(perms)
    for i, W_v_chunk in enumerate(W_v_chunks):
        W_v_perm.append(W_v_chunk[perms[i], :])
    W_v_perm = torch.cat(W_v_perm, dim=0)

    return W_v_perm, W_out_perm

def group_stddev_metric_plain(weight: torch.Tensor, ratio=0.25, final_group_size: int = 1024, centroid_number=16) -> \
List[int]:
    """
    Sort columns of weight by column-wise absolute value sum, group into sets of columns,
    then compute average row-wise stddev per group and finally generate a permutation
    such that each high-stddev group is followed by three low-stddev groups.

    Args:
        weight (torch.Tensor): [out_dim, in_dim] tensor, e.g., [2048, 10499]
        group_size (int): number of columns per group
        final_group_size (int): number of columns in a 1-high + 3-low stddev group set

    Returns:
        perm (List[int]): permutation of column indices with length == in_dim
    """
    # group_size = int(final_group_size // (centroid_number / 2))
    group_size = int(final_group_size * ratio)

    out_dim, in_dim = weight.shape
    assert in_dim >= group_size, "Input width must be >= group_size"
    # assert in_dim >= group_size, "Input width must be >= group_size"

    # 1. Column-wise absolute sum -> sort
    abs_sums = torch.sum(weight.abs(), dim=0)
    sorted_indices = torch.argsort(abs_sums, descending=True)
    sorted_weight = weight[:, sorted_indices]
    # abs_means = torch.mean(weight.abs(), dim=0)
    # sorted_indices = torch.argsort(abs_means, descending=True)
    # sorted_weight = weight[:, sorted_indices]

    # 2. Group columns (truncate to fit group_size)
    num_groups = in_dim // group_size
    trimmed_width = num_groups * group_size
    trimmed_weight = sorted_weight[:, :trimmed_width]
    trimmed_indices = sorted_indices[:trimmed_width]
    extra_indices = sorted_indices[trimmed_width:]  # save remaining columns

    # [out_dim, num_groups, group_size] -> [num_groups, out_dim, group_size]
    groups = trimmed_weight.view(out_dim, num_groups, group_size).permute(1, 0, 2)

    # 3. Compute average row-wise stddev for each group
    group_stddevs = []
    for i in range(num_groups):
        stds = torch.std(groups[i], dim=1)  # (out_dim,)
        avg_std = stds.mean().item()
        group_stddevs.append((i, avg_std))

    # 4. Reorder groups using high-low pairing
    group_stddevs.sort(key=lambda x: x[1], reverse=True)  # sort by stddev, descending
    used = set()
    permuted_group_indices = []

    i = 0
    number_per_group = int(final_group_size / group_size)
    # assert final_group_size % group_size == 0
    while len(used) < num_groups and i < len(group_stddevs):
        # Find unused high stddev group
        while i < len(group_stddevs) and group_stddevs[i][0] in used:
            i += 1
        if i >= len(group_stddevs): break
        high_idx = group_stddevs[i][0]
        used.add(high_idx)
        permuted_group_indices.append(high_idx)

        # Now find 3 unused low stddev groups
        j = len(group_stddevs) - 1
        count = 0
        while j >= 0 and count < (number_per_group - 1):
            low_idx = group_stddevs[j][0]
            if low_idx not in used:
                used.add(low_idx)
                permuted_group_indices.append(low_idx)
                count += 1
            j -= 1

    # 5. Flatten group indices to column indices
    final_perm = []
    for group_id in permuted_group_indices:
        col_start = group_id * group_size
        cols = trimmed_indices[col_start: col_start + group_size].tolist()
        final_perm.extend(cols)

    # 6. Append the extra columns that were not included in any group
    final_perm.extend(extra_indices.tolist())

    assert len(final_perm) == in_dim, f"final perm length {len(final_perm)} != {in_dim}"
    return final_perm

def permute_deepseek(model, ratio, final_group_size, config):
    for i, layer in enumerate(tqdm(model.model.layers, desc="Processing layers")):
        if i == 0:
            ## self-attention
            kv_b_proj = layer.self_attn.kv_b_proj.weight.detach().clone()
            kv_a_proj_with_mqa = layer.self_attn.kv_a_proj_with_mqa.weight.detach().clone()
            kv_a_layernorm = layer.self_attn.kv_a_layernorm.weight.detach().clone()

            kv_a_proj_with_mqa_perm, kv_b_proj_perm = kv_proj_deepseek(
                kv_a_proj_with_mqa,  # contains [W_k; W_v] stacked along out_features
                kv_b_proj,  # weight: [d_model, H*Dv]
                kv_a_layernorm,
                config)

            layer.self_attn.kv_b_proj.weight.data = kv_b_proj_perm
            layer.self_attn.kv_a_proj_with_mqa.weight.data = kv_a_proj_with_mqa_perm
            layer.self_attn.kv_a_layernorm.weight.data.fill_(1.0)

        else:
            ## self-attention
            kv_b_proj = layer.self_attn.kv_b_proj.weight.detach().clone()
            kv_a_proj_with_mqa = layer.self_attn.kv_a_proj_with_mqa.weight.detach().clone()
            kv_a_layernorm = layer.self_attn.kv_a_layernorm.weight.detach().clone()

            kv_a_proj_with_mqa_perm, kv_b_proj_perm = kv_proj_deepseek(
                kv_a_proj_with_mqa,  # contains [W_k; W_v] stacked along out_features
                kv_b_proj,  # weight: [d_model, H*Dv]
                kv_a_layernorm,
                config)

            layer.self_attn.kv_b_proj.weight.data = kv_b_proj_perm
            layer.self_attn.kv_a_proj_with_mqa.weight.data = kv_a_proj_with_mqa_perm
            layer.self_attn.kv_a_layernorm.weight.data.fill_(1.0)

            # mlp block
            for i, expert in enumerate(layer.mlp.experts):
                down_weight = expert.down_proj.weight  # [out_dim, in_dim]
                weight = down_weight.detach().clone()
                perm = group_stddev_metric_plain(
                    weight=weight,
                    ratio=ratio,
                    final_group_size=final_group_size
                )

                gate_weight = expert.gate_proj.weight.detach().clone()
                up_weight = expert.up_proj.weight.detach().clone()
                down_weight = expert.down_proj.weight.detach().clone()

                # apply permutations
                expert.gate_proj.weight.data = gate_weight[perm, :]
                expert.up_proj.weight.data = up_weight[perm, :]
                expert.down_proj.weight.data = down_weight[:, perm]

            down_weight = layer.mlp.shared_experts.down_proj.weight  # [out_dim, in_dim]
            weight = down_weight.detach().clone()
            perm = group_stddev_metric_plain(
                weight=weight,
                ratio=ratio,
                final_group_size=final_group_size
            )

            # clone before assignment (optional, for safety)
            gate_weight = layer.mlp.shared_experts.gate_proj.weight.detach().clone()
            up_weight = layer.mlp.shared_experts.up_proj.weight.detach().clone()
            down_weight = layer.mlp.shared_experts.down_proj.weight.detach().clone()

            # apply permutations
            layer.mlp.shared_experts.gate_proj.weight.data = gate_weight[perm, :]
            layer.mlp.shared_experts.up_proj.weight.data = up_weight[perm, :]
            layer.mlp.shared_experts.down_proj.weight.data = down_weight[:, perm]

def permute_qwen(model, ratio, final_group_size, config):
    with torch.no_grad():
        for i, layer in enumerate(tqdm(model.model.layers, desc="Processing layers")):
            v_weight = layer.self_attn.v_proj.weight.detach().clone()
            o_weight = layer.self_attn.o_proj.weight.detach().clone()

            v_proj_perm, o_proj_perm = apply_equivariant_from_o_proj(
                v_weight,
                o_weight,
                config
            )

            layer.self_attn.v_proj.weight.data = v_proj_perm
            layer.self_attn.o_proj.weight.data = o_proj_perm

            ## for mlp block
            for expert in layer.block_sparse_moe.experts:
                down_weight = expert.w2.weight  # [out_dim, in_dim]
                weight = down_weight.detach().clone()
                expert_perm = group_stddev_metric_plain(weight, ratio, final_group_size)
                expert.w1.weight.data = expert.w1.weight.data[expert_perm, :]
                expert.w3.weight.data = expert.w3.weight.data[expert_perm, :]
                expert.w2.weight.data = expert.w2.weight.data[:, expert_perm]

@torch.no_grad()
def permutation(model, model_type, model_config, groupsize):
    print("[INFO] permutation start.")
    if model_type == "deepseek":
        permute_deepseek(model, ratio=0.2, final_group_size=groupsize, config=model_config)
    elif model_type == "qwen":
        permute_qwen(model, ratio=0.2, final_group_size=groupsize, config=model_config)
