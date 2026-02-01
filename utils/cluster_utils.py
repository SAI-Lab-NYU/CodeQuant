import torch
import torch.nn.functional as F

from tqdm import tqdm

from utils.tensor_utils import group


def kmeans_clustering(group_tensor: torch.Tensor,
                      mask: torch.Tensor,
                      cluster_num: int,
                      centroids: torch.Tensor = None,
                      assignments: torch.Tensor = None,
                      kmeans_iteration: int = 300) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if centroids is None and assignments is None:
        centroids, assignments = kmeans(group_tensor,
                                        mask=mask,
                                        cluster_num=cluster_num,
                                        kmeans_iteration=kmeans_iteration)

    assignments_long = assignments.long()
    quant_tensor = torch.gather(centroids.unsqueeze(1).expand(-1, assignments_long.shape[1], -1), -1, assignments_long.unsqueeze(-1)).squeeze(-1) # [n, g]

    return quant_tensor, centroids, assignments


def kmeans(group_tensor: torch.Tensor,
           mask: torch.Tensor,
           cluster_num: int,
           kmeans_iteration: int = 300,
           tol: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    G, S = group_tensor.shape
    device = group_tensor.device

    group_ids = torch.arange(G, device=device).unsqueeze(1).expand(-1, S).reshape(-1)

    if cluster_num == 1:
        centroids = group_tensor.mean(dim=1, keepdim=True)
    else:
        centroids = init_centroids_kpp(group_tensor, cluster_num, mask)

        for _ in tqdm(range(kmeans_iteration), leave=False):
            dist = (group_tensor.unsqueeze(2) - centroids.unsqueeze(1)).pow(2)
            dist = dist.masked_fill(~mask.unsqueeze(2), float('inf'))

            assignments = torch.argmin(dist, dim=2)

            flat_vals_all = group_tensor.reshape(-1)
            flat_assign_all = assignments.reshape(-1)
            mask_flat = mask.reshape(-1)

            flat_vals = flat_vals_all * mask_flat.to(flat_vals_all.dtype)
            flat_cnts = mask_flat.to(flat_vals_all.dtype)

            index = group_ids * cluster_num + flat_assign_all

            sum1d = torch.zeros(G * cluster_num, dtype=group_tensor.dtype, device=device)
            sum1d.scatter_add_(0, index, flat_vals)
            sum2d = sum1d.view(G, cluster_num)

            cnt1d = torch.zeros(G * cluster_num, dtype=group_tensor.dtype, device=device)
            cnt1d.scatter_add_(0, index, flat_cnts)
            cnt2d = cnt1d.view(G, cluster_num)
            cnt2d.masked_fill_(cnt2d == 0, 1.0)

            new_centroids = sum2d / cnt2d
            if (centroids - new_centroids).abs().max() < tol:
                centroids = new_centroids
                break
            centroids = new_centroids

    final_dist = (group_tensor.unsqueeze(2) - centroids.unsqueeze(1)).pow(2)
    final_dist = final_dist.masked_fill(~mask.unsqueeze(2), float('inf'))
    final_assign = torch.argmin(final_dist, dim=2).to(torch.uint8)

    return centroids.to(group_tensor.dtype), final_assign


def init_centroids_kpp(group_tensor: torch.Tensor,
                       cluster_num: int,
                       mask: torch.Tensor):
    G, S = group_tensor.shape
    device, dtype = group_tensor.device, group_tensor.dtype
    g = torch.Generator(device=group_tensor.device)
    g.manual_seed(42)
    centroids = torch.empty((G, cluster_num), device=device, dtype=dtype)

    p0 = mask.to(dtype)
    p0 = p0 / (p0.sum(dim=1, keepdim=True) + 1e-8)
    idx0 = torch.multinomial(p0, 1, generator=g).squeeze(1)
    centroids[:, 0] = group_tensor[torch.arange(G, device=device), idx0]

    d = (group_tensor - centroids[:, 0].unsqueeze(1)).pow(2)
    d = d.masked_fill(~mask, 0.0)
    for i in range(1, cluster_num):
        p = d / (d.sum(dim=1, keepdim=True) + 1e-8)
        idx = torch.multinomial(p, 1, generator=g).squeeze(1)
        c_new = group_tensor[torch.arange(G, device=device), idx]
        centroids[:, i] = c_new
        d_new = (group_tensor - c_new.unsqueeze(1)).pow(2)
        d_new = d_new.masked_fill(~mask, float('inf'))
        d = torch.minimum(d, d_new)
    return centroids


def kmeans_recluster(org_group_weight: torch.Tensor,
                     centroids: torch.Tensor,
                     weight_group_size: int,
                     mask: torch.Tensor,
                     inputs: torch.Tensor,
                     target_inputs: torch.Tensor,
                     org_weight_shape: tuple,
                     eps: float = 1e-6):
    device = org_group_weight.device
    dtype  = org_group_weight.dtype
    G, S   = org_group_weight.shape
    O, D   = org_weight_shape

    X  = inputs.reshape(-1, D).float()
    Xh = target_inputs.reshape(-1, D).float()
    A_D = (X * X).mean(dim=0).clamp_min(eps)
    B_D = (X * Xh).mean(dim=0)

    A_map = A_D.unsqueeze(0).expand(O, -1).contiguous()
    B_map = B_D.unsqueeze(0).expand(O, -1).contiguous()

    A_diag, _ = group(A_map, group_size=weight_group_size)
    B_diag, _ = group(B_map, group_size=weight_group_size)
    A_diag = A_diag.to(dtype=dtype, device=device)
    B_diag = B_diag.to(dtype=dtype, device=device)

    Bw = (org_group_weight * B_diag).unsqueeze(-1)
    Ac = (A_diag.unsqueeze(-1) * centroids.unsqueeze(1))
    score = (Bw - Ac).pow(2).masked_fill(~mask.unsqueeze(-1), float('inf'))
    assignments = torch.argmin(score, dim=2).to(torch.long)

    quant = torch.gather(
        centroids.unsqueeze(1).expand(-1, S, -1),
        2, assignments.unsqueeze(-1)
    ).squeeze(-1)

    return quant, centroids, assignments

