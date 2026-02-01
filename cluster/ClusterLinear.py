import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional

from utils.cluster_utils import kmeans_clustering, kmeans_recluster
from utils.tensor_utils import group, degroup
from model.modeling_deepseek import MoEGate

class ClusterLinear(nn.Linear):
    def __init__(self, weight: torch.Tensor,
                 bias: torch.Tensor,
                 group_size: int,
                 cluster_num: int,
                 calibration_mode: bool,
                 module_type: str,
                 layer_name: str,
                 pre_compute_cluster: bool = False,
                 pre_cluster_weight: torch.Tensor = None,
                 pre_centroids: torch.Tensor = None,
                 pre_assignments: torch.Tensor = None,
                 lr: float = 1e-3):
        super().__init__(in_features=weight.size(1),
                         out_features=weight.size(0),
                         bias=(bias is not None))
        with torch.no_grad():
            self.weight.copy_(weight)
            self.weight.requires_grad_(False)
            if bias is not None:
                self.bias.copy_(bias)
                self.bias.requires_grad_(False)
        self.device = weight.device
        self.to(weight.device)

        del weight, bias

        self.module_type = module_type
        self.layer_name = layer_name

        self.group_size = group_size
        self.cluster_num = cluster_num
        self.lr = lr

        self.cluster_weight = None
        self.centroids = None
        self.assignments = None
        self.centroids_optimizer = None
        if not pre_compute_cluster:
            tqdm.write("[INFO] cluster weight not precomputed, clustering...")
            self._cluster()
        else:
            tqdm.write("[INFO] cluster weight precomputed, loading...")
            self._cluster_load(pre_cluster_weight, pre_centroids, pre_assignments)
            del pre_cluster_weight, pre_centroids, pre_assignments

        self.calibration_mode = calibration_mode
        self.calibration_output = None
        self.cached_input = None


    def set_calibration_mode(self, mode):
        self.calibration_mode = mode


    def _clear_calibration_cache(self):
        if hasattr(self, 'calibration_output'):
            del self.calibration_output
            self.calibration_output = None


    def _cluster(self):
        group_weight, mask = group(self.weight, group_size=self.group_size)
        cluster_weight, centroids, assignments = kmeans_clustering(group_tensor=group_weight,
                                                                   mask=mask,
                                                                   cluster_num=self.cluster_num,
                                                                   centroids=None,
                                                                   assignments=None)
        cluster_weight = degroup(cluster_weight, self.group_size, self.weight.shape).detach()
        self.cluster_weight = nn.Parameter(cluster_weight, requires_grad=True)
        self.centroids = nn.Parameter(centroids.to(self.weight.device))
        self.assignments = assignments.detach().to(torch.uint8).to('cpu')
        self.centroids_optimizer = torch.optim.SGD([self.centroids], lr=self.lr)

        del group_weight, mask, cluster_weight, centroids, assignments
        torch.cuda.empty_cache()


    def _cluster_load(self, pre_cluster_weight: torch.Tensor,
                      pre_centroids: torch.Tensor,
                      pre_assignments: torch.Tensor):
        self.cluster_weight = nn.Parameter(pre_cluster_weight.to(self.device, dtype=self.weight.dtype), requires_grad=True)
        self.centroids = nn.Parameter(pre_centroids.to(self.device, dtype=self.weight.dtype), requires_grad=True)
        self.assignments = pre_assignments.to(torch.uint8).to('cpu')
        self.centroids_optimizer = torch.optim.SGD([self.centroids], lr=self.lr)


    def forward(self, input_arg):
        if self.calibration_mode:
            if self.module_type == 'sa':
                tqdm.write(f"[DEBUG] {self.layer_name} calibration collecting...")
                self.calibration_output = F.linear(input_arg, self.weight, self.bias).detach().to("cpu")
                return self.calibration_output.to(input_arg.device)
            else:
                return F.linear(input_arg, self.weight, self.bias)
        else:
            self.cached_input = input_arg.detach().to('cpu')
            if self.module_type == 'sa':
                tqdm.write(f"[INFO] {self.layer_name} gradient computing...")
                self._loss(input_arg)
                return F.linear(input_arg, self.cluster_weight, self.bias).detach()
            else:
                return F.linear(input_arg, self.cluster_weight, self.bias)


    def _loss(self, input_arg):
        input_arg = input_arg.detach()
        calibrate_output = self.calibration_output.to(self.weight.device)
        outputs = F.linear(input_arg, self.cluster_weight, self.bias)
        loss = F.mse_loss(outputs, calibrate_output).to(input_arg.dtype)

        if self.cluster_weight.grad is not None:
            self.cluster_weight.grad.zero_()

        loss.backward()

        del loss, outputs, input_arg, calibrate_output
        self._clear_calibration_cache()
        torch.cuda.empty_cache()


    def _fine_tune(self):
        group_gradient, mask = group(self.cluster_weight.grad, group_size=self.group_size)
        G, S = group_gradient.shape

        self.assignments = self.assignments.to(self.cluster_weight.device).to(torch.long)

        flat_grad_all = group_gradient.reshape(-1)
        flat_assign_all = self.assignments.reshape(-1)
        flat_group_all = torch.arange(G, device=self.cluster_weight.device).unsqueeze(1).expand(-1, S).reshape(-1)
        mask_flat = mask.reshape(-1)

        flat_vals = flat_grad_all * mask_flat.to(flat_grad_all.dtype)
        flat_cnts = mask_flat.to(flat_grad_all.dtype)

        index = flat_group_all * self.cluster_num + flat_assign_all

        sum1d = torch.zeros(G * self.cluster_num, dtype=group_gradient.dtype, device=self.cluster_weight.device)
        sum1d.scatter_add_(0, index, flat_vals)
        gradients_sum = sum1d.view(G, self.cluster_num)

        cnt1d = torch.zeros(G * self.cluster_num, dtype=group_gradient.dtype, device=self.cluster_weight.device)
        cnt1d.scatter_add_(0, index, flat_cnts)
        cluster_count = cnt1d.view(G, self.cluster_num)
        cluster_count.masked_fill_(cluster_count == 0, 1.0)

        avg_gradients = gradients_sum / cluster_count

        self.centroids_optimizer.zero_grad()
        grad = avg_gradients.to(self.centroids.dtype).to(self.centroids.device)
        if self.centroids.grad is None:
            self.centroids.grad = grad.clone()
        else:
            self.centroids.grad.add_(grad)

        self.centroids_optimizer.step()
        self.assignments = self.assignments.to('cpu').to(torch.uint8)


    def fine_tune_centroids(self):
        if self.cluster_weight.grad is None:
            return

        tqdm.write(f"[INFO] {self.layer_name} fine tuning...")
        self._fine_tune()

        group_weight, mask = group(self.weight, group_size=self.group_size)
        inputs = self.cached_input.to(self.weight.device)
        updated_quantized_weight, centroids, assignments = kmeans_recluster(org_group_weight=group_weight,
                                                                            centroids=self.centroids,
                                                                            weight_group_size=self.group_size,
                                                                            mask=mask,
                                                                            inputs=inputs,
                                                                            target_inputs=inputs,
                                                                            org_weight_shape=self.weight.shape)
        self.assignments = assignments.to('cpu').to(torch.uint8)
        with torch.no_grad():
            self.cluster_weight.data.copy_(degroup(updated_quantized_weight, self.group_size, self.weight.shape).to(self.cluster_weight.device))


class ClusterLinear_deepseekGate(MoEGate):
    def __init__(
        self,
        weight: torch.Tensor,
        config,
        bias,
        group_size: int,
        cluster_num: int,
        calibration_mode: bool,
        module_type: str,
        layer_name: str,
        pre_compute_cluster: bool = False,
        pre_cluster_weight: Optional[torch.Tensor] = None,
        pre_centroids: Optional[torch.Tensor] = None,
        pre_assignments: Optional[torch.Tensor] = None,
        pre_mask: Optional[torch.Tensor] = None,
        lr: float = 1e-3,
        orig_gate: Optional[MoEGate] = None,
    ):
        super().__init__(config=config)

        if orig_gate is not None:
            for name in [
                "top_k", "capacity_factor", "alpha", "seq_aux",
                "topk_method", "n_group", "topk_group",
                "norm_topk_prob", "routed_scaling_factor",
                "scoring_func", "n_routed_experts"
            ]:
                if hasattr(orig_gate, name):
                    setattr(self, name, getattr(orig_gate, name))

        with torch.no_grad():
            self.weight.copy_(weight)
            self.weight.requires_grad_(False)
            self.bias = None
            if bias is not None:
                self.bias.copy_(bias)
                self.bias.requires_grad_(False)
        self.to(weight.device)

        self.module_type = module_type
        self.layer_name = layer_name
        self.group_size = group_size
        self.cluster_num = cluster_num
        self.lr = lr

        self.cluster_weight = None
        self.mask = None
        self.centroids = None
        self.assignments = None
        self.centroids_optimizer = None

        if not pre_compute_cluster:
            tqdm.write("[INFO] cluster weight not precomputed, clustering...")
            self._cluster()
        else:
            tqdm.write("[INFO] cluster weight precomputed, loading...")
            self._cluster_load(pre_cluster_weight, pre_centroids, pre_assignments, pre_mask)

        self.calibration_mode = calibration_mode
        self.calibration_output = None
        self.calibration_input = None
        self.cached_input = None

        self.close_form_mode = None


    def set_calibration_mode(self, mode):
        self.calibration_mode = mode


    def _cluster(self):
        group_weight, mask = group(self.weight, group_size=self.group_size)
        cluster_weight, centroids, assignments = kmeans_clustering(group_tensor=group_weight,
                                                                   mask=mask,
                                                                   cluster_num=self.cluster_num,
                                                                   centroids=None,
                                                                   assignments=None)
        cluster_weight = degroup(cluster_weight, self.group_size, self.weight.shape)
        self.cluster_weight = nn.Parameter(cluster_weight, requires_grad=True)
        self.mask = mask.detach().to(self.weight.device)
        self.centroids = nn.Parameter(centroids.to(self.weight.device))
        self.assignments = assignments.detach().to('cpu')

        self.centroids_optimizer = torch.optim.SGD([self.centroids], lr=self.lr)


    def _cluster_load(self, pre_cluster_weight: torch.Tensor,
                      pre_centroids: torch.Tensor,
                      pre_assignments: torch.Tensor,
                      pre_mask: torch.Tensor):
        dev = self.weight.device

        self.cluster_weight = nn.Parameter(pre_cluster_weight.to(dev, dtype=self.weight.dtype), requires_grad=True)
        self.centroids = nn.Parameter(pre_centroids.to(dev, dtype=self.weight.dtype), requires_grad=True)
        self.assignments = pre_assignments.to('cpu')
        self.mask = pre_mask.to(dev)

        self.centroids_optimizer = torch.optim.SGD([self.centroids], lr=self.lr)

    def fake_forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states, self.cluster_weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            group_scores = (
                scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss

    def forward(self, input_arg):
        if self.calibration_mode:
            tqdm.write(f"[INFO] {self.layer_name} calibration collecting...")
            topk_idx, topk_weight, aux_loss = super().forward(input_arg)

            self.calibration_input = input_arg.detach().to('cpu')
            self.calibration_output = F.linear(input_arg, self.weight, None).detach()
            return topk_idx, topk_weight, aux_loss
        else:
            if self.close_form_mode:
                tqdm.write(f"[INFO] {self.layer_name} close form clustering...")
                self._cluster_close_form(input_arg)
                return self.fake_forward(input_arg)
            else:
                self.cached_input = input_arg.detach().to('cpu')
                result = self.fake_forward(input_arg)
                return result


    def _loss(self, input_arg):
        input_arg = input_arg.detach()
        calibrate_output = self.calibration_output.to(self.weight.device).detach()

        outputs = F.linear(input_arg, self.cluster_weight, None)

        loss = F.mse_loss(outputs, calibrate_output).to(input_arg.dtype)

        self.cluster_weight.grad = None

        loss.backward()


    def _fine_tune(self):
        group_gradient, mask = group(self.cluster_weight.grad, group_size=self.group_size)

        G, S = group_gradient.shape

        self.assignments = self.assignments.to(self.cluster_weight.device).to(torch.long)

        flat_grad_all = group_gradient.reshape(-1)
        flat_assign_all = self.assignments.reshape(-1)
        flat_group_all = torch.arange(G, device=self.cluster_weight.device).unsqueeze(1).expand(-1, S).reshape(-1)
        mask_flat = mask.reshape(-1)

        flat_vals = flat_grad_all * mask_flat.to(flat_grad_all.dtype)
        flat_cnts = mask_flat.to(flat_grad_all.dtype)

        index = flat_group_all * self.cluster_num + flat_assign_all

        sum1d = torch.zeros(G * self.cluster_num, dtype=group_gradient.dtype, device=self.cluster_weight.device)
        sum1d.scatter_add_(0, index, flat_vals)
        gradients_sum = sum1d.view(G, self.cluster_num)

        cnt1d = torch.zeros(G * self.cluster_num, dtype=group_gradient.dtype, device=self.cluster_weight.device)
        cnt1d.scatter_add_(0, index, flat_cnts)
        cluster_count = cnt1d.view(G, self.cluster_num)
        cluster_count.masked_fill_(cluster_count == 0, 1.0)

        avg_gradients = gradients_sum / cluster_count

        self.centroids_optimizer.zero_grad()
        grad = avg_gradients.to(self.centroids.dtype).to(self.centroids.device)
        if self.centroids.grad is None:
            self.centroids.grad = grad.clone()
        else:
            self.centroids.grad.add_(grad)

        self.centroids_optimizer.step()
        self.assignments = self.assignments.to('cpu').to(torch.uint8)


    def fine_tune_centroids(self):
        if self.cluster_weight.grad is None:
            return

        tqdm.write(f"[INFO] {self.layer_name} fine tuning...")
        self._fine_tune()

        group_weight, _ = group(self.weight, group_size=self.group_size)
        updated_quantized_weight, centroids, assignments = kmeans_recluster(org_group_weight=group_weight,
                                                                            centroids=self.centroids,
                                                                            weight_group_size=self.group_size,
                                                                            mask=self.mask,
                                                                            inputs=self.cached_input.to(self.weight.device),
                                                                            target_inputs=self.cached_input.to(self.weight.device),
                                                                            org_weight_shape=self.weight.shape)
        self.assignments = assignments.to('cpu').to(torch.uint8)
        with torch.no_grad():
            self.cluster_weight.data.copy_(degroup(updated_quantized_weight, self.group_size, self.weight.shape).to(self.cluster_weight.device))
