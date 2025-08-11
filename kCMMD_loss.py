# Class-wise alignment (linear kernel): MMD equals the squared distance between class-conditional means; small values indicate alignment, not instability.

# Kernel-regularized CMMD (biased form): Add λI to K (K ← K + λI) to adjust diagonal mass; relevant only when using the biased estimator.

# Note: With unbiased=True (diagonals removed), λI has no effect; no matrix inversion is performed, so “ill-conditioning” isn’t applicable.

import torch
import torch.nn.functional as F

def rbf_kernel(X1, X2, gamma=1.0):
    sq_dist = torch.cdist(X1, X2, p=2) ** 2
    return torch.exp(-gamma * sq_dist)

def cmmd_loss(source_features, source_labels, target_features, target_pseudo_labels,
              num_classes, gamma=1.0, lambda_reg=1e-3, unbiased=True, eps=1e-8):
    device = source_features.device
    dtype  = source_features.dtype
    if source_labels.dim() == 1:
        source_probs = F.one_hot(source_labels.long(), num_classes=num_classes).to(dtype=dtype, device=device)
    else:
        source_probs = source_labels.to(device=device, dtype=dtype)
    if target_pseudo_labels.dim() == 1:
        target_probs = F.one_hot(target_pseudo_labels.long(), num_classes=num_classes).to(dtype=dtype, device=device)
    else:
        target_probs = target_pseudo_labels.to(device=device, dtype=dtype)
    K_ss = rbf_kernel(source_features, source_features, gamma)
    K_tt = rbf_kernel(target_features, target_features, gamma)
    K_st = rbf_kernel(source_features, target_features, gamma)
    def _remove_diag(M):
        return M - torch.diag_embed(torch.diagonal(M, dim1=-2, dim2=-1))
    if unbiased:
        K_ss_use = _remove_diag(K_ss)
        K_tt_use = _remove_diag(K_tt)
    else:
        Ns, Nt = K_ss.size(0), K_tt.size(0)
        K_ss_use = K_ss + lambda_reg * torch.eye(Ns, device=device, dtype=dtype)
        K_tt_use = K_tt + lambda_reg * torch.eye(Nt, device=device, dtype=dtype)
    total_loss = torch.zeros((), device=device, dtype=dtype)
    classes_present = 0
    for c in range(num_classes):
        ps = source_probs[:, c:c+1]
        pt = target_probs[:, c:c+1]
        n_s_eff = ps.sum()
        n_t_eff = pt.sum()
        if (n_s_eff < eps) or (n_t_eff < eps):
            continue
        W_ss = ps @ ps.T
        W_tt = pt @ pt.T
        W_st = ps @ pt.T
        if unbiased:
            W_ss_use = _remove_diag(W_ss)
            W_tt_use = _remove_diag(W_tt)
        else:
            W_ss_use = W_ss
            W_tt_use = W_tt
        num_ss = (K_ss_use * W_ss_use).sum()
        den_ss = W_ss_use.sum().clamp_min(eps)
        mean_K_ss = num_ss / den_ss
        num_tt = (K_tt_use * W_tt_use).sum()
        den_tt = W_tt_use.sum().clamp_min(eps)
        mean_K_tt = num_tt / den_tt
        mean_K_st = (K_st * W_st).sum() / W_st.sum().clamp_min(eps)
        loss_c = mean_K_ss + mean_K_tt - 2.0 * mean_K_st
        total_loss = total_loss + loss_c
        classes_present += 1
    if classes_present == 0:
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
    return total_loss / classes_present
