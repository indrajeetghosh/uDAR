#Classwise Alignmnet without gaussian kernel based:- Regularization is used because CMMD loss essentially measures the squared norm of the difference between the mean features of corresponding classes from source and target dataset i.e. if the means are very close to each other, the squared norm might become very small/neglible, potentially leading to numerical instability during optimization. This regularization is done direclt on feature space.

#Regularization based CMMD loss:- Kernel (regularization added directly to the kernel matrix, impacting the kernelized feature space - motivation is prevent overfitting and to enhance the stability - singularity or ill-conditioning during during matrix inversion or eigenvalue decomposition.

import torch
import torch.nn.functional as F

def rbf_kernel(X1, X2, gamma=1.0):
    # Compute squared Euclidean distances between each pair of vectors
    sq_dist = torch.cdist(X1, X2, p=2) ** 2
    K = torch.exp(-gamma * sq_dist)
    return K

def cmmd_loss(source_features, source_labels, target_features, target_pseudo_labels,
              num_classes, gamma=1.0, lambda_reg=1e-3):
    device = source_features.device
    dtype  = source_features.dtype
    eps    = 1e-8

    if source_labels.dim() == 1:
        source_probs = F.one_hot(source_labels.long(), num_classes=num_classes).to(dtype=dtype, device=device)
    else:
        source_probs = source_labels.to(device=device, dtype=dtype)

    if target_pseudo_labels.dim() == 1:
        target_probs = F.one_hot(target_pseudo_labels.long(), num_classes=num_classes).to(dtype=dtype, device=device)
    else:
        target_probs = target_pseudo_labels.to(device=device, dtype=dtype)

    K_ss = rbf_kernel(source_features, source_features, gamma)   # [Ns, Ns]
    K_tt = rbf_kernel(target_features, target_features, gamma)   # [Nt, Nt]
    K_st = rbf_kernel(source_features, target_features, gamma)   # [Ns, Nt]
    def _remove_diag(M):
        return M - torch.diag_embed(torch.diagonal(M, dim1=-2, dim2=-1))

    K_ss_no_diag = _remove_diag(K_ss)
    K_tt_no_diag = _remove_diag(K_tt)

    total_loss = torch.tensor(0.0, device=device, dtype=dtype)
    classes_present = 0
    for class_idx in range(num_classes):
        ps = source_probs[:, class_idx:class_idx+1]  # [Ns,1]
        pt = target_probs[:, class_idx:class_idx+1]  # [Nt,1]
        n_s_eff = ps.sum()
        n_t_eff = pt.sum()
        if (n_s_eff < eps) or (n_t_eff < eps):
            continue
        W_ss = ps @ ps.T      # [Ns, Ns]
        W_tt = pt @ pt.T      # [Nt, Nt]
        W_st = ps @ pt.T      # [Ns, Nt]

        W_ss_no_diag = _remove_diag(W_ss)
        W_tt_no_diag = _remove_diag(W_tt)

        num_ss = (K_ss_no_diag * W_ss_no_diag).sum()
        den_ss = W_ss_no_diag.sum().clamp_min(eps)
        mean_K_ss = num_ss / den_ss
        num_tt = (K_tt_no_diag * W_tt_no_diag).sum()
        den_tt = W_tt_no_diag.sum().clamp_min(eps)
        mean_K_tt = num_tt / den_tt
        mean_K_st = (K_st * W_st).sum() / W_st.sum().clamp_min(eps)
        loss_c = mean_K_ss + mean_K_tt - 2.0 * mean_K_st
        total_loss += loss_c
        classes_present += 1
    if classes_present == 0:
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
    cmmd = total_loss / classes_present
    cmmd.requires_grad_()  
    return cmmd
