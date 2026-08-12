import numpy as np
import torch
import torch.nn.functional as F

from .structure import get_distance_matrix


####################
# Contact analysis #
####################
def get_contacts(structure_file_path, chain_id, max_dist = 8):
    # Get distance matrix
    dist_a = get_distance_matrix(structure_file_path, chain_id)
    # Threshold distances to determine contacts
    contacts_a = dist_a.copy() <= max_dist
    return contacts_a

def compute_precision(predictions_a, targets_a, min_seq_sep = 6):
    """Top-K precision of a single predicted contact map.

    K is the number of true contacts in the valid region, i.e. precision@K where
    K = number of contacts -- the same definition of K used by
    `calculate_precision_batch` below. Unlike `compute_precisions`, which returns the
    binned {AUC, P@L, P@L2, P@L5} dictionary for a batch, this returns one scalar for
    one contact map, which is what the singular name and the reduced signature (no
    maxsep / src_lengths / override_length) imply.

    Only the strictly upper-triangular region with `j - i >= min_seq_sep` is scored, and
    pairs marked invalid with a negative target are excluded. Returns None when there is
    no true contact in the valid region (there is no precision to report), matching
    `calculate_precision_batch`.
    """
    if isinstance(predictions_a, np.ndarray):
        predictions_a = torch.from_numpy(predictions_a)
    if isinstance(targets_a, np.ndarray):
        targets_a = torch.from_numpy(targets_a)
    assert predictions_a.shape == targets_a.shape, "Predictions and targets must have the same shape"
    # This scores a single contact map: accept [L, L] or a batch of one, [1, L, L]
    if predictions_a.dim() == 3 and predictions_a.shape[0] == 1:
        predictions_a = predictions_a.squeeze(0)
        targets_a = targets_a.squeeze(0)
    if predictions_a.dim() != 2:
        raise ValueError(
            f"compute_precision scores a single contact map, got shape {tuple(predictions_a.shape)}. "
            "Use compute_precisions for a batch."
        )
    targets_a = targets_a.to(predictions_a.device)
    # Get valid indices
    seqlen = predictions_a.shape[1]
    seqlen_range = torch.arange(seqlen, device=predictions_a.device)
    valid_mask = seqlen_range[None, :] - seqlen_range[:, None] >= min_seq_sep
    # Some contact maps have -1 for invalid pairs
    valid_mask = valid_mask & (targets_a >= 0)

    # Number of true contacts in the valid region -- the K of precision@K
    true_contacts = (targets_a > 0) & valid_mask
    num_contacts = int(true_contacts.sum().item())
    if num_contacts == 0:
        return None

    # Invalid pairs are pushed to the bottom of the ranking so they can never be
    # selected among the top K
    if not predictions_a.is_floating_point():
        predictions_a = predictions_a.float()
    scored = predictions_a.masked_fill(~valid_mask, float("-inf"))
    topk_indices = scored.flatten().argsort(descending=True)[:num_contacts]
    topk_hits = true_contacts.flatten()[topk_indices]
    return (topk_hits.sum() / num_contacts).item()

# ######################
# # Contact evaluation #
# ######################
# def calculate_precision_batch(pred_contacts, true_contacts, minsep=6, maxsep=None):
#     if pred_contacts.shape != true_contacts.shape:
#         raise ValueError("Predicted and true contact matrices must have the same shape")
#     if pred_contacts.dim() == 2:
#         pred_contacts = pred_contacts.unsqueeze(0)
#     if true_contacts.dim() == 2:
#         true_contacts = true_contacts.unsqueeze(0)
#     # Get batch size and longest sequence length
#     B, L, _ = pred_contacts.shape
#     # Get device
#     device = pred_contacts.device
#     if true_contacts.device != device:
#         true_contacts = true_contacts.to(device)
#     # Create valid mask (only consider upper triangular matrix)
#     # Padded regions have -1 in true_contacts
#     # Valid mask for both predicted and true contact matrices
#     seqlen_range = torch.arange(L, device=device)
#     sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
#     sep = sep.unsqueeze(0)
#     valid_mask = sep >= minsep
#     if maxsep is not None:
#         valid_mask &= sep < maxsep
#     valid_mask = valid_mask & (true_contacts >= 0)
    
#     # Fill prediction matrix with -inf if it's not valid
#     pred_contacts = pred_contacts.masked_fill(~valid_mask, float('-inf'))

#     # Get upper triangular predictions and true contacts
#     # Predictions have been masked with -inf if they are not valid, so when we sort and take the topk, we are only taking valid predictions
#     x_ind, y_ind = np.triu_indices(L, minsep)
#     predictions_upper = pred_contacts[:, x_ind, y_ind]
#     true_upper = true_contacts[:, x_ind, y_ind]
    
#     # Determine number of true contacts in valid region
#     K = true_contacts.masked_fill(~valid_mask, 0).sum(dim=-1).sum(dim=-1)
    
#     # Get Top-K predictions and pad if not enough predictions
#     max_k = int(K.max().item())
#     if max_k == 0:
#         return None
#     indices = predictions_upper.argsort(dim=-1, descending=True)[:, :max_k]
#     topk_targets = true_upper[torch.arange(B).unsqueeze(1), indices]
#     if topk_targets.size(1) < max_k:
#         topk_targets = torch.nn.functional.pad(topk_targets, [0, max_k - topk_targets.size(1)])
#     # Get cumulative sum of true positives
#     cumulative_dist = topk_targets.type_as(pred_contacts).cumsum(dim=-1)
#     # Compute precision based on true number of contacts
#     gather_lengths = K.unsqueeze(1)
#     gather_indices = (
#         torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths
#     ).type(torch.long) - 1
#     gather_indices = gather_indices.clamp_min(0)
#     # Bin cumulative sum of true positives with intervals of sequence length
#     binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
#     binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(binned_cumulative_dist)
#     # Get precisions 
#     pl5 = binned_precisions[:, 1]
#     pl2 = binned_precisions[:, 4]
#     pl = binned_precisions[:, 9]
#     auc = binned_precisions.mean(dim=-1)

#     return {"AUC": auc, "P@L5": pl5, "P@L2": pl2, "P@L": pl}

# def evaluate_contact_prediction(
#     predictions: torch.Tensor,
#     targets: torch.Tensor,
# ) -> Dict[str, float]:
#     if isinstance(targets, np.ndarray):
#         targets = torch.from_numpy(targets)
#     if isinstance(predictions, np.ndarray):
#         predictions = torch.from_numpy(predictions)
#     contact_ranges = [
#         ("local", 3, 6),
#         ("short", 6, 12),
#         ("medium", 12, 24),
#         ("long", 24, None),
#     ]
#     metrics = {}
#     targets = targets.to(predictions.device)
#     for name, minsep, maxsep in contact_ranges:
#         rangemetrics = calculate_precision_batch(
#             predictions,
#             targets,
#             minsep=minsep,
#             maxsep=maxsep,
#         )
#         if rangemetrics is not None:
#             for key, val in rangemetrics.items():
#                 metrics[f"{name}_{key}"] = list(val.float().cpu().numpy())
#     return metrics

def compute_precisions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: torch.Tensor | None = None,
    minsep: int = 6,
    maxsep: int | None = None,
    override_length: int | None = None,  # for casp
):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)
    override_length = (targets[0, 0] >= 0).sum()

    # Check sizes
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, "
            f"targets of size {targets.size()}"
        )
    device = predictions.device

    batch_size, seqlen, _ = predictions.size()
    seqlen_range = torch.arange(seqlen, device=device)

    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    valid_mask = sep >= minsep
    valid_mask = valid_mask & (targets >= 0)  # negative targets are invalid

    if maxsep is not None:
        valid_mask &= sep < maxsep

    if src_lengths is not None:
        valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
        valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
    else:
        src_lengths = torch.full([batch_size], seqlen, device=device, dtype=torch.long)

    predictions = predictions.masked_fill(~valid_mask, float("-inf"))

    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    topk = seqlen if override_length is None else max(seqlen, override_length)
    indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
    topk_targets = targets_upper[torch.arange(batch_size).unsqueeze(1), indices]
    if topk_targets.size(1) < topk:
        topk_targets = F.pad(topk_targets, [0, topk - topk_targets.size(1)])

    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

    gather_lengths = src_lengths.unsqueeze(1)
    if override_length is not None:
        gather_lengths = override_length * torch.ones_like(
            gather_lengths, device=device
        )

    gather_indices = (
        torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths
    ).type(torch.long) - 1

    binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(
        binned_cumulative_dist
    )

    pl5 = binned_precisions[:, 1]
    pl2 = binned_precisions[:, 4]
    pl = binned_precisions[:, 9]
    auc = binned_precisions.mean(-1)

    return {"AUC": auc, "P@L": pl, "P@L2": pl2, "P@L5": pl5}

def evaluate_contact_prediction(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    contact_ranges = [
        ("local", 3, 6),
        ("short", 6, 12),
        ("medium", 12, 24),
        ("long", 24, None),
        ("morethansix", 6, None)
    ]
    metrics = {}
    targets = targets.to(predictions.device)
    for name, minsep, maxsep in contact_ranges:
        rangemetrics = compute_precisions(
            predictions,
            targets,
            minsep=minsep,
            maxsep=maxsep,
        )
        for key, val in rangemetrics.items():
            metrics[f"{name}_{key}"] = val.item()
    return metrics

# Get top L pairs
def get_top_L_pairs(predicted_contacts_a, minsep=24):
    L = predicted_contacts_a.shape[0]
    # Only consider upper triangular part
    triu_idx = np.triu_indices_from(predicted_contacts_a, 1)
    # Mask out pairs < 24 residues apart
    mask = np.abs(triu_idx[0] - triu_idx[1]) >= minsep
    filtered_i = triu_idx[0][mask]
    filtered_j = triu_idx[1][mask]
    # Get values for these filtered indices
    vals = predicted_contacts_a[filtered_i, filtered_j]
    # Get top L
    L = min(L, len(vals))
    cutoff = np.sort(vals)[::-1][L-1]
    vals_sort_idx = vals >= cutoff
    return filtered_i[vals_sort_idx], filtered_j[vals_sort_idx]

def get_p_at_k(gt_contacts_a, pred_contacts_a, minsep=24, upper_triangle=True):
    assert gt_contacts_a.shape == pred_contacts_a.shape, f"Size mismatch. Received predictions of size {pred_contacts_a.shape}, targets of size {gt_contacts_a.shape}"
    # Subset for upper triangle if necessary (for monomer but not for hetero-oligomer)
    if upper_triangle:
        triu_idx = np.triu_indices_from(gt_contacts_a, 1)
        mask = np.abs(triu_idx[0] - triu_idx[1]) >= minsep
        filtered_i, filtered_j = triu_idx[0][mask], triu_idx[1][mask]
    else:
        filtered_i, filtered_j = np.indices(gt_contacts_a.shape)
        filtered_i, filtered_j = filtered_i.flatten(), filtered_j.flatten()
    gt_contact_labels = gt_contacts_a[filtered_i, filtered_j]
    pred_contact_vals = pred_contacts_a[filtered_i, filtered_j]
    
    # Get total number of ground truth contacts
    k = gt_contact_labels.sum()

    # Sort predictions by descending order and take the top k
    top_k_indices = np.argpartition(pred_contact_vals, -k)[-k:]
    
    p_at_k = (gt_contact_labels[top_k_indices] == 1).sum() / k
    return p_at_k

def get_p_at_l(gt_contacts_a, pred_contacts_a, minsep=24, upper_triangle=True):
    assert gt_contacts_a.shape == pred_contacts_a.shape, f"Size mismatch. Received predictions of size {pred_contacts_a.shape}, targets of size {gt_contacts_a.shape}"
    # Subset for upper triangle if necessary (for monomer but not for hetero-oligomer)
    if upper_triangle:
        triu_idx = np.triu_indices_from(gt_contacts_a, 1)
        mask = np.abs(triu_idx[0] - triu_idx[1]) >= minsep
        filtered_i, filtered_j = triu_idx[0][mask], triu_idx[1][mask]
    else:
        filtered_i, filtered_j = np.indices(gt_contacts_a.shape)
        filtered_i, filtered_j = filtered_i.flatten(), filtered_j.flatten()
    gt_contact_labels = gt_contacts_a[filtered_i, filtered_j]
    pred_contact_vals = pred_contacts_a[filtered_i, filtered_j]
    
    # Get total number of ground truth contacts
    k = gt_contacts_a.shape[0]

    # Sort predictions by descending order and take the top k
    top_k_indices = np.argpartition(pred_contact_vals, -k)[-k:]
    
    p_at_l = (gt_contact_labels[top_k_indices] == 1).sum() / k
    return p_at_l
