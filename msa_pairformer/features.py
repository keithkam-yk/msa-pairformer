import einx
import numpy as np
import torch

from .tokens import aa2tok_d, esmtok_to_pairformertok_d


def convert_tokens_esm2pairformer(batch_tokens: torch.Tensor, device: torch.device = torch.device('cpu')):
    # MSA Transformer uses a CLS token, but MSA Pairformer does not
    # We'll remove the first token to account for this disparity
    return torch.from_numpy(np.vectorize(esmtok_to_pairformertok_d.get)(batch_tokens.cpu().numpy()))[:, :, 1:].to(device)

def msa_mlm(
    msa_t: torch.tensor, 
    mask_tok: int = aa2tok_d['<mask>'],
    pad_tok: int = aa2tok_d['<pad>'],
    mask_prob: float = 0.15,
    mask_ratio: float = 0.8,
    mutate_ratio: float = 0.1,
    keep_ratio: float = 0.1,
    mutate_pssm: bool = True,
    mutate_tok_low: int = 0,
    mutate_tok_high: int = 19,
    query_only: bool = False
):
    # Create masked input (don't mask padding tokens)
    masked_msas = msa_t.clone()
    flat_msas = masked_msas.view(-1)
    if query_only:
        nMSAs, depth, length = masked_msas.shape
        intervals_t = torch.arange(nMSAs).reshape(nMSAs, 1) * depth * length
        non_pad_indices = (torch.arange(length).unsqueeze(0).repeat(nMSAs, 1) + intervals_t).flatten()
        non_pad_indices = non_pad_indices[flat_msas[non_pad_indices] != pad_tok]
    else:
        non_pad_indices = torch.nonzero(flat_msas != pad_tok, as_tuple=False).view(-1)
    
    # Calculate the number of positions to mask
    num_mask = int(len(non_pad_indices) * mask_prob)

    # Generate mask indices
    mlm_indices = np.random.choice(non_pad_indices.numpy(), num_mask, replace=False)
    mask_ub = int(mask_ratio * num_mask)
    mask_indices = mlm_indices[:mask_ub]
    mutate_ub = mask_ub + int(mutate_ratio * num_mask)
    mutate_indices = mlm_indices[mask_ub : mutate_ub]
    keep_indices = mlm_indices[mutate_ub:]

    # Apply MLM (masking and mutating)
    masked_msas.view(-1)[mask_indices] = torch.tensor(mask_tok)
    if mutate_pssm:
        # Get column indices of mutate_indices
        batch_indices, seq_indices, pos_indices = torch.unravel_index(torch.tensor(mutate_indices), msa_t.shape)
        # Compute PSSM of mutate_indices (exclude padding tokens)
        counts_t = torch.nn.functional.one_hot(msa_t, num_classes=msa_t.shape[-1]).sum(dim=1)[:, :, :26]
        pssm = counts_t / counts_t.sum(dim=-1, keepdim=True)
        probs_t = pssm[batch_indices, pos_indices]
        probs_t = probs_t[mutate_indices]
        new_toks_t = torch.multinomial(probs_t, num_samples=1)
        # new_toks_t = torch.stack([torch.multinomial(probs_t[i], num_samples=1) for i in range(len(mutate_indices))]).squeeze()
        # Apply mutations
        masked_msas.view(-1)[mutate_indices] = new_toks_t
    else:
        masked_msas.view(-1)[mutate_indices] = torch.randint_like(input = masked_msas.view(-1)[mutate_indices], low=mutate_tok_low, high=mutate_tok_high+1)

    # Return masked MSA and indices of tokens to predict
    return masked_msas, mlm_indices

def prep_molecule_feats(msa_input):
    batch_size = msa_input.shape[0]
    seq_len = msa_input.shape[2]
    molecule_feats = torch.stack([
        torch.ones(size=(batch_size, seq_len)), # molecule_idx
        torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1), # token_index
        torch.zeros(size=(batch_size, seq_len)), # molecule_idx
        torch.zeros(size=(batch_size, seq_len)),# entity_id,
        torch.zeros(size=(batch_size, seq_len)), # sym_id
    ], dim=-1).reshape((batch_size, seq_len, 5))
    return molecule_feats

def get_relative_positions(msa_input):
    batch_size = msa_input.shape[0]
    seq_len = msa_input.shape[2]
    token_indices = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    return token_indices

def onehot_msa(msa_input, device=torch.device('cpu')):
    return torch.nn.functional.one_hot(msa_input, num_classes=len(aa2tok_d)).to(device)

def prepare_msa_masks(msa_input, device=torch.device('cpu')):
    # msa_input is of shape [b, s, n]
    mask = (msa_input != aa2tok_d['<pad>']).any(dim=1) # [b, n]
    msa_mask = (msa_input != aa2tok_d['<pad>']).any(dim=2) # [b, s]
    full_mask = (msa_input != aa2tok_d['<pad>']) # [b, s, n]
    pairwise_mask = einx.logical_and("... i, ... j -> ... i j", mask, mask) # [b, n, n]
    return mask.to(device), msa_mask.to(device), full_mask.to(device), pairwise_mask.to(device)

def prepare_inputs(batch, device):
    msa_repr = batch['msas_onehot'].float().to(device)
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(batch['msas'])
    msa_mask = msa_mask.to(device)
    mask = mask.to(device)
    full_mask = full_mask.to(device)
    pairwise_mask = pairwise_mask.to(device)
    molecule_feats = batch['molecule_feats'].to(device)
    return msa_repr, mask, msa_mask, full_mask, pairwise_mask, molecule_feats

def prepare_inputs_bf16(batch, device):
    msa_repr = batch['msas_onehot'].bfloat16().to(device)
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(batch['msas'])
    msa_mask = msa_mask.to(device)
    mask = mask.to(device)
    full_mask = full_mask.to(device)
    pairwise_mask = pairwise_mask.to(device)
    molecule_feats = batch['molecule_feats'].bfloat16().to(device)
    return msa_repr, mask, msa_mask, full_mask, pairwise_mask, molecule_feats

def create_msa_subset_mask(subset_msa_idx_l, L):
    b = len(subset_msa_idx_l)
    mask = torch.zeros(b, L, L)
    for i in range(b):
        indices = torch.tensor(subset_msa_idx_l[i])
        row_mask = torch.zeros(L, dtype=torch.bool)
        row_mask[indices] = True
        mask[i] = row_mask.unsqueeze(0) & row_mask.unsqueeze(1)
    return mask > 0
