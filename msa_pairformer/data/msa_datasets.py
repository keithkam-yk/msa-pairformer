import os
from copy import deepcopy
from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

from ..features import msa_mlm, prep_molecule_feats, prepare_msa_masks
from ..msa import MSA
from ..tokens import aa2tok_d, nTokenTypes


class MSADataset(Dataset):
    def __init__(
        self,
        msa_dir=None,
        msa_paths = None,
        max_seq_length = 1024,
        max_msa_depth = 1024,
        max_tokens = 2**17,
        min_depth = 4,
        transform = None,
        random_query: bool = False,
        min_query_coverage: float = 0.8,
        hhfilter_kwargs: dict = {},
        secondary_filter_method: str = "greedy"
    ):
        """
        data_dir is the parent directory that stores all of the MSA .a3m files
        """
        assert (msa_dir is not None) or (msa_paths is not None), "Must provide MSA paths or directory path"
        super().__init__()
        self.msa_dir = msa_dir
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.max_msa_depth = max_msa_depth
        self.max_tokens = max_tokens
        self.min_depth = min_depth
        self.random_query = random_query
        self.min_query_coverage = min_query_coverage
        self.hhfilter_kwargs = hhfilter_kwargs
        self.secondary_filter_method = secondary_filter_method
        if msa_paths is None:
            self.msa_paths = glob(os.path.join(msa_dir, "*/a3m/*.a3m"))
        else:
            self.msa_paths = msa_paths
        
    def __len__(self):
        return len(self.msa_paths)

    def __getitem__(self, idx):
        res_d = {}
        # Get MSA file path
        msa_path = self.msa_paths[idx]
        # Create MSA object
        msa = MSA(
            msa_path,
            max_length = self.max_seq_length,
            max_seqs = self.max_msa_depth,
            max_tokens = self.max_tokens,
            diverse_select_method = "hhfilter",
            random_query = self.random_query,
            min_query_coverage = self.min_query_coverage,
            hhfilter_kwargs = self.hhfilter_kwargs,
            secondary_filter_method = self.secondary_filter_method
        )
        # Get tokenized MSA
        res_d['tokenized_msa'] = msa.diverse_tokenized_msa
        # Add MSA depth
        res_d['n_diverse_seqs'] = msa.n_diverse_seqs
        # Add file path
        res_d['file_path'] = msa_path
        # Add sequence indices
        res_d['seq_indices'] = msa.select_diverse_indices
        return res_d     

class CollateAFBatch:
    def __init__(
        self,
        max_seq_length,
        max_seq_depth,
        min_seq_depth,
        pad_tok=aa2tok_d['<pad>'],
        mask_tok=aa2tok_d['<mask>'], 
        mask_prob=0.15,
        mask_ratio=0.8,
        mutate_ratio=0.1,
        keep_ratio=0.1,
        tok_low=0,
        tok_high=25,
        query_only=False,
        mutate_pssm=False,
    ):
        self.max_seq_depth = max_seq_depth
        self.max_seq_length = max_seq_length
        self.min_seq_depth = min_seq_depth
        self.pad_tok = pad_tok
        self.mask_tok = mask_tok
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
        self.mutate_ratio = mutate_ratio
        self.keep_ratio = keep_ratio
        self.tok_low = tok_low
        self.tok_high = tok_high
        self.mutate_pssm = mutate_pssm
        self.query_only = query_only

    def __call__(self, batch):
        # Initialize output dictionary
        output_dict = {}

        # Skip MSAs with too few sequences
        valid_msa = [b['n_diverse_seqs'] >= self.min_seq_depth for b in batch]
        if not any(valid_msa):
            return None
        # Get batch size
        og_batch_size = len(batch)
        effective_batch_size = sum(valid_msa)
        
        # Match sequence lengths and stack tensors
        tokenized_msa_l = [batch[i]['tokenized_msa'] for i in range(og_batch_size) if valid_msa[i]]
        # shape = [batch_size] + np.max([seq.shape for seq in tokenized_msa_l], 0).tolist()
        shape = [effective_batch_size, self.max_seq_depth, self.max_seq_length]
        dtype = tokenized_msa_l[0].dtype
        if isinstance(tokenized_msa_l[0], np.ndarray):
            msas = np.full(shape, self.pad_tok, dtype=dtype)
        elif isinstance(tokenized_msa_l[0], torch.Tensor):
            msas = torch.full(shape, self.pad_tok, dtype=dtype)
        for msa, seq in zip(msas, tokenized_msa_l, strict=False):
            msaslice = tuple(slice(dim) for dim in seq.shape)
            msa[msaslice] = seq
        output_dict['msas'] = msas
    
        # Create masked input
        if self.mask_prob > 0:
            masked_msas, mlm_indices = msa_mlm(
                msas, mask_tok=self.mask_tok, pad_tok=self.pad_tok, mask_prob=self.mask_prob, mask_ratio=self.mask_ratio, 
                mutate_ratio=self.mutate_ratio, keep_ratio=self.keep_ratio, mutate_tok_low=self.tok_low, 
                mutate_tok_high=self.tok_high, query_only=self.query_only, mutate_pssm=self.mutate_pssm
            )
        else:
            masked_msas = deepcopy(msas)
            mlm_indices = None
        # One-hot encode masked/mutated MSA
        masked_msas_onehot = one_hot(masked_msas, num_classes = nTokenTypes)
        output_dict['msas_onehot'] = masked_msas_onehot
        if self.query_only:
            if mlm_indices is None:
                output_dict['masked_idx'] = None
            else:
                _, _, n_pos = msas.shape
                batch_idx, seq_idx, pos_idx = np.unravel_index(mlm_indices, msas.shape)
                query_only_pos_idx = pos_idx + batch_idx * n_pos
                output_dict['masked_idx'] = query_only_pos_idx
        else:
            output_dict['masked_idx'] = mlm_indices
        output_dict['unmasked_msas_onehot'] = one_hot(msas, num_classes = nTokenTypes)
    
        # Initialize additional molecule features
        molecule_feats = prep_molecule_feats(output_dict['msas_onehot'])
        output_dict['molecule_feats'] = molecule_feats
        # Store file path
        output_dict['file_path'] = [batch[i]['file_path'] for i in range(og_batch_size) if valid_msa[i]]
        # Store sequence indices
        output_dict['seq_indices'] = [batch[i]['seq_indices'] for i in range(og_batch_size) if valid_msa[i]]
        # Store MSA depth
        output_dict['msa_depths'] = torch.tensor([batch[i]['n_diverse_seqs'] for i in range(og_batch_size) if valid_msa[i]])
        # Prepare masks
        mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(output_dict['msas'])
        output_dict['mask'] = mask
        output_dict['msa_mask'] = msa_mask
        output_dict['full_mask'] = full_mask
        output_dict['pairwise_mask'] = pairwise_mask
        return output_dict

class trRosettaContactMSADataset(Dataset):
    def __init__(
        self,
        paired_paths_l: list[tuple[str | Path, str | Path]],
        max_seq_length: int = 1024,
        max_msa_depth: int = 1024,
        min_msa_depth: int = 8,
        max_tokens: int = 2**17,
        random_query: bool = False,
        min_query_coverage: float = 0.9,
        hhfilter_kwargs: dict = {},
        secondary_filter_method: str = "greedy"
    ):
        self.paired_paths_l = paired_paths_l
        self.max_seq_length = max_seq_length
        self.max_msa_depth = max_msa_depth
        self.max_tokens = max_tokens
        self.random_query = random_query
        self.min_query_coverage = min_query_coverage
        self.hhfilter_kwargs = hhfilter_kwargs
        self.secondary_filter_method = secondary_filter_method

    def __len__(self):
        return len(self.paired_paths_l)

    def __getitem__(self, idx):
        # Get MSA file path
        msa_path, npz_file_path = self.paired_paths_l[idx]
        # Create MSA mobject
        msa = MSA(
            msa_path,
            max_length = self.max_seq_length,
            max_seqs = self.max_msa_depth,
            max_tokens = self.max_tokens,
            diverse_select_method = "hhfilter",
            random_query = self.random_query,
            min_query_coverage = self.min_query_coverage
        )
        res_d = {}
        # Get tokenized MSA
        res_d['tokenized_msa'] = msa.diverse_tokenized_msa
        res_d['n_diverse_seqs'] = msa.n_diverse_seqs
        # Get contact map
        npz_obj = np.load(npz_file_path)
        res_d['contact_map'] = torch.tensor((npz_obj['dist6d'] > 0) & (npz_obj['dist6d'] < 8))
        # Add file path
        res_d['msa_file_path'] = msa_path
        res_d['npz_file_path'] = npz_file_path
        # Get random crop bounds
        res_d['msa_crop_bounds'] = (msa.random_crop_min, msa.random_crop_max)
        return res_d

class CollatetrRosettaContactMSABatch:
    def __init__(
        self, 
        max_seq_length: int, 
        max_seq_depth: int,
        min_seq_depth: int,
        pad_tok: int = aa2tok_d['<pad>']
    ):
        self.max_seq_depth = max_seq_depth
        self.max_seq_length = max_seq_length
        self.min_seq_depth = min_seq_depth
        self.pad_tok = pad_tok

    def __call__(self, batch):
        # Initialize output dictionary
        output_dict = {}

        # Skip MSAs with too few sequences
        valid_msa = [b['n_diverse_seqs'] >= self.min_seq_depth for b in batch]
        if not any(valid_msa):
            return None
    
        # Get batch size
        og_batch_size = len(batch)
        effective_batch_size = sum(valid_msa)
        
        # Match sequence lengths and stack tensors
        tokenized_msa_l = [batch[i]['tokenized_msa'] for i in range(og_batch_size) if valid_msa[i]]
        shape = [effective_batch_size, self.max_seq_depth, self.max_seq_length]
        dtype = tokenized_msa_l[0].dtype
        if isinstance(tokenized_msa_l[0], np.ndarray):
            msas = np.full(shape, self.pad_tok, dtype=dtype)
        elif isinstance(tokenized_msa_l[0], torch.Tensor):
            msas = torch.full(shape, self.pad_tok, dtype=dtype)
        for msa, seq in zip(msas, tokenized_msa_l, strict=False):
            msaslice = tuple(slice(dim) for dim in seq.shape)
            msa[msaslice] = seq
        output_dict['msas'] = msas
        # Match lengths of contact maps
        contact_map_l = [batch[i]['contact_map'].float() for i in range(og_batch_size) if valid_msa[i]]
        msa_crop_bounds_l = [batch[i]['msa_crop_bounds'] for i in range(og_batch_size) if valid_msa[i]]
        contact_map_l = [contact_map_l[i][msa_crop_bounds_l[i][0]:msa_crop_bounds_l[i][1], msa_crop_bounds_l[i][0]:msa_crop_bounds_l[i][1]] for i in range(len(contact_map_l))]
        shape = [effective_batch_size, shape[-1], shape[-1]]
        dtype = contact_map_l[0].dtype
        if isinstance(contact_map_l[0], np.ndarray):
            contacts = np.full(shape, -1, dtype=dtype)
        elif isinstance(contact_map_l[0], torch.Tensor):
            contacts = torch.full(shape, -1, dtype=dtype)
        for i, contact_map in enumerate(contact_map_l):
            contacts[i, :contact_map.shape[0], :contact_map.shape[1]] = contact_map
        
        output_dict['contact_map'] = contacts

        # One-hot encode MSA
        msas_onehot = one_hot(msas, num_classes = nTokenTypes)
        output_dict['msas_onehot'] = msas_onehot
        # Initialize additional molecule features
        molecule_feats = prep_molecule_feats(output_dict['msas_onehot'])
        output_dict['molecule_feats'] = molecule_feats
        # Store file path
        output_dict['msa_file_path'] = [batch[i]['msa_file_path'] for i in range(og_batch_size) if valid_msa[i]]
        output_dict['npz_file_path'] = [batch[i]['npz_file_path'] for i in range(og_batch_size) if valid_msa[i]]
        # Prepare masks
        mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(output_dict['msas'])
        output_dict['mask'] = mask
        output_dict['msa_mask'] = msa_mask
        output_dict['full_mask'] = full_mask
        output_dict['pairwise_mask'] = pairwise_mask
        return output_dict
