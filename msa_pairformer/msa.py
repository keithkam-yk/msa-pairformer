import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from scipy.spatial.distance import cdist
from torch.nn.functional import one_hot

from .tokens import aa2tok_d, nTokenTypes


class MSA:
    def __init__(
        self,
        msa_file_path: str | Path, # Path to MSA file
        max_length: int = 1024, # Maximum length of the MSA (default is 1024)
        max_tokens: int = 1048576, # Maximum number of tokens in the MSA (default is 1024 * 1024)
        max_seqs: int = 1024, # Maximum number of sequences in the MSA (default is 1024)
        diverse_select_method = "hhfilter", # Method to select diverse sequences (default is "hhfilter")
        secondary_filter_method: str = "greedy",  # Options: "greedy" or "random" (default is "greedy")
        random_query: bool = False, # Whether to select a random query sequence (default is to use the first sequence as the query)
        min_query_coverage: float = 0.8, # Minimum query coverage if selecting a random query sequence
        parser_kwargs: dict = {
            "keep_insertions": False,
            "to_upper": False,
            "remove_lowercase_cols": False
        },
        hhfilter_kwargs: dict = {}
    ):
        self.msa_file_path = msa_file_path
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.diverse_select_method = diverse_select_method
        self.secondary_filter_method = secondary_filter_method
        self.random_query = random_query

        assert self.diverse_select_method in ["greedy", "hhfilter", "none"], "Diverse select method must be either 'greedy' or 'hhfilter', or 'none' if no diverse selection is desired"
        assert self.secondary_filter_method in ["greedy", "random", "none"], "Secondary filter method must be either 'greedy' or 'random', or 'none' if no secondary filter is desired"

        # Parse MSA file
        self.seq_l, self.ids_l = self.parse_a3m_file(**parser_kwargs)
        self.seq_a = self.seq_list_to_arr(self.seq_l)
        if self.random_query:
            # Select random sequence index and swap with query sequence
            # Ensure that randomly selected sequence has at least min_query_coverage (e.g. 80%) coverage
            valid_indices = np.where((self.seq_a != '-').sum(axis=-1) >= (self.seq_a.shape[-1] * min_query_coverage))[0]
            if len(valid_indices) > 0:
                random_idx = np.random.choice(valid_indices)
                self.seq_a[[0, random_idx]] = self.seq_a[[random_idx, 0]]
                # Swap ids
                tmp_id = self.ids_l[0]
                self.ids_l[0] = self.ids_l[random_idx]
                self.ids_l[random_idx] = tmp_id

        # Get random crop based on max length
        self.random_crop, self.random_crop_min, self.random_crop_max = self.get_random_crop(self.seq_a, max_length=self.max_length)
        # Select sequences based on max depth (greedy selection or hhfilter)
        # Determine maximum depth based on sequence length and max tokens
        # Currently capping the MSA to some number (max_seqs) of sequences
        # Could instead use min(full_msa_depth, max_tokens // sequence_length)
        sequence_length = self.random_crop.shape[1]
        self.subset_depth = min(self.max_tokens // sequence_length, self.max_seqs)
        if self.diverse_select_method == "none":
            self.select_diverse_msa = self.random_crop
            self.select_diverse_indices = np.arange(self.random_crop.shape[0])
        else:
            self.select_diverse_msa, self.select_diverse_indices = self.select_diverse(
                msa_a=self.random_crop,
                num_seqs=self.max_seqs,
                method=diverse_select_method,
                hhfilter_kwargs=hhfilter_kwargs,
                secondary_filter_method=self.secondary_filter_method
            )
        # Tokenize diverse MSA
        self.diverse_tokenized_msa = self.tokenize_msa(self.select_diverse_msa)
        self.n_diverse_seqs = self.diverse_tokenized_msa.shape[0]

    def compute_pssm(self, msa_a):
        # Get counts excluding padding token (26) and create probability distribution
        nSeqs, nPos = msa_a.shape
        probs = torch.zeros(nPos, aa2tok_d['<pad>'])  # Initialize output tensor [nPos, 26]

        # For each position in sequence
        for i in range(nPos):
            pos_tensor = msa_a[:, i]  # Get all sequences at this position
            mask = pos_tensor != aa2tok_d['<pad>']  # Mask padding tokens
            counts = torch.bincount(pos_tensor[mask], minlength=aa2tok_d['<pad>'])  # Count tokens at this position
            probs[i] = counts.float() / counts.sum()  # Convert to probabilities
        return probs

    def generate_random_sequences(self, probs, nSeqs):
        # Sample k sequences from probability distribution
        sampled_seqs = torch.zeros(nSeqs, probs.shape[0], dtype=torch.long)
        for i in range(probs.shape[0]):
            sampled_seqs[:, i] = torch.multinomial(probs[i], nSeqs, replacement=True)
        return sampled_seqs

    def parse_a3m_file(
        self,
        keep_insertions: bool = False,
        to_upper: bool = False,
        remove_lowercase_cols: bool = False, # Any lowercase columns in the query sequence
        **kwargs
    ):
        """
        Parse sequences from a3m file. 
        Returns list of full length sequences aligned to query sequence (top row)
        keep_insertions determines whether to keep insertions in sequences
        to_upper determines whether to convert sequences to uppercase (unnecessary if removing insertions)
        """
        seq_l = []
        ids_l = []
        valid_indices = None
        with open(self.msa_file_path) as oFile:
            for record in SeqIO.parse(oFile, "fasta"):
                sequence = str(record.seq)
                if remove_lowercase_cols:
                    if valid_indices is None:
                        valid_indices = [i for i, aa in enumerate(sequence) if aa.isupper()]
                    sequence = "".join([sequence[i] for i in valid_indices])
                if not keep_insertions:
                    sequence = re.sub(r"[a-z]|\.|\*", "", sequence)
                if to_upper:
                    sequence = sequence.upper()
                seq_l.append(sequence)
                ids_l.append(record.name)
        return seq_l, ids_l

    @property
    def inverse_covariance(self):
        if not hasattr(self, "_inverse_covariance"):
            # One hot encode
            msa_onehot = one_hot(self.diverse_tokenized_msa, num_classes=nTokenTypes)
            # Get shape
            n, l, a = msa_onehot.shape
            flat_msa = msa_onehot.reshape(n, -1)
            # Compute covariance
            c = torch.cov(flat_msa.T)
            # Inverse covariance
            shrink = 4.5 / np.sqrt(n) * torch.eye(c.shape[0])
            ic = torch.linalg.inv(c + shrink)
            # Sum across amino acid inverse covariances (1-21 in our case)
            ic = ic.reshape(l, a, l, a)[:, 1:21, :, 1:21].sum((1, 3))
            self._inverse_covariance = ic
        return self._inverse_covariance

    def tokenize_msa(self, msa_a):
        return torch.from_numpy(np.vectorize(aa2tok_d.get)(msa_a))

    def seq_list_to_arr(self, seq_l):
        return np.array([list(seq) for seq in seq_l])

    def get_random_crop(self, msa_a, max_length: int = 1024):
        seq_len = msa_a.shape[1]
        if seq_len <= max_length:
            return msa_a, 0, seq_len-1
        start = np.random.randint(0, seq_len - max_length)
        return self.seq_a[:, start:start+max_length], start, start+max_length-1

    def select_diverse(self, msa_a, num_seqs: int, method: str = "hhfilter", hhfilter_kwargs: dict = {}, secondary_filter_method: str = "greedy"):
        assert method in ['greedy', 'hhfilter'], "Method must be either 'greedy' or 'hhfilter'"
        if method == 'greedy':
            return self.greedy_select(msa_a, num_seqs)
        elif method == 'hhfilter':
            hhfilter_kwargs = {**hhfilter_kwargs, "diff": num_seqs}
            filtered_msa, kept_indices = self.hhfilter_select(msa_a, **hhfilter_kwargs)
            # If hhfilter returns more sequences than maximum depth,
            # maximize diversity with maximum depth
            if num_seqs < filtered_msa.shape[0]:
                if secondary_filter_method == "greedy":
                    # Greedily select for maximum MSA diversity
                    filtered_msa, greedy_kept_indices = self.greedy_select(filtered_msa, num_seqs)
                    kept_indices = [kept_indices[i] for i in greedy_kept_indices]
                elif secondary_filter_method == "random":
                    # Randomly select num_seqs sequences (always include query sequence)
                    random_indices = np.concatenate([[0], np.random.choice(np.arange(1, filtered_msa.shape[0]), num_seqs-1, replace=False)])
                    filtered_msa = filtered_msa[random_indices]
                    kept_indices = [kept_indices[i] for i in random_indices]
                elif secondary_filter_method == "none":
                    pass
                else:
                    raise ValueError(f"Secondary filter method must be either 'greedy' or 'random', got {secondary_filter_method}")
            return filtered_msa, kept_indices
        else:
            raise ValueError("Method must be either 'greedy' or 'hhfilter'")

    def greedy_select(self, msa_a, num_seqs: int):
        tokenized_msa = self.tokenize_msa(msa_a)
        # Already below depth threshold
        curr_depth = msa_a.shape[0]
        if curr_depth <= num_seqs:
            return msa_a, np.arange(curr_depth)
        # Greedily maximize diversity
        all_indices = np.arange(curr_depth)
        indices = [0]
        pairwise_distances = np.zeros((0, curr_depth))
        for _ in range(num_seqs - 1):
            dist = cdist(tokenized_msa[indices[-1:]], tokenized_msa, "hamming")
            pairwise_distances = np.concatenate([pairwise_distances, dist])
            shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
            shifted_index = np.argmax(shifted_distance)
            index = np.delete(all_indices, indices)[shifted_index]
            indices.append(index)
        indices = sorted(indices)
        return msa_a[indices], indices

    def hhfilter_select(
        self,
        msa_a,
        M = "a3m",
        seq_id: int=90,
        diff: int=0, # Number of sequences
        cov: int=70,
        qid: int=30,
        qsc: float=-20.0,
        maxseq: int=None,
        binary="hhfilter",
    ):
        # Get tmp directory from environment
        tmpdir_base = os.environ.get("TMPDIR", "/tmp/")
        with tempfile.TemporaryDirectory(dir=tmpdir_base) as tmpdirname:
            tmpdir = Path(tmpdirname)
            random_prefix = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), size=8))
            fasta_file = tmpdir / f"{random_prefix}.input.fasta"
            fasta_file.write_text(
                "\n".join([f">{i}\n{''.join(seq)}" for i, seq in enumerate(msa_a)])
            )
            output_file = tmpdir / f"{random_prefix}.output.fasta"
            command = " ".join(
                [
                    f"{binary}",
                    f"-i {fasta_file}",
                    f"-M {M}",
                    f"-o {output_file}",
                    f"-id {seq_id}",
                    f"-diff {diff}",
                    f"-cov {cov}",
                    f"-qid {qid}",
                    f"-qsc {qsc}",
                ]
            ).split(" ")
            if maxseq is not None:
                command.append(f"-maxseq {maxseq}")
            result = subprocess.run(command, capture_output=True)
            result.check_returncode()
            with output_file.open() as f:
                indices = [int(line[1:].strip()) for line in f if line.startswith(">")]
            return msa_a[indices], indices
