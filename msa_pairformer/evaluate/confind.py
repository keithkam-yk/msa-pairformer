"""CONFIND contact-map wrappers: run the binary, then read what it wrote.

Nothing in this repository calls these five functions, and that is not a sign
they are dead. `model.py` exposes `predict_confind_contacts`, so the concept is
live; these are how a user produces the input that head is scored against. They
are public API whose only callers are outside the tree.
"""

import logging
import multiprocessing as mp
import shlex
import subprocess

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_confind(
    structure_file_path, 
    output_contact_file_path, 
    bin_path = "/home/ubuntu/tools/confind/confind",
    rot_lib_path = "/home/ubuntu/tools/confind/rotlibs"
):
    """Run confind."""
    cmd = f"{bin_path} --p {structure_file_path} --o {output_contact_file_path} --rLib {rot_lib_path}"
    res = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True)
    logger.debug("%s", res.stdout)
    return res

def run_confind_mp(param_d):
    """Run confind in parallel."""
    structure_file_path = param_d["structure_file_path"]
    output_contact_file_path = param_d["output_contact_file_path"]
    bin_path = param_d["bin_path"]
    rot_lib_path = param_d["rot_lib_path"]  
    run_confind(structure_file_path, output_contact_file_path, bin_path, rot_lib_path)

def run_batch_confind(
    structure_file_paths_l: list, 
    output_contact_file_paths_l: list, 
    bin_path = "/home/ubuntu/tools/confind/confind", 
    rot_lib_path = "/home/ubuntu/tools/confind/rotlibs",
    nproc = None,
    cpu_buffer: int = 4
):
    """Run confind in batch."""
    if nproc is None:
        nproc = mp.cpu_count() - cpu_buffer
    zipped_paths = zip(structure_file_paths_l, output_contact_file_paths_l, strict=False)
    param_d_l = [{'structure_file_path': structure_file_path, 'output_contact_file_path': output_contact_file_path, 'bin_path': bin_path, 'rot_lib_path': rot_lib_path} for 
                 structure_file_path, output_contact_file_path in zipped_paths]
    with mp.Pool(processes=nproc) as pool:
        list(tqdm(pool.imap(run_confind_mp, param_d_l), total=len(param_d_l), leave=True))
        pool.close()

def extract_confind_contacts(confind_file_path):
    with open(confind_file_path) as oFile:
        lines = oFile.readlines()
        # Get protein length
        max_res_idx = int(lines[-2].split()[1].split(',')[1])
        length = len(lines[-1].split()[1:])
        contact_a = np.zeros((max_res_idx, max_res_idx))
        for line in lines:
            if line.startswith('contact'):
                split_line = line.split()
                res_i = int(split_line[1].split(',')[1]) - 1
                res_j = int(split_line[2].split(',')[1]) - 1
                c = float(split_line[3])
                if c > contact_a[res_i, res_j]:
                    contact_a[res_i, res_j] = c
                    contact_a[res_j, res_i] = c
    return contact_a

def extract_homooligomeric_confind_contacts(confind_file_path):
    with open(confind_file_path) as oFile:
        lines = oFile.readlines()
        # Get protein length
        observed_chains_d = {}
        for line in lines:
            if line.startswith("freedom"):
                chain_id = line.split()[1].split(',')[0]
                res_idx = int(line.split()[1].split(',')[1])
                if chain_id not in observed_chains_d:
                    observed_chains_d[chain_id] = res_idx
                else:
                    if res_idx > observed_chains_d[chain_id]:
                        observed_chains_d[chain_id] = res_idx
        max_res_idx = max(list(observed_chains_d.values()))
        contact_a = np.zeros((max_res_idx, max_res_idx))
        # Get contacts
        for line in lines:
            if line.startswith('contact'):
                split_line = line.split()
                res_i = int(split_line[1].split(',')[1]) - 1
                res_j = int(split_line[2].split(',')[1]) - 1
                c = float(split_line[3])
                if c > contact_a[res_i, res_j]:
                    contact_a[res_i, res_j] = c
                    contact_a[res_j, res_i] = c
    return contact_a
