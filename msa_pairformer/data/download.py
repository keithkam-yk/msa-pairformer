import logging
import multiprocessing as mp
import shlex
import subprocess

from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_AF_pdb(prot_id, out_dir):
    url = f"https://alphafold.ebi.ac.uk/files/AF-{prot_id}-F1-model_v4.pdb"
    cmd = f"wget -P {out_dir} {url}"
    res = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True)
    logger.debug("%s", res.stdout)

def download_rcsb_pdb(pdb_id, out_dir, log=False, cif=False):
    if not cif:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        cmd = f"wget {url} -O {out_dir}/{pdb_id}.pdb"
    else:
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        cmd = f"wget {url} -O {out_dir}/{pdb_id}.cif"
    res = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True)
    if log:
        logger.debug("%s", res.stdout)

def download_rcsb_pdb_mp(param_d: dict):
    pdb_id = param_d["pdb_id"]
    out_dir = param_d["out_dir"]
    log = param_d["log"]
    cif = param_d["cif"]
    download_rcsb_pdb(pdb_id, out_dir, log, cif)

def download_rcsb_pdbs(pdb_ids_l: list, out_dir: str, log: bool = False, cpu_buffer: int = 4, cif: bool = False):
    with mp.Pool(processes=mp.cpu_count() - cpu_buffer) as pool:
        list(tqdm(pool.imap(download_rcsb_pdb_mp, [{'pdb_id': pdb_id, 'out_dir': out_dir, 'log': log, 'cif': cif} for pdb_id in pdb_ids_l]), total=len(pdb_ids_l), leave=True))
        pool.close()
