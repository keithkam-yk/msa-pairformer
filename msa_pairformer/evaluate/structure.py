import numpy as np
from Bio.PDB import MMCIFIO, PDBIO, MMCIFParser, Select


#############
# PDB utils #
#############
def write_chain_from_pdb(structure_file_path, chain_ids, out_file_path: None, ignore_hetatm = True):
    """Write chains from a PDB file."""
    atom_counter = 1
    atom_label_l = ['ATOM  ', 'HETATM'] if ignore_hetatm else ['ATOM  ']
    if out_file_path is None:
        chain_id_str = '_'.join(chain_ids)
        out_file_path = structure_file_path.replace('.pdb', f'_{chain_id_str}.pdb')
    with open(structure_file_path) as inFile, open(out_file_path, "w") as outFile:
        for line in inFile.readlines():
            if line[:6] in ['HEADER', 'TITLE ']:
                outFile.write(line)
                continue
            line_chain = line[21]
            if line_chain not in chain_ids:
                continue
            if line[:6] in atom_label_l:
                new_line = (line[:6] + str(atom_counter).rjust(5) + line[11:])
                outFile.write(new_line)
                atom_counter += 1
            if line.startswith('TER'):
                new_ter = ("TER   " + str(atom_counter).rjust(5) + line[11:])
                outFile.write(new_ter)
            if line.startswith('END'):
                outFile.write(line)
    return out_file_path

def write_chain_from_cif(structure_file_path, chain_ids, out_file_path):
    # Create MMCIFParser
    parser = MMCIFParser(QUIET=True)
    # Parse structure from CIF file
    structure = parser.get_structure('protein_structure', structure_file_path)
    # Create subclass of Select to filter out chains
    class ChainSelect(Select):
        def __init__(self, chain_ids):
            self.chain_ids = chain_ids
        def accept_chain(self, chain):
            if chain.id in self.chain_ids:
                return True
            else:
                return False
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(out_file_path, ChainSelect(chain_ids))

def convert_cif_to_pdb(input_cif_path, output_pdb_path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('protein_structure', input_cif_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_path)

def get_distance_matrix(structure_file_path, chain_id):
    # Load coordinates
    with open(structure_file_path) as oFile:
        cif_lines_l = [l for l in oFile.readlines() if 
                       (l.startswith('ATOM') and (l.split()[6] == chain_id) and
                        (((l.split()[3] == 'CA') and (l.split()[5]=='GLY')) or 
                         ((l.split()[3]=='CB') and (l.split()[5] != 'GLY'))))]
    coords_a = np.array([[float(xyz) for xyz in l.split()[10:13]] for l in cif_lines_l])
    # Compute distances
    dist_a = np.linalg.norm(coords_a[:, None] - coords_a, axis=-1)
    return dist_a

def get_coords(structure_file_path, chain_id):
    # Load coordinates
    with open(structure_file_path) as oFile:
        cif_lines_l = [l for l in oFile.readlines() if 
                       (l.startswith('ATOM') and (l.split()[4] == chain_id) and
                        (((l.split()[2] == 'CA') and (l.split()[3]=='GLY')) or 
                         ((l.split()[2]=='CB') and (l.split()[3] != 'GLY'))))]
    coords_a = np.array([[float(xyz) for xyz in l.split()[6:9]] for l in cif_lines_l])
    return coords_a

def get_coords_cif(structure_file_path, chain_id):
    """
    Extract CA (GLY) / CB (all others) coordinates for a single chain from an mmCIF file.

    Returns:
        coords_a    : np.ndarray of shape (N, 3) — coordinates of CA/CB atoms
        sequence    : str — full one-letter sequence of the chain (in residue order)
        coord_indices: list[int] — indices into `sequence` for each row of coords_a
    """
    FIELDS = {
        "_atom_site.group_PDB",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",   # residue sequence number
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    }

    THREE_TO_ONE = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "MSE": "M"
    }

    with open(structure_file_path) as f:
        lines = f.readlines()

    # --- Parse _atom_site loop header ---
    col_index = {}
    in_atom_site_loop = False
    col_counter = 0
    atom_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            in_atom_site_loop = False
            col_counter = 0
            col_index = {}
            continue
        if stripped.startswith("_atom_site."):
            in_atom_site_loop = True
            if stripped in FIELDS:
                col_index[stripped] = col_counter
            col_counter += 1
            continue
        if in_atom_site_loop and stripped and not stripped.startswith("_") and not stripped.startswith("#"):
            atom_lines.append(stripped)
        if stripped == "#" and atom_lines:
            break

    missing = FIELDS - set(col_index.keys())
    if missing:
        raise ValueError(f"CIF file missing expected _atom_site fields: {missing}")

    i_group = col_index["_atom_site.group_PDB"]
    i_atom  = col_index["_atom_site.label_atom_id"]
    i_resn  = col_index["_atom_site.label_comp_id"]
    i_chain = col_index["_atom_site.label_asym_id"]
    i_seqid = col_index["_atom_site.label_seq_id"]
    i_x     = col_index["_atom_site.Cartn_x"]
    i_y     = col_index["_atom_site.Cartn_y"]
    i_z     = col_index["_atom_site.Cartn_z"]

    # --- First pass: collect all residues in order to build the full sequence ---
    # Use an ordered dict keyed by seq_id to deduplicate (many atoms per residue)
    residue_map = {}   # seq_id (int) -> three-letter code
    for line in atom_lines:
        cols = line.split()
        if cols[i_group] != "ATOM" or cols[i_chain] != chain_id:
            continue
        seq_id = int(cols[i_seqid])
        if seq_id not in residue_map:
            residue_map[seq_id] = cols[i_resn]

    # Sort by seq_id to ensure correct order, then build sequence + a seq_id -> position index
    sorted_seq_ids = sorted(residue_map.keys())
    sequence = "".join(THREE_TO_ONE.get(residue_map[s], "X") for s in sorted_seq_ids)
    seqid_to_pos = {seq_id: pos for pos, seq_id in enumerate(sorted_seq_ids)}

    # --- Second pass: extract CA/CB coordinates and their positions in the sequence ---
    coords = []
    coord_indices = []
    seen_seqids = set()   # one coordinate per residue

    for line in atom_lines:
        cols = line.split()
        if cols[i_group] != "ATOM" or cols[i_chain] != chain_id:
            continue
        atom_name = cols[i_atom]
        resn      = cols[i_resn]
        seq_id    = int(cols[i_seqid])
        if not ((atom_name == "CA" and resn == "GLY") or
                (atom_name == "CB" and resn != "GLY")):
            continue
        if seq_id in seen_seqids:   # guard against duplicate ATOM records
            continue
        seen_seqids.add(seq_id)
        coords.append([float(cols[i_x]), float(cols[i_y]), float(cols[i_z])])
        coord_indices.append(seqid_to_pos[seq_id])

    return np.array(coords), sequence, coord_indices
