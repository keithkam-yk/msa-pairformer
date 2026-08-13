import numpy as np

# Amino acid code to character
code2aa_d = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V"
}

# Tokenize amino acids
aa2tok_d = {
    "A": 0, # ALA
    "R": 1, # ARG
    "N": 2, # ASN
    "D": 3, # ASP
    "C": 4, # CYS
    "E": 5, # GLU
    "Q": 6, # GLN
    "G": 7, # GLY
    "H": 8, # HIS
    "I": 9, # ILE
    "L": 10, # LEU
    "K": 11, # LYS
    "M": 12, # MET
    "F": 13, # PHE
    "P": 14, # PRO
    "S": 15, # SER
    "T": 16, # THR
    "W": 17, # TRP
    "Y": 18, # TYR
    "V": 19, # VAL
    "X": 20, # UNK
    "B": 21, # ASP or ASN
    "Z": 22, # GLU or GLN
    "U": 23, # SEC
    "O": 24, # PYL
    "-": 25, # GAP
    "<pad>": 26, # Padded positions
    "<mask>": 27, # Mask
}
tok2aa_d = {aa2tok_d[k]:k for k in aa2tok_d}
nTokenTypes = len(np.unique(list(aa2tok_d.values())))

# ESM token to amino acid
ESM_SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>"
]
esm_tok2aa_d = dict(enumerate(ESM_SEQUENCE_VOCAB))
esm_aa2tok_d = {alph: ind for ind, alph in enumerate(ESM_SEQUENCE_VOCAB)}
esmtok_to_pairformertok_d = {esm_aa2tok_d[aa]: aa2tok_d[aa] if aa in aa2tok_d else -1 for aa in ESM_SEQUENCE_VOCAB}
esmtok_to_pairformertok_d[esm_aa2tok_d['<unk>']] = aa2tok_d['X'] # Handle unknown tokens with X
# MSA Pairformer was trained without insertions ".", so they are not part of the vocabulary
# Full list of tokens that do not appear in the MSA Pairformer are <cls>, <eos>, <unk>, <null_1>, <.>
# All are mapped to -1 using esmtok_to_pairformertok_d
