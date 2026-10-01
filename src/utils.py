# NCBI Translation Table 11: Standard Bacterial and Archaeal Code
GENETIC_CODE_11 = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*',
    'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W',
}

# NCBI Translation Table 4: Mycoplasma/Spiroplasma Code
# TGA codes for Tryptophan ('W') instead of Stop ('*')
GENETIC_CODE_4 = GENETIC_CODE_11.copy()
GENETIC_CODE_4['TGA'] = 'W'

TABLE_CONFIGS = {
    11: {
        "name": "Bacterial, Archaeal, and Plant Plastid Code",
        "codon_table": GENETIC_CODE_11,
        "stop_codons": frozenset({"TAA", "TAG", "TGA"}),
        "start_codons": frozenset({"ATG", "GTG", "TTG"})
    },
    4: {
        "name": "Mycoplasma/Spiroplasma Code",
        "codon_table": GENETIC_CODE_4,
        "stop_codons": frozenset({"TAA", "TAG"}),  # TGA is not a stop
        "start_codons": frozenset({"ATG", "GTG", "TTG"})
    }
}


def get_table_config(table_id: int = 11):
    """Returns the translation and boundary rules for a given NCBI genetic code table."""
    if table_id not in TABLE_CONFIGS:
        raise ValueError(
            f"Unsupported prokaryotic genetic code table: {table_id}. "
            f"Supported options: 11 (Standard Bacterial) and 4 (Mycoplasma/Spiroplasma)."
        )
    
    return TABLE_CONFIGS[table_id]


def translate_dna(dna_seq: str, table_id: int = 11) -> str:
    """Translates a DNA coding sequence into an amino acid sequence using the selected table"""
    config = get_table_config(table_id)
    codon_table = config["codon_table"]
    
    seq = dna_seq.upper()
    protein = []
    for i in range(0, len(seq)-2, 3):
        codon = seq[i:i+3]
        aa = codon_table.get(codon, 'X')
        if aa == "*":                    # Stop codon terminates 
            break
        protein.append(aa)

    return "".join(protein)

# Alias in case of typographical reference
trnaslate_dna = translate_dna


def reverse_complement(dna: str) -> str:
    """Returns the reverse complement of a DNA string"""
    comp = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

    return dna.translate(comp)[::-1]
