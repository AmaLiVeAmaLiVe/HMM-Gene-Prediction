import os
from Bio import SeqIO

def test_gff3_format_compliance():
    gff_path = "data/test/NC_000913.3/NC_000913.3_predicted.gff3"
    if not os.path.exists(gff_path):
        return  # Run after generation

    with open(gff_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert lines[0] == "##gff-version 3", "First line of GFF3 must be '##gff-version 3'"

    data_lines = [l for l in lines if not l.startswith("#")]
    for line in data_lines:
        fields = line.split("\t")
        assert len(fields) == 9, f"GFF3 record must have 9 tab-delimited columns: {line}"
        
        start = int(fields[3])
        end = int(fields[4])
        strand = fields[6]
        attributes = fields[8]

        assert start >= 1, f"GFF3 start coordinates must be >= 1: {start}"
        assert end >= start, f"GFF3 end must be >= start: {start} > {end}"
        assert strand in ("+", "-"), f"Strand must be '+' or '-': {strand}"
        assert "ID=" in attributes, f"Attributes column missing 'ID=' tag: {attributes}"

def test_protein_faa_compliance():
    faa_path = "data/test/NC_000913.3/NC_000913.3_proteins.faa"
    if not os.path.exists(faa_path):
        return  # Run after generation

    records = list(SeqIO.parse(faa_path, "fasta"))
    assert len(records) > 0, "Protein FASTA is empty"

    valid_aas = set("ACDEFGHIKLMNPQRSTVWYX")
    for rec in records:
        seq_str = str(rec.seq).upper()
        assert len(seq_str) >= 60, f"Protein sequence shorter than minimum 60 aa (180 bp): {len(seq_str)}"
        assert not seq_str.endswith("*"), "Translation utility should strip terminal stop codons"
        assert set(seq_str).issubset(valid_aas), f"Encountered non-standard amino acid characters in {rec.id}"

if __name__ == "__main__":
    test_gff3_format_compliance()
    test_protein_faa_compliance()
    print("[+] All output format compliance checks passed successfully.")