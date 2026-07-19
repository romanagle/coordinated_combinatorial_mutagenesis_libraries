import pandas as pd

in_csv = "RNCMPT00111_train.csv"      # your current file
out_csv = "RNCMPT00111_train_seq.csv" # new file with sequence + score

L = 41
NUCS = ["A", "C", "G", "U"]
score_col = "RNCMPT00111"

# Load the CSV
df = pd.read_csv(in_csv)

sequences = []

for _, row in df.iterrows():
    seq_chars = []
    for pos in range(L):
        cols = [f"pos{pos}_{n}" for n in NUCS]
        onehot_vals = row[cols].values

        if onehot_vals.sum() == 0:
            # no nucleotide set -> unknown
            base = "N"
        else:
            # pick the nuc with the highest value (should be the 1.0)
            base = NUCS[onehot_vals.argmax()]
        seq_chars.append(base)

    sequences.append("".join(seq_chars))

# Build new dataframe
out_df = pd.DataFrame({
    "sequence": sequences,
    "score": df[score_col].values
})

out_df.to_csv(out_csv, index=False)
print(f"Wrote {out_csv}")