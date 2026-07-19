import random
from itertools import combinations

WT = 'AAGAACGUUAGAUACUUCGAUAGGAACACAGUUGUGCUGAA'
L = len(WT)
NUCS = ['A', 'C', 'G', 'U']

sequences = []
seen = set()

def add(seq):
    if seq not in seen:
        seen.add(seq)
        sequences.append(seq)

# 1. Exhaustive single mutants (SSM): 41 positions × 3 non-WT nucs = 123
for i in range(L):
    for nuc in NUCS:
        if nuc != WT[i]:
            s = WT[:i] + nuc + WT[i+1:]
            add(s)

n_ssm = len(sequences)
print(f"SSM sequences: {n_ssm}")

# 2. All double mutants (pairwise): C(41,2) × 3 × 3 = 820 × 9 = 7380
for i, j in combinations(range(L), 2):
    for ni in NUCS:
        if ni == WT[i]:
            continue
        for nj in NUCS:
            if nj == WT[j]:
                continue
            s = list(WT)
            s[i] = ni
            s[j] = nj
            add(''.join(s))

n_pairwise = len(sequences) - n_ssm
print(f"Pairwise double-mutant sequences: {n_pairwise}")

# 3. 4 random triple mutants
random.seed(42)
n_triple = 0
while n_triple < 4:
    positions = random.sample(range(L), 3)
    s = list(WT)
    for pos in positions:
        alt = [n for n in NUCS if n != WT[pos]]
        s[pos] = random.choice(alt)
    add(''.join(s))
    new_count = len(sequences) - n_ssm - n_pairwise
    if new_count > n_triple:
        n_triple = new_count

print(f"Random triple-mutant sequences: {n_triple}")
print(f"\nTotal library size: {len(sequences)}")

out_path = 'mRNA_RBP/outputs/msi1_library.txt'
with open(out_path, 'w') as f:
    for seq in sequences:
        f.write(seq + '\n')

print(f"Saved to {out_path}")
