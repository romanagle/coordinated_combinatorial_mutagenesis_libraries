import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("RNCMPT00111_graphs/RNCMPT00111_train_seq.csv")

plt.figure()
plt.hist(df["score"], bins=5000, density=True)
plt.xlabel("Score")
plt.ylabel("Density")
plt.xlim((-5, 10))
plt.title("Score distribution of RNCMPT00111 Training Data")
plt.tight_layout()
plt.savefig("dist_of_training_scores.png", dpi=200)
plt.close()

