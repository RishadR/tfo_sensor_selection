import numpy as np
import pandas as pd

CSV_PATH = 'data/invivo_data.csv'
N_SPLITS = 3
BOUNDARY = 60

data = pd.read_csv(CSV_PATH)
print(data.head(10))
n = len(data)
n_boundaries = N_SPLITS - 1
group_size = (n - n_boundaries * BOUNDARY) // N_SPLITS

labels = list("abcde")[:N_SPLITS]
validation_idx = np.empty(n, dtype=object)
pos = 0
for i, label in enumerate(labels):
    end = pos + group_size if i < N_SPLITS - 1 else n
    validation_idx[pos:end] = label
    pos = end
    if i < N_SPLITS - 1:
        validation_idx[pos:pos + BOUNDARY] = "z"
        pos += BOUNDARY

data["validation_idx"] = validation_idx
data.to_csv(CSV_PATH, index=False)

print(data["validation_idx"].value_counts().sort_index())
print(f"Saved to {CSV_PATH}")