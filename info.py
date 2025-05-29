import numpy as np
from pathlib import Path
import numpy as np
y = np.load("preprocessed_2a/A01/train_y.npy")
print(np.unique(y))  # → [0 1 2 3]가 되어야 함


# root = Path("./preprocessed_2b")  # or "./preprocessed" if that's your folder

# for subj_dir in sorted(root.iterdir()):
#     if not subj_dir.is_dir():
#         continue
#     try:
#         Xtr = np.load(subj_dir / "train_X.npy")
#         ytr = np.load(subj_dir / "train_y.npy")
#         Xte = np.load(subj_dir / "test_X.npy")
#         yte = np.load(subj_dir / "test_y.npy")

#         print(f"{subj_dir.name}:")
#         print(f"  train_X shape: {Xtr.shape}, train_y shape: {ytr.shape}")
#         print(f"  test_X  shape: {Xte.shape}, test_y  shape: {yte.shape}")
#     except Exception as e:
#         print(f"❌ Failed to load from {subj_dir.name}: {e}")
