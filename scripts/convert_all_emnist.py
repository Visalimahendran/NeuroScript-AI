import numpy as np
import os

EMNIST_FILES = [
    "data/emnist/emnist_letter_train.npz",
    "data/emnist/emnist_number_train.npz",
    "data/emnist/emnist-byclass-train.npz",
    "data/emnist/emnist-byclass-test.npz",
]

OUTPUT_PATH = "data/npz_files/handwriting_data.npz"
os.makedirs("data/npz_files", exist_ok=True)

all_images = []
all_labels = []

def load_images_labels(data, path):
    keys = data.files

    # Common EMNIST formats
    if "images" in keys and "labels" in keys:
        return data["images"], data["labels"]

    if "training_images" in keys and "training_labels" in keys:
        return data["training_images"], data["training_labels"]

    if "testing_images" in keys and "testing_labels" in keys:
        return data["testing_images"], data["testing_labels"]

    if "x_train" in keys and "y_train" in keys:
        return data["x_train"], data["y_train"]

    if "x_test" in keys and "y_test" in keys:
        return data["x_test"], data["y_test"]

    # ❗ Images-only file → skip
    if "testing_images" in keys or "images" in keys:
        print(f"⚠️ Skipping {path} (no labels found)")
        return None, None

    raise KeyError(f"Unsupported NPZ format. Keys found: {keys}")


for path in EMNIST_FILES:
    print("Loading:", path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")

    data = np.load(path)
    images, labels = load_images_labels(data, path)

    if images is None or labels is None:
        continue
    labels = labels.reshape(-1)

    # Normalize
    images = images.astype("float32") / 255.0

    # Ensure correct shape
    if images.ndim == 3:  # (N, 28, 28)
        images = images.reshape(-1, 1, 28, 28)

    all_images.append(images)
    all_labels.append(labels)

# ---- MERGE ----
images = np.concatenate(all_images, axis=0)
labels = np.concatenate(all_labels, axis=0)

# ---- MAP TO 3 MENTAL HEALTH CLASSES ----
# 0 = Normal, 1 = Mild, 2 = Severe
mh_labels = np.zeros_like(labels)
mh_labels[labels > 20] = 1
mh_labels[labels > 35] = 2

# ---- LIMIT SIZE (CPU FRIENDLY) ----
MAX_SAMPLES = 50000
images = images[:MAX_SAMPLES]
mh_labels = mh_labels[:MAX_SAMPLES]

# ---- SAVE FINAL DATASET ----
np.savez(
    OUTPUT_PATH,
    images=images,
    labels=mh_labels
)

print("\n✅ EMNIST conversion COMPLETED successfully!")
print("Saved to:", OUTPUT_PATH)
print("Images shape:", images.shape)
print("Labels distribution:", np.bincount(mh_labels))
