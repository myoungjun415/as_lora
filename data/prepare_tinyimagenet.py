# pip install datasets numpy
from datasets import load_dataset
import numpy as np
from pathlib import Path

# ======================
# Config
# ======================
SEED = 42
NUM_CLIENTS = 3
DIR_ALPHA = 0.5

rng = np.random.default_rng(SEED)


# ======================
# Common utilities
# ======================
def get_output_dir(task_name: str) -> Path:
    return Path(f"{task_name}_federated_{NUM_CLIENTS}clients_alpha{DIR_ALPHA}")


def get_image_col(split):
    for col in ["image", "img"]:
        if col in split.column_names:
            return col
    raise KeyError(f"Image column not found. columns={split.column_names}")


def get_label_col(split):
    for col in ["label"]:
        if col in split.column_names:
            return col
    raise KeyError(f"Label column not found. columns={split.column_names}")


def pil_to_rgb_np(image):
    return np.array(image.convert("RGB"), dtype=np.uint8)


def dirichlet_split_indices_by_label(labels, num_classes, num_clients, alpha, rng):
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)

        ratios = rng.dirichlet(alpha * np.ones(num_clients))
        alloc = np.floor(len(idx_c) * ratios).astype(int)
        alloc[-1] += (len(idx_c) - alloc.sum())

        assert alloc.sum() == len(idx_c)

        start = 0
        for cid in range(num_clients):
            k = alloc[cid]
            if k > 0:
                client_indices[cid].append(idx_c[start:start + k])
            start += k

    client_indices = [
        np.concatenate(parts) if len(parts) > 0 else np.array([], dtype=np.int64)
        for parts in client_indices
    ]

    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])

    return client_indices


# ======================
# Tiny-ImageNet preprocessing
# ======================
def prepare_tinyimagenet():
    print("\n===== Preparing Tiny-ImageNet =====")
    ds = load_dataset("zh-plus/tiny-imagenet")

    train = ds["train"]
    valid = ds["valid"]

    img_col = get_image_col(train)
    label_col = get_label_col(train)

    X_train = np.stack([pil_to_rgb_np(im) for im in train[img_col]]).astype(np.uint8)
    y_train = np.array(train[label_col], dtype=np.int64)

    X_valid = np.stack([pil_to_rgb_np(im) for im in valid[img_col]]).astype(np.uint8)
    y_valid = np.array(valid[label_col], dtype=np.int64)

    num_classes = int(y_train.max()) + 1
    assert num_classes == 200, f"Expected 200 classes, got {num_classes}"

    client_indices = dirichlet_split_indices_by_label(
        labels=y_train,
        num_classes=num_classes,
        num_clients=NUM_CLIENTS,
        alpha=DIR_ALPHA,
        rng=rng,
    )

    outdir = get_output_dir("tiny_imagenet")
    outdir.mkdir(parents=True, exist_ok=True)

    for cid, idx in enumerate(client_indices):
        Xc = X_train[idx]
        yc = y_train[idx]
        np.savez_compressed(outdir / f"client_{cid}.npz", images=Xc, labels=yc)

        binc = np.bincount(yc, minlength=num_classes)
        print(
            f"client{cid}: n={len(yc)} | "
            f"nonzero_classes={int((binc > 0).sum())}"
        )

    np.savez_compressed(outdir / "test.npz", images=X_valid, labels=y_valid)

    print("Done. Saved to:", outdir.resolve())


# ======================
# Main
# ======================
def main():
    prepare_tinyimagenet()


if __name__ == "__main__":
    main()