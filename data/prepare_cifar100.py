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
    for col in ["img", "image"]:
        if col in split.column_names:
            return col
    raise KeyError(f"Image column not found. columns={split.column_names}")


def get_label_col(split):
    for col in ["fine_label"]:
        if col in split.column_names:
            return col
    raise KeyError(f"Label column not found. columns={split.column_names}")


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
# CIFAR-100 preprocessing
# ======================
def prepare_cifar100():
    print("\n===== Preparing CIFAR-100 =====")
    ds = load_dataset("cifar100")

    train = ds["train"]
    test = ds["test"]

    img_col = get_image_col(train)
    label_col = get_label_col(train)

    X_train = np.stack([np.array(im) for im in train[img_col]]).astype(np.uint8)
    y_train = np.array(train[label_col], dtype=np.int64)

    X_test = np.stack([np.array(im) for im in test[img_col]]).astype(np.uint8)
    y_test = np.array(test[label_col], dtype=np.int64)

    num_classes = int(y_train.max()) + 1
    assert num_classes == 100, f"Expected 100 classes, got {num_classes}"

    client_indices = dirichlet_split_indices_by_label(
        labels=y_train,
        num_classes=num_classes,
        num_clients=NUM_CLIENTS,
        alpha=DIR_ALPHA,
        rng=rng,
    )

    outdir = get_output_dir("cifar100")
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

    np.savez_compressed(outdir / "test.npz", images=X_test, labels=y_test)

    print("Done. Saved to:", outdir.resolve())


# ======================
# Main
# ======================
def main():
    prepare_cifar100()


if __name__ == "__main__":
    main()