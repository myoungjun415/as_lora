# pip install datasets pandas numpy
from datasets import load_dataset
import pandas as pd
import numpy as np
from pathlib import Path

# ======================
# Config
# ======================
SEED = 42
NUM_CLIENTS = 6
DIR_ALPHA = 0.5


# ======================
# Common utilities
# ======================
def get_output_dir(task_name: str) -> Path:
    return Path(f"{task_name}_federated_{NUM_CLIENTS}clients_alpha{DIR_ALPHA}")


def dirichlet_split_by_label(
    df_train: pd.DataFrame,
    label_col: str = "label",
    num_clients: int = NUM_CLIENTS,
    alpha: float = DIR_ALPHA,
    seed: int = SEED,
):
    rng = np.random.default_rng(seed)

    labels = sorted(df_train[label_col].unique().tolist())
    label_dfs = []
    label_allocs = []

    for i, label in enumerate(labels):
        df_label = (
            df_train[df_train[label_col] == label]
            .sample(frac=1.0, random_state=seed + i)
            .reset_index(drop=True)
        )
        n_label = len(df_label)

        ratio = rng.dirichlet(alpha * np.ones(num_clients))
        alloc = np.floor(n_label * ratio).astype(int)
        alloc[-1] += (n_label - alloc.sum())

        assert alloc.sum() == n_label

        label_dfs.append(df_label)
        label_allocs.append(alloc)

    clients = []
    starts = [0] * len(labels)

    for cid in range(num_clients):
        parts = []
        for li in range(len(labels)):
            k = label_allocs[li][cid]
            part = label_dfs[li].iloc[starts[li]:starts[li] + k]
            starts[li] += k
            parts.append(part)

        cdf = (
            pd.concat(parts, axis=0)
            .sample(frac=1.0, random_state=seed + cid)
            .reset_index(drop=True)
        )
        clients.append(cdf)

    return clients


def save_clients_and_dev(clients, outdir: Path, dev_df: pd.DataFrame):
    outdir.mkdir(parents=True, exist_ok=True)

    for cid, cdf in enumerate(clients):
        cdf.to_csv(outdir / f"client_{cid}.csv", index=False)

    dev_df.to_csv(outdir / "dev.csv", index=False)


# ======================
# Task-specific preprocessors
# ======================
def prepare_sst2():
    print("\n===== Preparing SST-2 =====")
    ds = load_dataset("glue", "sst2")

    train = ds["train"]
    dev = ds["validation"]

    df_train = pd.DataFrame({
        "sentence": train["sentence"],
        "label": train["label"],
    }).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    df_dev = pd.DataFrame({
        "sentence": dev["sentence"],
        "label": dev["label"],
    })

    clients = dirichlet_split_by_label(df_train)
    outdir = get_output_dir("sst2")
    save_clients_and_dev(clients, outdir, df_dev)

    print("Done. Saved to:", outdir.resolve())


def prepare_qqp():
    print("\n===== Preparing QQP =====")
    ds = load_dataset("glue", "qqp")

    train = ds["train"]
    dev = ds["validation"]

    df_train = pd.DataFrame({
        "question1": train["question1"],
        "question2": train["question2"],
        "label": train["label"],
    }).dropna().sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    df_dev = pd.DataFrame({
        "question1": dev["question1"],
        "question2": dev["question2"],
        "label": dev["label"],
    }).dropna().reset_index(drop=True)

    clients = dirichlet_split_by_label(df_train)
    outdir = get_output_dir("qqp")
    save_clients_and_dev(clients, outdir, df_dev)

    print("Done. Saved to:", outdir.resolve())


def prepare_mnli():
    print("\n===== Preparing MNLI =====")
    ds = load_dataset("glue", "mnli")

    train = ds["train"]
    dev_matched = ds["validation_matched"]
    dev_mismatched = ds["validation_mismatched"]

    df_train = pd.DataFrame({
        "premise": train["premise"],
        "hypothesis": train["hypothesis"],
        "label": train["label"],
    }).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    clients = dirichlet_split_by_label(df_train)
    outdir = get_output_dir("mnli")
    outdir.mkdir(parents=True, exist_ok=True)

    for cid, cdf in enumerate(clients):
        cdf.to_csv(outdir / f"client_{cid}.csv", index=False)

    pd.DataFrame({
        "premise": dev_matched["premise"],
        "hypothesis": dev_matched["hypothesis"],
        "label": dev_matched["label"],
    }).to_csv(outdir / "dev_m.csv", index=False)

    pd.DataFrame({
        "premise": dev_mismatched["premise"],
        "hypothesis": dev_mismatched["hypothesis"],
        "label": dev_mismatched["label"],
    }).to_csv(outdir / "dev_mm.csv", index=False)

    print("Done. Saved to:", outdir.resolve())


def prepare_snli():
    print("\n===== Preparing SNLI =====")
    ds = load_dataset("snli")

    train = ds["train"]
    dev = ds["validation"]

    df_train = pd.DataFrame({
        "premise": train["premise"],
        "hypothesis": train["hypothesis"],
        "label": train["label"],
    })

    df_dev = pd.DataFrame({
        "premise": dev["premise"],
        "hypothesis": dev["hypothesis"],
        "label": dev["label"],
    })

    df_train = df_train[df_train["label"] != -1].reset_index(drop=True)
    df_dev = df_dev[df_dev["label"] != -1].reset_index(drop=True)

    df_train = df_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    clients = dirichlet_split_by_label(df_train)
    outdir = get_output_dir("snli")
    save_clients_and_dev(clients, outdir, df_dev)

    print("Done. Saved to:", outdir.resolve())


def prepare_qnli():
    print("\n===== Preparing QNLI =====")
    ds = load_dataset("glue", "qnli")

    train = ds["train"]
    dev = ds["validation"]

    df_train = pd.DataFrame({
        "question": train["question"],
        "sentence": train["sentence"],
        "label": train["label"],
    }).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    df_dev = pd.DataFrame({
        "question": dev["question"],
        "sentence": dev["sentence"],
        "label": dev["label"],
    })

    clients = dirichlet_split_by_label(df_train)
    outdir = get_output_dir("qnli")
    save_clients_and_dev(clients, outdir, df_dev)

    print("Done. Saved to:", outdir.resolve())


# ======================
# Main
# ======================
def main():
    prepare_sst2()
    prepare_qqp()
    prepare_mnli()
    prepare_snli()
    prepare_qnli()
    print("\nGLUE datasets are processed successfully.")


if __name__ == "__main__":
    main()