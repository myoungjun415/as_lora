# pip install datasets pandas numpy
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# ======================
# Config
# ======================
SEED = 42
NUM_CLIENTS = 6


# ======================
# Common utilities
# ======================
def get_output_dir(task_name: str) -> Path:
    return Path(f"{task_name}_federated_{NUM_CLIENTS}clients_equal")


def equal_split_dataframe(df_train: pd.DataFrame, num_clients: int = NUM_CLIENTS):
    n = len(df_train)
    base_size = n // num_clients
    remainder = n % num_clients

    clients = []
    start = 0

    for cid in range(num_clients):
        size = base_size + (1 if cid < remainder else 0)
        part = df_train.iloc[start:start + size].reset_index(drop=True)
        start += size
        clients.append(part)

    return clients


def save_clients_and_dev(clients, outdir: Path, dev_df: pd.DataFrame):
    outdir.mkdir(parents=True, exist_ok=True)

    for cid, cdf in enumerate(clients):
        cdf.to_csv(outdir / f"client_{cid}.csv", index=False)

    dev_df.to_csv(outdir / "dev.csv", index=False)


def print_client_sizes(clients, task_name: str):
    print(f"\n[{task_name}] client sizes")
    for cid, cdf in enumerate(clients):
        print(f"Client {cid}: {len(cdf)} samples")


# ======================
# Task-specific preprocessors
# ======================
def prepare_squad11():
    print("\n===== Preparing SQuAD 1.1 =====")
    ds = load_dataset("squad")

    train = ds["train"]
    dev = ds["validation"]

    df_train = pd.DataFrame({
        "id": train["id"],
        "title": train["title"],
        "context": train["context"],
        "question": train["question"],
        "answers": train["answers"],
    }).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    df_dev = pd.DataFrame({
        "id": dev["id"],
        "title": dev["title"],
        "context": dev["context"],
        "question": dev["question"],
        "answers": dev["answers"],
    })

    clients = equal_split_dataframe(df_train)
    outdir = get_output_dir("squad1_1")
    save_clients_and_dev(clients, outdir, df_dev)

    print("Done. Saved to:", outdir.resolve())
    print_client_sizes(clients, "SQuAD 1.1")


def make_is_impossible(answer_list):
    return [len(a["text"]) == 0 for a in answer_list]


def prepare_squad20():
    print("\n===== Preparing SQuAD 2.0 =====")
    ds = load_dataset("squad_v2")

    train = ds["train"]
    dev = ds["validation"]

    train_is_impossible = make_is_impossible(train["answers"])
    dev_is_impossible = make_is_impossible(dev["answers"])

    df_train = pd.DataFrame({
        "id": train["id"],
        "title": train["title"],
        "context": train["context"],
        "question": train["question"],
        "answers": train["answers"],
        "is_impossible": train_is_impossible,
    }).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    df_dev = pd.DataFrame({
        "id": dev["id"],
        "title": dev["title"],
        "context": dev["context"],
        "question": dev["question"],
        "answers": dev["answers"],
        "is_impossible": dev_is_impossible,
    })

    clients = equal_split_dataframe(df_train)
    outdir = get_output_dir("squad2_0")
    save_clients_and_dev(clients, outdir, df_dev)

    print("Done. Saved to:", outdir.resolve())
    print_client_sizes(clients, "SQuAD 2.0")


# ======================
# Main
# ======================
def main():
    prepare_squad11()
    prepare_squad20()
    print("\nSQuAD datasets are processed successfully.")


if __name__ == "__main__":
    main()