from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


USE_DP = True
MAX_GRAD_NORM = 2.0
DELTA = 1e-5

MODEL_NAME = "roberta-large"
NUM_LABELS = 3
CLIP_NORM = 2.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = (
    torch.float32
    if USE_DP
    else (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float32
    )
)

MAX_LEN = 128
ROUNDS = 100
LOCAL_STEPS = 20
NUM_CLIENTS = 6
ACTIVE_CLIENTS = 6
DATA_DIR = Path("snli_federated_6clients_alpha0.5")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def build_lora_roberta(num_labels: int = NUM_LABELS) -> nn.Module:
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    lconf = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["query", "value"],
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(base, lconf)

    for name, p in model.named_parameters():
        p.requires_grad = "lora" in name

    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Model params: total={total/1e6:.2f}M, trainable={trainable/1e6:.2f}M")

    return model


def _normalize_key(key: str) -> str:
    prefixes = ("module.", "_module.", "model.", "wrapped.", "net.", "base_model.")
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def lora_only_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    full = model.state_dict()
    return {k: v.detach().cpu() for k, v in full.items() if "lora_" in k}


def load_lora_state_dict_(model: nn.Module, lora_sd: Dict[str, torch.Tensor]):
    with torch.no_grad():
        own = model.state_dict()
        own_keys_map = {_normalize_key(k): k for k in own.keys()}

        for k, v in lora_sd.items():
            nk = _normalize_key(k)
            if nk in own_keys_map:
                dst_key = own_keys_map[nk]
                own[dst_key].copy_(v.to(dtype=own[dst_key].dtype, device=own[dst_key].device))

        model.load_state_dict(own)


@dataclass
class FederatedClient:
    cid: int
    model: nn.Module
    optimizer: torch.optim.Optimizer
    privacy_engine: Optional[PrivacyEngine] = None
    dp_wrapped: bool = False

    def to(self, device: torch.device, dtype: torch.dtype):
        self.model.to(device=device, dtype=dtype)
        return self

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def clip_grads(self, max_norm: float = CLIP_NORM):
        grads = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if grads:
            nn.utils.clip_grad_norm_(grads, max_norm)

    def lora_state(self) -> Dict[str, torch.Tensor]:
        return lora_only_state_dict(self.model)


class FederatedServer:
    def __init__(self, global_model: nn.Module):
        self.global_model = global_model

    def broadcast(self) -> Dict[str, torch.Tensor]:
        return lora_only_state_dict(self.global_model)

    def aggregate_mean(self, client_states: List[Dict[str, torch.Tensor]]):
        if not client_states:
            return None

        agg: Dict[str, torch.Tensor] = {}
        for sd in client_states:
            for k, v in sd.items():
                if k not in agg:
                    agg[k] = v.clone().float()
                else:
                    agg[k] += v.float()

        for k in agg:
            agg[k] /= float(len(client_states))

        load_lora_state_dict_(self.global_model, agg)
        return agg


def make_clients(k: int, server: FederatedServer, lr: float) -> List[FederatedClient]:
    clients: List[FederatedClient] = []
    server_lora = server.broadcast()

    for i in range(k):
        local_model = copy.deepcopy(server.global_model)
        load_lora_state_dict_(local_model, server_lora)
        opt = SGD(filter(lambda p: p.requires_grad, local_model.parameters()), lr=lr, weight_decay=0.0)
        client = FederatedClient(cid=i, model=local_model, optimizer=opt).to(DEVICE, DTYPE)
        clients.append(client)

    return clients


def _ensure_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


class SNLIClientDataset(Dataset):
    def __init__(self, csv_path: Path, tokenizer, max_len: int = 128):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["label"].isin([0, 1, 2])]
        self.df = self.df.dropna(subset=["premise", "hypothesis", "label"])
        self.df["premise"] = self.df["premise"].astype(str)
        self.df["hypothesis"] = self.df["hypothesis"].astype(str)
        self.df = self.df.reset_index(drop=True)

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        premise = str(row["premise"])
        hypothesis = str(row["hypothesis"])
        label = int(row["label"])

        enc = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def make_client_loaders(batch_size: int, max_len: int = MAX_LEN) -> List[DataLoader]:
    tok = _ensure_tokenizer()
    loaders: List[DataLoader] = []

    for cid in range(NUM_CLIENTS):
        ds = SNLIClientDataset(DATA_DIR / f"client_{cid}.csv", tok, max_len=max_len)

        if USE_DP:
            sample_rate = batch_size / len(ds)
            batch_sampler = UniformWithReplacementSampler(
                num_samples=len(ds),
                sample_rate=sample_rate,
            )
            dl = DataLoader(
                ds,
                batch_sampler=batch_sampler,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )

        loaders.append(dl)

    return loaders


def make_dev_loader(batch_size: int = 64, max_len: int = MAX_LEN) -> DataLoader:
    tok = _ensure_tokenizer()
    ds = SNLIClientDataset(DATA_DIR / "dev.csv", tok, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


@torch.no_grad()
def evaluate_snli(model: nn.Module, dev_loader: DataLoader, device=DEVICE) -> float:
    model.eval()
    correct = 0
    total = 0
    last_logits = None

    for batch in dev_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
        logits = outputs.logits
        last_logits = logits
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].numel()

    if last_logits is not None:
        probs = torch.softmax(last_logits, dim=-1)
        conf = probs.max(dim=-1).values.mean().item()
        print(f"avg_conf(last_batch)={conf:.3f}")

    return correct / max(1, total)


def train_local_steps(
    client: FederatedClient,
    dataloader: DataLoader,
    steps: int = LOCAL_STEPS,
    physical_batch_size: int = 64,
):
    client.model.train()

    before_state = client.lora_state()
    optimizer = client.optimizer

    data_iter = (
        BatchMemoryManager(
            data_loader=dataloader,
            max_physical_batch_size=physical_batch_size,
            optimizer=optimizer,
        )
        if USE_DP
        else dataloader
    )

    step_cnt = 0

    with data_iter as memory_safe_loader:
        for batch in memory_safe_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            out = client.model(**batch)
            loss = out.loss
            loss.backward()

            if not USE_DP:
                client.clip_grads(CLIP_NORM)

            optimizer.step()

            step_cnt += 1
            if step_cnt == steps:
                print(f"client{client.cid} last_step_loss={loss.item():.4f}")
                break

    clear_grad_samples(client.model)
    log_lora_weight_change(client, before_state)


def clear_grad_samples(model):
    for p in model.parameters():
        if hasattr(p, "grad_sample"):
            p.grad_sample = None


def compute_client_noise_multipliers(
    data_dir: Path,
    num_clients: int,
    batch_size: int,
    rounds: int,
    local_steps: int,
    target_epsilon: float,
    delta: float,
) -> List[float]:
    steps = rounds * local_steps
    sigmas = []

    print(f"DP steps per client: {steps}")
    print(f"Target epsilon: {target_epsilon}, delta: {delta}")
    print("-" * 72)
    print(f"{'Client':>6} | {'Data size':>10} | {'q':>10} | {'sigma':>10}")
    print("-" * 72)

    for cid in range(num_clients):
        df = pd.read_csv(data_dir / f"client_{cid}.csv")
        client_size = len(df)
        q = batch_size / client_size

        sigma = get_noise_multiplier(
            sample_rate=q,
            steps=steps,
            target_epsilon=target_epsilon,
            target_delta=delta,
            accountant="prv",
        )
        sigmas.append(sigma)

        print(f"{cid:>6} | {client_size:>10} | {q:>10.6f} | {sigma:>10.4f}")

    print("-" * 72)
    return sigmas


def attach_dp_to_clients(
    clients: List[FederatedClient],
    client_loaders: List[DataLoader],
    noise_multipliers: List[float],
) -> List[DataLoader]:
    dp_loaders: List[DataLoader] = []

    for cid, client in enumerate(clients):
        if not USE_DP:
            dp_loaders.append(client_loaders[cid])
            continue

        pe = PrivacyEngine()
        model, optimizer, dp_loader = pe.make_private(
            module=client.model,
            optimizer=client.optimizer,
            data_loader=client_loaders[cid],
            noise_multiplier=noise_multipliers[cid],
            max_grad_norm=MAX_GRAD_NORM,
        )

        client.model = model
        client.optimizer = optimizer
        client.privacy_engine = pe
        client.dp_wrapped = True
        dp_loaders.append(dp_loader)

        client.model.to(device=DEVICE, dtype=DTYPE)

    return dp_loaders


def log_lora_weight_change(client: FederatedClient, before: Dict[str, torch.Tensor]):
    after = client.lora_state()
    total_diff, total_norm = 0.0, 0.0

    for k in before.keys():
        diff = (after[k] - before[k]).float()
        total_diff += diff.norm().item()
        total_norm += after[k].float().norm().item()

    ratio = total_diff / (total_norm + 1e-8)
    print(f"[Client {client.cid}] delta_lora={total_diff:.4f}, ratio={ratio:.4e}")


def log_global_lora_change(server: FederatedServer, before: Dict[str, torch.Tensor], round_id: int):
    after = lora_only_state_dict(server.global_model)
    total_diff, total_norm = 0.0, 0.0

    for k in before.keys():
        diff = (after[k] - before[k]).float()
        total_diff += diff.norm().item()
        total_norm += after[k].float().norm().item()

    ratio = total_diff / (total_norm + 1e-8)
    print(f"[Round {round_id}] delta_global_lora={total_diff:.4f}, ratio={ratio:.4e}")


def run_federated_training(args):
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        args.output_dir,
        f"{args.task}_{args.method}_eps{str(args.epsilon).replace('.', 'p')}_bs{args.batch_size}_lra{str(args.lr_a).replace('.', 'p')}_lrb{str(args.lr_b).replace('.', 'p')}_seed{args.seed}_{timestamp}.csv"
    )
    log_rows = []

    global_model = build_lora_roberta(NUM_LABELS).to(device=DEVICE, dtype=DTYPE)
    server = FederatedServer(global_model)

    clients = make_clients(NUM_CLIENTS, server, lr=args.lr_a)

    client_loaders = make_client_loaders(batch_size=args.batch_size)
    dev_loader = make_dev_loader()

    best_acc = 0.0

    noise_multipliers = compute_client_noise_multipliers(
        data_dir=DATA_DIR,
        num_clients=NUM_CLIENTS,
        batch_size=args.batch_size,
        rounds=ROUNDS,
        local_steps=LOCAL_STEPS,
        target_epsilon=args.epsilon,
        delta=DELTA,
    )

    client_loaders = attach_dp_to_clients(
        clients,
        client_loaders,
        noise_multipliers=noise_multipliers,
    )

    rng = random.Random(args.seed)

    for r in range(1, ROUNDS + 1):
        active_ids = rng.sample(range(NUM_CLIENTS), ACTIVE_CLIENTS)

        server_lora = server.broadcast()
        for cid in active_ids:
            load_lora_state_dict_(clients[cid].model, server_lora)

        for cid in active_ids:
            train_local_steps(
                clients[cid],
                client_loaders[cid],
                steps=LOCAL_STEPS,
            )

        before_server = lora_only_state_dict(server.global_model)
        agg_state = server.aggregate_mean([clients[cid].lora_state() for cid in active_ids])
        log_global_lora_change(server, before_server, r)

        acc = evaluate_snli(server.global_model, dev_loader)

        best_acc = max(best_acc, acc)

        log_rows.append({
            "round": r,
            "dev_acc": acc,
            "best_acc": best_acc,
        })

        print(f"[Round {r:03d}] active={active_ids} dev_acc={acc:.4f} agg_keys={len(agg_state) if agg_state else 0}")

    print(f"\nFinal best dev_acc = {best_acc:.4f}\n")
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"Saved training log to {log_path}")


def run(args):
    run_federated_training(args)