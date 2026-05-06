from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Iterable

import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


USE_DP = True
MAX_GRAD_NORM = 2.0
DELTA = 1e-5
GPU_LIST = ["cuda:0", "cuda:0"]

AB_SCHEDULE = list("BA")

MODEL_NAME = "roberta-large"
NUM_LABELS = 2
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
LOCAL_STEPS = 10
NUM_CLIENTS = 6
ACTIVE_CLIENTS = 6
DATA_DIR = Path("qqp_federated_6clients_alpha0.5")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


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


def partial_load_lora_state_dict_(model, server_sd, mode: str):
    with torch.no_grad():
        own = model.state_dict()
        own_keys = {_normalize_key(x): x for x in own.keys()}

        for k, v in server_sd.items():
            nk = _normalize_key(k)

            if mode == "A" and "lora_A" not in nk:
                continue
            if mode == "B" and "lora_B" not in nk:
                continue

            if nk in own_keys:
                dst = own[own_keys[nk]]
                dst.copy_(v.to(dtype=dst.dtype, device=dst.device))

        model.load_state_dict(own)


@dataclass
class FederatedClient:
    cid: int
    model: nn.Module
    optimizer_A: torch.optim.Optimizer
    optimizer_B: torch.optim.Optimizer
    privacy_engine: Optional[PrivacyEngine] = None
    dp_wrapped: bool = False

    def to(self, device: torch.device, dtype: torch.dtype):
        self.model.to(device=device, dtype=dtype)
        return self

    def zero_grad(self):
        self.optimizer_A.zero_grad(set_to_none=True)
        self.optimizer_B.zero_grad(set_to_none=True)

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

        agg = {}
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

    def aggregate_mean_filtered(self, client_states, mode: str):
        if not client_states:
            return None

        agg = {}
        for sd in client_states:
            for k, v in sd.items():
                nk = _normalize_key(k)

                if mode == "A" and "lora_A" not in nk:
                    continue
                if mode == "B" and "lora_B" not in nk:
                    continue

                if k not in agg:
                    agg[k] = v.clone().float()
                else:
                    agg[k] += v.float()

        for k in agg:
            agg[k] /= float(len(client_states))

        load_lora_state_dict_(self.global_model, agg)
        return agg


def make_clients(k: int, server: FederatedServer, lr_a: float, lr_b: float) -> List[FederatedClient]:
    clients = []
    server_lora = server.broadcast()

    for i in range(k):
        device = torch.device(GPU_LIST[i % len(GPU_LIST)])
        local_model = copy.deepcopy(server.global_model)
        load_lora_state_dict_(local_model, server_lora)

        params_A = [p for n, p in local_model.named_parameters() if "lora_A" in n]
        params_B = [p for n, p in local_model.named_parameters() if "lora_B" in n]

        opt_A = SGD(params_A, lr=lr_a)
        opt_B = SGD(params_B, lr=lr_b)

        client = FederatedClient(
            cid=i,
            model=local_model.to(device, DTYPE),
            optimizer_A=opt_A,
            optimizer_B=opt_B,
        )
        clients.append(client)

    return clients


def _ensure_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


class QQPClientDataset(Dataset):
    def __init__(self, csv_path: Path, tokenizer, max_len: int = 128):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=["question1", "question2", "label"]).reset_index(drop=True)
        self.df["question1"] = self.df["question1"].astype(str)
        self.df["question2"] = self.df["question2"].astype(str)

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q1 = str(row["question1"])
        q2 = str(row["question2"])
        label = int(row["label"])

        enc = self.tokenizer(
            q1,
            q2,
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
        ds = QQPClientDataset(DATA_DIR / f"client_{cid}.csv", tok, max_len=max_len)

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
    ds = QQPClientDataset(DATA_DIR / "dev.csv", tok, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


@torch.no_grad()
def evaluate_qqp(model: nn.Module, dev_loader: DataLoader, device=DEVICE) -> float:
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


def _cycle(dataloader: DataLoader) -> Iterable:
    while True:
        for b in dataloader:
            yield b


def mask_lora_grads(model, mode: str):
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if mode == "A" and "lora_B" in name:
            p.grad.zero_()
        elif mode == "B" and "lora_A" in name:
            p.grad.zero_()


def log_lora_weight_change_by_mode(client, before, mode):
    after = client.lora_state()

    stats = {
        "A_diff": 0.0,
        "A_norm": 0.0,
        "B_diff": 0.0,
        "B_norm": 0.0,
    }

    for k in before.keys():
        diff = (after[k] - before[k]).float().norm().item()
        norm = after[k].float().norm().item()

        if "lora_A" in k:
            stats["A_diff"] += diff
            stats["A_norm"] += norm
        elif "lora_B" in k:
            stats["B_diff"] += diff
            stats["B_norm"] += norm

    print(
        f"[Client {client.cid} | mode={mode}] "
        f"delta_A={stats['A_diff']:.4f} (ratio={stats['A_diff']/(stats['A_norm']+1e-8):.2e}) | "
        f"delta_B={stats['B_diff']:.4f} (ratio={stats['B_diff']/(stats['B_norm']+1e-8):.2e})"
    )


def train_local_steps(cid, client, dataloader, steps, mode):
    client.model.train()
    batch_iter = _cycle(dataloader)

    before_state = client.lora_state()
    device = torch.device(GPU_LIST[cid % len(GPU_LIST)])

    for i in range(steps):
        batch = next(batch_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        client.zero_grad()
        out = client.model(**batch)
        loss = out.loss
        loss.backward()

        if not USE_DP:
            client.clip_grads(CLIP_NORM)

        mask_lora_grads(client.model, mode)

        if mode == "A":
            client.optimizer_A.step()
        elif mode == "B":
            client.optimizer_B.step()

        clear_grad_samples(client.model)

        if i == steps - 1:
            print(f"client{client.cid} last_step_loss={loss.item():.4f}")

    log_lora_weight_change_by_mode(client, before_state, mode)


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


def attach_dp_to_clients(clients, client_loaders, noise_multipliers):
    dp_loaders = []

    for cid, client in enumerate(clients):
        if not USE_DP:
            dp_loaders.append(client_loaders[cid])
            continue

        pe = PrivacyEngine()

        model, opt_A, dp_loader = pe.make_private(
            module=client.model,
            optimizer=client.optimizer_A,
            data_loader=client_loaders[cid],
            noise_multiplier=noise_multipliers[cid],
            max_grad_norm=MAX_GRAD_NORM,
        )

        _, opt_B, _ = pe.make_private(
            module=model,
            optimizer=client.optimizer_B,
            data_loader=dp_loader,
            noise_multiplier=noise_multipliers[cid],
            max_grad_norm=MAX_GRAD_NORM,
        )

        client.model = model
        client.optimizer_A = opt_A
        client.optimizer_B = opt_B
        client.privacy_engine = pe
        client.dp_wrapped = True

        dp_loaders.append(dp_loader)
        client.model.to(device=torch.device(GPU_LIST[cid % len(GPU_LIST)]), dtype=DTYPE)

    return dp_loaders


def run_federated_training(args):
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        args.output_dir,
        f"{args.task}_{args.method}_eps{str(args.epsilon).replace('.', 'p')}_bs{args.batch_size}_lra{str(args.lr_a).replace('.', 'p')}_lrb{str(args.lr_b).replace('.', 'p')}_seed{args.seed}_{timestamp}.csv",
    )

    log_rows = []

    global_model = build_lora_roberta(NUM_LABELS).to(device=DEVICE, dtype=DTYPE)
    server = FederatedServer(global_model)

    clients = make_clients(NUM_CLIENTS, server, lr_a=args.lr_a, lr_b=args.lr_b)

    client_loaders = make_client_loaders(batch_size=args.batch_size)
    dev_loader = make_dev_loader()

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
    best_acc = 0.0

    for r in range(1, ROUNDS + 1):
        mode = AB_SCHEDULE[(r - 1) % len(AB_SCHEDULE)]
        print(f"\n[Round {r}] aggregation_mode={mode}")

        active_ids = rng.sample(range(NUM_CLIENTS), ACTIVE_CLIENTS)

        server_lora = server.broadcast()
        for cid in active_ids:
            partial_load_lora_state_dict_(clients[cid].model, server_lora, "A")
            partial_load_lora_state_dict_(clients[cid].model, server_lora, "B")

        for cid in active_ids:
            train_local_steps(cid, clients[cid], client_loaders[cid], steps=LOCAL_STEPS, mode=mode)

        agg_state = server.aggregate_mean_filtered(
            [clients[cid].lora_state() for cid in active_ids],
            mode=mode,
        )

        acc = evaluate_qqp(server.global_model, dev_loader)
        best_acc = max(best_acc, acc)

        log_rows.append({
            "round": r,
            "dev_acc": acc,
            "best_acc": best_acc,
        })

        print(
            f"[Round {r:03d}] active={active_ids} "
            f"dev_acc={acc:.4f} "
            f"agg_keys={len(agg_state) if agg_state else 0}"
        )

    print(f"\nFinal best dev_acc = {best_acc:.4f}\n")
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"Saved training log to {log_path}")


def run(args):
    run_federated_training(args)