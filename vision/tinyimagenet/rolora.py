from __future__ import annotations

import copy
import csv
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Iterable

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor
from peft import LoraConfig, get_peft_model

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.uniform_sampler import UniformWithReplacementSampler


USE_DP = True
MAX_GRAD_NORM = 2.0
DELTA = 1e-5

MODEL_NAME = "google/vit-base-patch16-224-in21k"
NUM_LABELS = 200

NUM_CLIENTS = 3
ACTIVE_CLIENTS = 3

ROUNDS = 100
LOCAL_STEPS = 10
MAX_LEN = None

AB_SCHEDULE = list("BA")
GPU_LIST = ["cuda:0", "cuda:0"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    full = model.state_dict()
    out = {}

    for k, v in full.items():
        nk = _normalize_key(k)

        if "lora_" in nk:
            out[k] = v.detach().cpu()
        elif nk.endswith("classifier.weight") or nk.endswith("classifier.bias"):
            out[k] = v.detach().cpu()

    return out


def load_lora_state_dict_(model: nn.Module, lora_sd: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        own = model.state_dict()
        own_keys_map = {_normalize_key(k): k for k in own.keys()}

        for k, v in lora_sd.items():
            nk = _normalize_key(k)
            if nk in own_keys_map:
                dst_key = own_keys_map[nk]
                own[dst_key].copy_(v.to(dtype=own[dst_key].dtype, device=own[dst_key].device))

        model.load_state_dict(own)


def load_trainable_state_dict_(model: nn.Module, sd: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        own = model.state_dict()
        own_keys_map = {_normalize_key(k): k for k in own.keys()}

        for k, v in sd.items():
            nk = _normalize_key(k)
            if nk in own_keys_map:
                dst_key = own_keys_map[nk]
                own[dst_key].copy_(v.to(dtype=own[dst_key].dtype, device=own[dst_key].device))

        model.load_state_dict(own)


def clear_grad_samples(model: nn.Module) -> None:
    for p in model.parameters():
        if hasattr(p, "grad_sample"):
            p.grad_sample = None


def mask_lora_grads(model: nn.Module, mode: str) -> None:
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if mode == "A" and "lora_B" in name:
            p.grad.zero_()
        elif mode == "B" and "lora_A" in name:
            p.grad.zero_()


def build_lora_vit(num_labels: int = NUM_LABELS) -> nn.Module:
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    base = AutoModelForImageClassification.from_pretrained(MODEL_NAME, config=config)

    in_features = base.classifier.in_features
    base.classifier = nn.Linear(in_features, num_labels)

    lconf = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["query", "value"],
        bias="none",
    )
    model = get_peft_model(base, lconf)

    for n, p in model.named_parameters():
        if ("lora_" in n) or ("classifier" in n):
            p.requires_grad = True
        else:
            p.requires_grad = False

    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print(f"Model params: total={total / 1e6:.2f}M, trainable={trainable / 1e6:.2f}M")
    return model


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

    def lora_state(self) -> Dict[str, torch.Tensor]:
        return lora_only_state_dict(self.model)

    def trainable_state(self) -> Dict[str, torch.Tensor]:
        return trainable_state_dict(self.model)


class FederatedServer:
    def __init__(self, global_model: nn.Module):
        self.global_model = global_model

    def broadcast(self) -> Dict[str, torch.Tensor]:
        return trainable_state_dict(self.global_model)

    def aggregate_mean_filtered(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        mode: str,
    ):
        if not client_states:
            return None

        agg: Dict[str, torch.Tensor] = {}

        for sd in client_states:
            for k, v in sd.items():
                nk = _normalize_key(k)

                if "lora_" in nk:
                    if mode == "A" and "lora_A" not in nk:
                        continue
                    if mode == "B" and "lora_B" not in nk:
                        continue
                elif nk.endswith("classifier.weight") or nk.endswith("classifier.bias"):
                    pass
                else:
                    continue

                if k not in agg:
                    agg[k] = v.clone().float()
                else:
                    agg[k] += v.float()

        for k in agg:
            agg[k] /= float(len(client_states))

        load_trainable_state_dict_(self.global_model, agg)
        return agg


def make_clients(
    k: int,
    server: FederatedServer,
    lr_a: float,
    lr_b: float,
) -> List[FederatedClient]:
    clients: List[FederatedClient] = []
    server_sd = server.broadcast()

    for i in range(k):
        device = torch.device(GPU_LIST[i % len(GPU_LIST)])
        local_model = copy.deepcopy(server.global_model)
        load_trainable_state_dict_(local_model, server_sd)
        local_model = local_model.to(device=device, dtype=DTYPE)

        params_A = []
        params_B = []

        for n, p in local_model.named_parameters():
            if not p.requires_grad:
                continue

            nk = _normalize_key(n)

            if "lora_A" in nk:
                params_A.append(p)
            elif "lora_B" in nk:
                params_B.append(p)
            elif nk.endswith("classifier.weight") or nk.endswith("classifier.bias"):
                params_A.append(p)
                params_B.append(p)

        optimizer = SGD(
            [
                {"params": params_A, "lr": lr_a},
                {"params": params_B, "lr": lr_b},
            ],
            lr=lr_b,
            weight_decay=0.0,
        )

        clients.append(FederatedClient(cid=i, model=local_model, optimizer=optimizer))

    return clients


def _ensure_processor():
    return AutoImageProcessor.from_pretrained(MODEL_NAME)


class TinyImageNetNPZDatasetProcessor(Dataset):
    def __init__(self, npz_path: Path, processor):
        data = np.load(npz_path)
        self.images = data["images"]
        self.labels = data["labels"]
        self.processor = processor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        y = int(self.labels[i])

        proc = self.processor(images=img, return_tensors="pt")
        pixel_values = proc["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(y, dtype=torch.long),
        }


def make_client_loaders(
    data_dir: Path,
    batch_size: int,
) -> tuple[List[DataLoader], DataLoader]:
    processor = _ensure_processor()
    loaders: List[DataLoader] = []

    for cid in range(NUM_CLIENTS):
        c_ds = TinyImageNetNPZDatasetProcessor(data_dir / f"client_{cid}.npz", processor)

        if USE_DP:
            sample_rate = batch_size / max(1, len(c_ds))
            batch_sampler = UniformWithReplacementSampler(
                num_samples=len(c_ds),
                sample_rate=sample_rate,
            )
            dl = DataLoader(
                c_ds,
                batch_sampler=batch_sampler,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            dl = DataLoader(
                c_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )

        loaders.append(dl)

    dev_ds = TinyImageNetNPZDatasetProcessor(data_dir / "test.npz", processor)
    dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False, num_workers=0)

    return loaders, dev_loader


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
        data = np.load(data_dir / f"client_{cid}.npz")
        client_size = len(data["labels"])
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
        client.model.to(device=DEVICE, dtype=DTYPE)

        dp_loaders.append(dp_loader)

    return dp_loaders


def _cycle(dataloader: DataLoader) -> Iterable:
    while True:
        for b in dataloader:
            yield b


def train_local_steps(
    cid: int,
    client: FederatedClient,
    dataloader: DataLoader,
    steps: int,
    mode: str,
):
    client.model.train()
    batch_iter = _cycle(dataloader)
    device = next(client.model.parameters()).device

    for i in range(steps):
        batch = next(batch_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        client.zero_grad()
        out = client.model(**batch)
        loss = out.loss
        loss.backward()

        mask_lora_grads(client.model, mode)
        client.optimizer.step()
        clear_grad_samples(client.model)

        if i == steps - 1:
            print(f"client{client.cid} last_step_loss={loss.item():.4f}")


@torch.no_grad()
def evaluate_tinyimagenet(model: nn.Module, dev_loader: DataLoader, device=DEVICE) -> float:
    model.eval()
    correct = 0
    total = 0

    for batch in dev_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].numel()

    return correct / max(1, total)


def run_federated_training(args):
    seed_everything(args.seed)

    data_dir = (
        Path(args.data_dir)
        if hasattr(args, "data_dir") and args.data_dir
        else Path("tinyimagenet_federated_3clients_alpha0.5")
    )

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        args.output_dir,
        f"{args.task}_{args.method}_eps{str(args.epsilon).replace('.', 'p')}_bs{args.batch_size}_lra{str(args.lr_a).replace('.', 'p')}_lrb{str(args.lr_b).replace('.', 'p')}_seed{args.seed}_{timestamp}.csv",
    )

    noise_multipliers = compute_client_noise_multipliers(
        data_dir=data_dir,
        num_clients=NUM_CLIENTS,
        batch_size=args.batch_size,
        rounds=ROUNDS,
        local_steps=LOCAL_STEPS,
        target_epsilon=args.epsilon,
        delta=DELTA,
    )

    global_model = build_lora_vit(NUM_LABELS).to(device=DEVICE, dtype=DTYPE)
    server = FederatedServer(global_model)

    clients = make_clients(NUM_CLIENTS, server, lr_a=args.lr_a, lr_b=args.lr_b)
    client_loaders, dev_loader = make_client_loaders(data_dir=data_dir, batch_size=args.batch_size)
    client_loaders = attach_dp_to_clients(clients, client_loaders, noise_multipliers=noise_multipliers)

    rng = random.Random(args.seed)
    best_acc = 0.0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "mode", "dev_acc", "best_acc"])

    for r in range(1, ROUNDS + 1):
        mode = AB_SCHEDULE[(r - 1) % len(AB_SCHEDULE)]
        print(f"\n[Round {r}] mode={mode}")

        active_ids = rng.sample(range(NUM_CLIENTS), ACTIVE_CLIENTS)

        server_sd = server.broadcast()
        for cid in active_ids:
            load_trainable_state_dict_(clients[cid].model, server_sd)

        for cid in active_ids:
            train_local_steps(
                cid=cid,
                client=clients[cid],
                dataloader=client_loaders[cid],
                steps=LOCAL_STEPS,
                mode=mode,
            )

        agg_state = server.aggregate_mean_filtered(
            [clients[cid].trainable_state() for cid in active_ids],
            mode=mode,
        )

        acc = evaluate_tinyimagenet(server.global_model, dev_loader)
        best_acc = max(best_acc, acc)

        print(
            f"[Round {r:03d}] active={active_ids} "
            f"dev_acc={acc:.4f} best_acc={best_acc:.4f} "
            f"agg_keys={len(agg_state) if agg_state else 0}"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([r, mode, acc, best_acc])

        if USE_DP:
            eps_logs = []
            for cid in active_ids:
                pe = clients[cid].privacy_engine
                if pe is not None:
                    eps_logs.append((cid, pe.accountant.get_epsilon(delta=DELTA)))

            if eps_logs:
                eps_str = ", ".join([f"c{cid}: ε={eps:.2f}" for cid, eps in eps_logs])
                print(f"DP spent (δ={DELTA}): {eps_str}")

    print(f"\nFinal best acc = {best_acc:.4f}")
    print(f"Saved log to {log_path}")


def run(args):
    run_federated_training(args)