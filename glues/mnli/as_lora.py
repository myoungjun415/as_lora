from __future__ import annotations

import copy
import csv
import math
import os
import random
import re
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


USE_RANDOM_PROJ = True
PROJ_RATIO = 1.0
PROJ_MATS = {}

NUM_LAYERS = 24
WARMUP_FRAC = 0.1
EMA_BETA = 0.9
SCORE_EPS = 1e-8
FD_EPS = 1e-3

USE_DP = True
MAX_GRAD_NORM = 2.0
DELTA = 1e-5

AB_SCHEDULE = list("BA")

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
LOCAL_STEPS = 10
NUM_CLIENTS = 6
ACTIVE_CLIENTS = 6
DATA_DIR = Path("mnli_federated_6clients_alpha0.5")

T0 = 1.0
Tmin = 0.1
GAMMA = 0.95
EXPLORE_P = 0.0

_LAYER_RE = re.compile(r"\.layer\.(\d+)\.")


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


def get_layer_id_from_name(name: str) -> Optional[int]:
    m = _LAYER_RE.search(name)
    if m:
        return int(m.group(1))
    return None


def is_lora_A(name: str) -> bool:
    return "lora_A" in name


def is_lora_B(name: str) -> bool:
    return "lora_B" in name


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


def get_or_create_projection(lid: int, mode: str, dim: int, device):
    key = (lid, mode, dim, str(device))

    if key in PROJ_MATS:
        return PROJ_MATS[key]

    if PROJ_RATIO != 1.0:
        raise ValueError("This implementation assumes dimension-preserving projection, so PROJ_RATIO must be 1.0.")

    R = torch.randn(dim, dim, device=device, dtype=torch.float32) / math.sqrt(dim)
    PROJ_MATS[key] = R
    return R


def project_lora_gradient(name: str, g: torch.Tensor, lid: int) -> torch.Tensor:
    if not USE_RANDOM_PROJ:
        return g

    if g.ndim != 2:
        flat = g.view(-1)
        dim = flat.numel()
        R = get_or_create_projection(lid, "flat", dim, g.device)
        z = torch.matmul(R, flat)
        return z.view_as(g)

    if is_lora_A(name):
        dim = g.shape[1]
        R = get_or_create_projection(lid, "A", dim, g.device)
        return g @ R

    if is_lora_B(name):
        dim = g.shape[0]
        R = get_or_create_projection(lid, "B", dim, g.device)
        return R @ g

    return g


def compute_loss_no_grad(model: nn.Module, batch: Dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        out = model(**batch)
        return float(out.loss.detach().float().item())


def fd_curvature_for_param(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    param_name: str,
    projected_grad: torch.Tensor,
    fd_eps: float,
) -> float:
    params = dict(model.named_parameters())
    if param_name not in params:
        return 0.0

    p = params[param_name]
    gnorm = projected_grad.norm().item()
    if gnorm < 1e-12:
        return 0.0

    v = projected_grad / (projected_grad.norm() + SCORE_EPS)

    was_training = model.training
    model.eval()

    with torch.no_grad():
        base_loss = compute_loss_no_grad(model, batch)

        p.add_(fd_eps * v)
        plus_loss = compute_loss_no_grad(model, batch)

        p.add_(-2.0 * fd_eps * v)
        minus_loss = compute_loss_no_grad(model, batch)

        p.add_(fd_eps * v)

    if was_training:
        model.train()

    curvature = (plus_loss - 2.0 * base_loss + minus_loss) / (fd_eps ** 2)
    return float(curvature)


def fd_score_layerwise(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    eta: float,
    fd_eps: float = FD_EPS,
):
    scores = [{"A": 0.0, "B": 0.0} for _ in range(NUM_LAYERS)]

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if "lora_" not in name:
            continue

        lid = get_layer_id_from_name(name)
        if lid is None or lid >= NUM_LAYERS:
            continue

        g = p.grad.detach().float()
        g_tilde = project_lora_gradient(name, g, lid)
        g2 = torch.sum(g_tilde * g_tilde).item()

        if g2 < 1e-12:
            continue

        curvature = fd_curvature_for_param(
            model=model,
            batch=batch,
            param_name=name,
            projected_grad=g_tilde,
            fd_eps=fd_eps,
        )

        score = g2 - 0.5 * eta * curvature * g2

        if is_lora_A(name):
            scores[lid]["A"] += score
        elif is_lora_B(name):
            scores[lid]["B"] += score

    return scores


@dataclass
class FederatedClient:
    cid: int
    model: nn.Module
    optimizer: Optional[torch.optim.Optimizer] = None
    privacy_engine: Optional[PrivacyEngine] = None
    dp_wrapped: bool = False

    def to(self, device: torch.device, dtype: torch.dtype):
        self.model.to(device=device, dtype=dtype)
        return self

    def zero_grad(self):
        if self.optimizer is None:
            raise ValueError("optimizer is None")
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

    def aggregate_mean_layerwise(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        layer_modes: List[str],
    ):
        if not client_states:
            return None

        agg = {}

        for sd in client_states:
            for k, v in sd.items():
                nk = _normalize_key(k)

                if "lora_" not in nk:
                    continue

                lid = get_layer_id_from_name(nk)
                if lid is None or lid >= len(layer_modes):
                    continue

                mode = layer_modes[lid]

                if mode == "A" and not is_lora_A(nk):
                    continue
                if mode == "B" and not is_lora_B(nk):
                    continue

                if k not in agg:
                    agg[k] = v.clone().float()
                else:
                    agg[k] += v.float()

        for k in agg:
            agg[k] /= float(len(client_states))

        load_lora_state_dict_(self.global_model, agg)
        return agg


def make_clients(k: int, server: FederatedServer) -> List[FederatedClient]:
    clients = []
    server_lora = server.broadcast()

    for i in range(k):
        local_model = copy.deepcopy(server.global_model)
        load_lora_state_dict_(local_model, server_lora)

        client = FederatedClient(
            cid=i,
            model=local_model.to(DEVICE, DTYPE),
            optimizer=None,
        )
        clients.append(client)

    return clients


def _ensure_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


class MNLIClientDataset(Dataset):
    def __init__(self, csv_path: Path, tokenizer, max_len: int = 128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        premise = row["premise"]
        hypothesis = row["hypothesis"]
        label = int(row["label"])

        if not isinstance(premise, str):
            premise = "" if pd.isna(premise) else str(premise)
        if not isinstance(hypothesis, str):
            hypothesis = "" if pd.isna(hypothesis) else str(hypothesis)

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


def select_param_groups_by_layer_modes(
    model: nn.Module,
    layer_modes: List[str],
):
    assert len(layer_modes) == NUM_LAYERS

    params_A = []
    params_B = []

    for name, p in model.named_parameters():
        p.requires_grad = False

        if "lora_" not in name:
            continue

        lid = get_layer_id_from_name(name)
        if lid is None:
            continue

        mode = layer_modes[lid]

        if mode == "A" and is_lora_A(name):
            p.requires_grad = True
            params_A.append(p)
        elif mode == "B" and is_lora_B(name):
            p.requires_grad = True
            params_B.append(p)

    return params_A, params_B


def make_client_loaders(batch_size: int, max_len: int = MAX_LEN) -> List[DataLoader]:
    tok = _ensure_tokenizer()
    loaders: List[DataLoader] = []

    for cid in range(NUM_CLIENTS):
        ds = MNLIClientDataset(DATA_DIR / f"client_{cid}.csv", tok, max_len=max_len)

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


def make_dev_loaders(batch_size: int = 64, max_len: int = MAX_LEN):
    tok = _ensure_tokenizer()
    ds_m = MNLIClientDataset(DATA_DIR / "dev_m.csv", tok, max_len=max_len)
    ds_mm = MNLIClientDataset(DATA_DIR / "dev_mm.csv", tok, max_len=max_len)

    dl_m = DataLoader(ds_m, batch_size=batch_size, shuffle=False, drop_last=False)
    dl_mm = DataLoader(ds_mm, batch_size=batch_size, shuffle=False, drop_last=False)
    return dl_m, dl_mm


def ema_update(prev: float | None, new: float, beta: float) -> float:
    if prev is None:
        return new
    return beta * prev + (1.0 - beta) * new


@torch.no_grad()
def evaluate_mnli(model: nn.Module, dev_loader: DataLoader, device=DEVICE) -> float:
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


def get_temperature(r, warmup_rounds, T0=0.8, Tmin=0.1, gamma=0.97):
    if r <= warmup_rounds:
        return T0
    k = r - warmup_rounds
    return max(Tmin, T0 * (gamma ** k))


def choose_mode_softmax(SA, SB, temp=0.2, explore_p=0.05):
    a = SA if SA is not None else 0.0
    b = SB if SB is not None else 0.0

    if random.random() < explore_p:
        return "A" if random.random() < 0.5 else "B", "explore"

    ea = math.exp(a / max(1e-6, temp))
    eb = math.exp(b / max(1e-6, temp))
    pA = ea / (ea + eb)
    return ("A" if random.random() < pA else "B"), f"soft(pA={pA:.2f})"


def compute_mode_probs(
    SA: float | None,
    SB: float | None,
    temp: float,
    eps: float = 1e-8,
):
    if SA is None and SB is None:
        return 0.5, 0.5
    if SA is None:
        return 0.0, 1.0
    if SB is None:
        return 1.0, 0.0

    t = max(temp, eps)
    a = SA / t
    b = SB / t

    m = max(a, b)
    ea = math.exp(a - m)
    eb = math.exp(b - m)

    Z = ea + eb
    pA = ea / Z
    pB = eb / Z

    return pA, pB


def clear_grad_samples(model):
    for p in model.parameters():
        if hasattr(p, "grad_sample"):
            p.grad_sample = None


def train_local_steps(
    cid,
    client,
    dataloader,
    round_id,
    steps,
    eta,
):
    client.model.train()
    batch_iter = _cycle(dataloader)
    layer_scores = [{"A": 0.0, "B": 0.0} for _ in range(NUM_LAYERS)]

    for i in range(steps):
        batch = next(batch_iter)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        client.zero_grad()
        out = client.model(**batch)
        loss = out.loss
        loss.backward()

        if i == steps - 1:
            layer_scores = fd_score_layerwise(
                model=client.model,
                batch=batch,
                eta=eta,
                fd_eps=FD_EPS,
            )
            print(f"client{client.cid} last_step_loss={loss.item():.4f}")

        if not USE_DP:
            client.clip_grads(CLIP_NORM)

        client.optimizer.step()
        clear_grad_samples(client.model)

    return layer_scores


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


pe_list = [PrivacyEngine() for _ in range(NUM_CLIENTS)]


def attach_dp_to_clients(
    clients: List[FederatedClient],
    client_loaders: List[DataLoader],
    layer_modes: List[str],
    lr_a: float,
    lr_b: float,
    noise_multipliers: List[float],
):
    assert len(layer_modes) == NUM_LAYERS
    dp_loaders = []

    for cid, client in enumerate(clients):
        if not USE_DP:
            dp_loaders.append(client_loaders[cid])
            continue

        new_model = build_lora_roberta(NUM_LABELS)
        new_model.to(device=DEVICE, dtype=DTYPE)

        with torch.no_grad():
            src = client.model._module if hasattr(client.model, "_module") else client.model
            new_model.load_state_dict(src.state_dict(), strict=True)
            lora_sd = lora_only_state_dict(client.model)
            load_lora_state_dict_(new_model, lora_sd)

        params_A, params_B = select_param_groups_by_layer_modes(new_model, layer_modes)

        if len(params_A) == 0 and len(params_B) == 0:
            raise RuntimeError(f"[Client {cid}] No LoRA params selected. Check layer_modes.")

        param_groups = []
        if len(params_A) > 0:
            param_groups.append({"params": params_A, "lr": lr_a})
        if len(params_B) > 0:
            param_groups.append({"params": params_B, "lr": lr_b})

        optimizer = SGD(param_groups, weight_decay=0.0)

        new_model, optimizer, dp_loader = pe_list[cid].make_private(
            module=new_model,
            optimizer=optimizer,
            data_loader=client_loaders[cid],
            noise_multiplier=noise_multipliers[cid],
            max_grad_norm=MAX_GRAD_NORM,
        )

        client.model = new_model
        client.optimizer = optimizer
        client.privacy_engine = pe_list[cid]
        client.dp_wrapped = True

        dp_loaders.append(dp_loader)

    return dp_loaders


def log_global_lora_change(server: FederatedServer, before: Dict[str, torch.Tensor], round_id: int):
    after = lora_only_state_dict(server.global_model)
    total_diff, total_norm = 0.0, 0.0

    for k in before.keys():
        diff = (after[k] - before[k]).float()
        total_diff += diff.norm().item()
        total_norm += after[k].float().norm().item()

    ratio = total_diff / (total_norm + 1e-8)
    print(f"[Round {round_id}] delta_global_lora={total_diff:.4f}, ratio={ratio:.4e}")


def fmt(x, fmt_str=".2e"):
    if x is None:
        return "None"
    return format(x, fmt_str)


def run_federated_training(args):
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = os.path.join(
        args.output_dir,
        f"{args.task}_{args.method}_eps{str(args.epsilon).replace('.', 'p')}_bs{args.batch_size}_lra{str(args.lr_a).replace('.', 'p')}_lrb{str(args.lr_b).replace('.', 'p')}_seed{args.seed}_{timestamp}.csv",
    )
    mode_csv_path = os.path.join(
        args.output_dir,
        f"{args.task}_{args.method}_layer_modes_{timestamp}.csv",
    )

    global_model = build_lora_roberta(NUM_LABELS).to(device=DEVICE, dtype=DTYPE)
    server = FederatedServer(global_model)

    clients = make_clients(NUM_CLIENTS, server)
    base_client_loaders = make_client_loaders(batch_size=args.batch_size)
    dev_loader_m, dev_loader_mm = make_dev_loaders()

    noise_multipliers = compute_client_noise_multipliers(
        data_dir=DATA_DIR,
        num_clients=NUM_CLIENTS,
        batch_size=args.batch_size,
        rounds=ROUNDS,
        local_steps=LOCAL_STEPS,
        target_epsilon=args.epsilon,
        delta=DELTA,
    )

    rng = random.Random(args.seed)
    warmup_rounds = int(ROUNDS * WARMUP_FRAC)

    S_A_ema = [None] * NUM_LAYERS
    S_B_ema = [None] * NUM_LAYERS
    mode_log = []
    acc_m_log = []
    acc_mm_log = []
    acc_avg_log = []

    best_m = 0.0
    best_mm = 0.0
    best_avg = 0.0

    layer_modes = ["A"] * NUM_LAYERS
    layer_mode_src = [""] * NUM_LAYERS

    for r in range(1, ROUNDS + 1):
        if r <= warmup_rounds:
            base_mode = AB_SCHEDULE[(r - 1) % len(AB_SCHEDULE)]
            for l in range(NUM_LAYERS):
                layer_modes[l] = base_mode
                layer_mode_src[l] = "warmup"
        else:
            temp = get_temperature(
                r,
                warmup_rounds,
                T0=T0,
                Tmin=Tmin,
                gamma=GAMMA,
            )
            for l in range(NUM_LAYERS):
                m, tag = choose_mode_softmax(
                    S_A_ema[l],
                    S_B_ema[l],
                    temp=temp,
                    explore_p=EXPLORE_P,
                )
                layer_modes[l] = m
                layer_mode_src[l] = f"adaptive-{tag}"

        mode_row = {"round": r}
        for l in range(NUM_LAYERS):
            mode_row[f"L{l:02d}"] = layer_modes[l]
        mode_log.append(mode_row)

        active_ids = rng.sample(range(NUM_CLIENTS), ACTIVE_CLIENTS)

        server_lora = server.broadcast()
        for cid in active_ids:
            load_lora_state_dict_(clients[cid].model, server_lora)

        dp_client_loaders = attach_dp_to_clients(
            clients,
            base_client_loaders,
            layer_modes,
            lr_a=args.lr_a,
            lr_b=args.lr_b,
            noise_multipliers=noise_multipliers,
        )

        client_layer_scores = []
        eta_for_score = max(args.lr_a, args.lr_b)

        for cid in active_ids:
            layer_scores = train_local_steps(
                cid=cid,
                client=clients[cid],
                dataloader=dp_client_loaders[cid],
                round_id=r,
                steps=LOCAL_STEPS,
                eta=eta_for_score,
            )
            client_layer_scores.append(layer_scores)

        avg_layer_scores = [{"A": 0.0, "B": 0.0} for _ in range(NUM_LAYERS)]
        K = len(client_layer_scores)

        for scores in client_layer_scores:
            for l in range(NUM_LAYERS):
                avg_layer_scores[l]["A"] += scores[l]["A"]
                avg_layer_scores[l]["B"] += scores[l]["B"]

        for l in range(NUM_LAYERS):
            avg_layer_scores[l]["A"] /= K
            avg_layer_scores[l]["B"] /= K

        for l in range(NUM_LAYERS):
            if layer_modes[l] == "A":
                S_A_ema[l] = ema_update(S_A_ema[l], avg_layer_scores[l]["A"], EMA_BETA)
            elif layer_modes[l] == "B":
                S_B_ema[l] = ema_update(S_B_ema[l], avg_layer_scores[l]["B"], EMA_BETA)

        print(
            "[Layer scores]",
            ", ".join(
                f"L{l}:A={avg_layer_scores[l]['A']:.2e},B={avg_layer_scores[l]['B']:.2e}"
                for l in range(NUM_LAYERS)
            )
        )

        print(f"\n[Round {r:03d} | Layer-wise A/B scores and probabilities]")
        for l in range(NUM_LAYERS):
            if r <= warmup_rounds:
                pA = 1.0 if layer_modes[l] == "A" else 0.0
                pB = 1.0 - pA
                temp_str = "warmup"
            else:
                pA, pB = compute_mode_probs(S_A_ema[l], S_B_ema[l], temp)
                temp_str = f"T={temp:.2f}"

            print(
                f"  L{l:02d} | "
                f"raw(A={avg_layer_scores[l]['A']:.2e}, B={avg_layer_scores[l]['B']:.2e}) | "
                f"ema(A={fmt(S_A_ema[l])}, B={fmt(S_B_ema[l])}) | "
                f"pA={pA:.2f} pB={pB:.2f} | "
                f"mode={layer_modes[l]} ({layer_mode_src[l]}) | "
                f"{temp_str}"
            )

        before_server = lora_only_state_dict(server.global_model)

        agg_state = server.aggregate_mean_layerwise(
            [clients[cid].lora_state() for cid in active_ids],
            layer_modes=layer_modes,
        )

        log_global_lora_change(server, before_server, r)

        acc_m = evaluate_mnli(server.global_model, dev_loader_m)
        acc_mm = evaluate_mnli(server.global_model, dev_loader_mm)
        acc_avg = 0.5 * (acc_m + acc_mm)

        acc_m_log.append(acc_m)
        acc_mm_log.append(acc_mm)
        acc_avg_log.append(acc_avg)

        best_m = max(best_m, acc_m)
        best_mm = max(best_mm, acc_mm)
        best_avg = max(best_avg, acc_avg)

        print(
            f"[Round {r:03d}] active={active_ids} "
            f"dev_m_acc={acc_m:.4f} "
            f"dev_mm_acc={acc_mm:.4f} "
            f"dev_avg_acc={acc_avg:.4f} "
            f"agg_keys={len(agg_state) if agg_state else 0}"
        )

    pd.DataFrame(mode_log).to_csv(mode_csv_path, index=False)

    pd.DataFrame(
        {
            "round": list(range(1, len(acc_m_log) + 1)),
            "dev_m_acc": acc_m_log,
            "dev_mm_acc": acc_mm_log,
            "dev_avg_acc": acc_avg_log,
            "best_m_acc": np.maximum.accumulate(acc_m_log),
            "best_mm_acc": np.maximum.accumulate(acc_mm_log),
            "best_avg_acc": np.maximum.accumulate(acc_avg_log),
        }
    ).to_csv(log_path, index=False)

    print(f"Saved training log to {log_path}")
    print(f"Saved layer-wise modes to {mode_csv_path}")
    print(f"\nFinal best dev_m_acc = {best_m:.4f}")
    print(f"Final best dev_mm_acc = {best_mm:.4f}")
    print(f"Final best dev_avg_acc = {best_avg:.4f}\n")


def run(args):
    run_federated_training(args)