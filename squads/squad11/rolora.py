from __future__ import annotations

import ast
import collections
import copy
import csv
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Iterable

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

USE_DP = True
MAX_GRAD_NORM = 2.0
DELTA = 1e-5

MODEL_NAME = "roberta-large"
AB_SCHEDULE = list("BA")

NUM_CLIENTS = 6
ACTIVE_CLIENTS = 6

MAX_LEN = 384
DOC_STRIDE = 128
ROUNDS = 100
LOCAL_STEPS = 10

GPU_LIST = ["cuda:0", "cuda:0"]

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


def seed_everything(seed: int = 45):
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


def lora_only_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    full = model.state_dict()
    return {k: v.detach().cpu() for k, v in full.items() if "lora_" in k}


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


def build_lora_roberta_qa() -> nn.Module:
    config = AutoConfig.from_pretrained(MODEL_NAME)
    base = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, config=config)

    lconf = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["query", "value"],
        bias="none",
        task_type="QUESTION_ANS",
    )
    model = get_peft_model(base, lconf)

    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name

    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print(f"Model params: total={total/1e6:.2f}M, trainable={trainable/1e6:.2f}M")
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


class FederatedServer:
    def __init__(self, global_model: nn.Module):
        self.global_model = global_model

    def broadcast(self) -> Dict[str, torch.Tensor]:
        return lora_only_state_dict(self.global_model)

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
                if mode == "A" and "lora_A" not in k:
                    continue
                if mode == "B" and "lora_B" not in k:
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
    clients: List[FederatedClient] = []
    server_lora = server.broadcast()

    for i in range(k):
        device = torch.device(GPU_LIST[i % len(GPU_LIST)])
        local_model = copy.deepcopy(server.global_model)
        load_lora_state_dict_(local_model, server_lora)
        local_model = local_model.to(device=device, dtype=DTYPE)

        params_A = [p for n, p in local_model.named_parameters() if "lora_A" in n and p.requires_grad]
        params_B = [p for n, p in local_model.named_parameters() if "lora_B" in n and p.requires_grad]

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


def _ensure_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


def _answers_from_cell(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return {"text": [""], "answer_start": [0]}
    return {"text": [""], "answer_start": [0]}


def _char_to_token_span(offset_mapping, start_char: int, end_char: int):
    start_tok = None
    end_tok = None

    for i, (s, e) in enumerate(offset_mapping):
        if s <= start_char < e:
            start_tok = i
            break

    for i in range(len(offset_mapping) - 1, -1, -1):
        s, e = offset_mapping[i]
        if s < end_char <= e:
            end_tok = i
            break

    return start_tok, end_tok


class SQuADClientDataset(Dataset):
    def __init__(self, csv_path: Path, tokenizer, max_len: int = 384):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        context = row["context"]
        question = row["question"]

        if not isinstance(context, str):
            context = "" if pd.isna(context) else str(context)
        if not isinstance(question, str):
            question = "" if pd.isna(question) else str(question)

        context = context.strip()
        question = question.strip()

        if len(context) == 0:
            context = " "
        if len(question) == 0:
            question = " "

        answers = _answers_from_cell(row["answers"])

        ans_text = answers["text"][0] if len(answers.get("text", [])) > 0 else ""
        ans_start = int(answers["answer_start"][0]) if len(answers.get("answer_start", [])) > 0 else 0
        ans_end = ans_start + len(ans_text)

        enc = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offset_mapping = enc["offset_mapping"].squeeze(0).tolist()
        sequence_ids = enc.sequence_ids(0)

        ctx_offsets = []
        for off, sid in zip(offset_mapping, sequence_ids):
            if sid != 1:
                ctx_offsets.append((0, 0))
            else:
                ctx_offsets.append(tuple(off))

        start_tok, end_tok = _char_to_token_span(ctx_offsets, ans_start, ans_end)

        if start_tok is None or end_tok is None:
            start_tok = 0
            end_tok = 0

        item = {k: v.squeeze(0) for k, v in enc.items() if k != "offset_mapping"}
        item["start_positions"] = torch.tensor(start_tok, dtype=torch.long)
        item["end_positions"] = torch.tensor(end_tok, dtype=torch.long)
        return item


def make_client_loaders(data_dir: Path, batch_size: int, max_len: int = MAX_LEN) -> List[DataLoader]:
    tok = _ensure_tokenizer()
    loaders: List[DataLoader] = []

    for cid in range(NUM_CLIENTS):
        ds = SQuADClientDataset(data_dir / f"client_{cid}.csv", tok, max_len=max_len)

        if USE_DP:
            sample_rate = batch_size / max(1, len(ds))
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


def compute_client_noise_multipliers(
    data_dir: Path,
    num_clients: int,
    batch_size: int,
    rounds: int,
    local_steps: int,
    target_epsilon: float,
    delta: float,
) -> List[float]:
    from opacus.accountants.utils import get_noise_multiplier

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
def evaluate_squad_em_f1(
    model,
    tokenizer,
    squad_dev_hf_dataset,
    batch_size=16,
    max_length=384,
    doc_stride=128,
    max_answer_length=30,
    n_best_size=20,
    device=None,
):
    metric = evaluate.load("squad")

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    def preprocess(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []

        for i in range(len(tokenized["offset_mapping"])):
            sample_idx = sample_mapping[i]
            ex_id = examples["id"][sample_idx]
            tokenized["example_id"].append(ex_id)

            seq_ids = tokenized.sequence_ids(i)
            tokenized["offset_mapping"][i] = [
                o if (seq_ids[k] == 1) else None
                for k, o in enumerate(tokenized["offset_mapping"][i])
            ]

        return tokenized

    features = squad_dev_hf_dataset.map(
        preprocess,
        batched=True,
        remove_columns=squad_dev_hf_dataset.column_names,
    )

    def collate_fn(batch):
        keys = ["input_ids", "attention_mask"]
        if "token_type_ids" in batch[0]:
            keys.append("token_type_ids")
        out = {k: torch.tensor([b[k] for b in batch], dtype=torch.long) for k in keys}
        out["example_id"] = [b["example_id"] for b in batch]
        out["offset_mapping"] = [b["offset_mapping"] for b in batch]
        return out

    loader = DataLoader(features, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_start_logits, all_end_logits = [], []
    all_example_ids, all_offset_mappings = [], []

    for batch in loader:
        example_ids = batch.pop("example_id")
        offset_mappings = batch.pop("offset_mapping")

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        all_start_logits.append(outputs.start_logits.detach().cpu().numpy())
        all_end_logits.append(outputs.end_logits.detach().cpu().numpy())
        all_example_ids.extend(example_ids)
        all_offset_mappings.extend(offset_mappings)

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    ex_by_id = {ex["id"]: ex for ex in squad_dev_hf_dataset}

    features_per_example = collections.defaultdict(list)
    for i, ex_id in enumerate(all_example_ids):
        features_per_example[ex_id].append(i)

    predictions = []

    for ex_id, feat_indices in features_per_example.items():
        context = ex_by_id[ex_id]["context"]
        best_score = -1e30
        best_text = ""

        for fi in feat_indices:
            start_log = all_start_logits[fi]
            end_log = all_end_logits[fi]
            offsets = all_offset_mappings[fi]

            start_indexes = np.argsort(start_log)[-n_best_size:][::-1]
            end_indexes = np.argsort(end_log)[-n_best_size:][::-1]

            for s in start_indexes:
                for e in end_indexes:
                    if s >= len(offsets) or e >= len(offsets):
                        continue
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s:
                        continue
                    if (e - s + 1) > max_answer_length:
                        continue

                    score = start_log[s] + end_log[e]
                    if score > best_score:
                        start_char = offsets[s][0]
                        end_char = offsets[e][1]
                        best_score = score
                        best_text = context[start_char:end_char]

        predictions.append({"id": ex_id, "prediction_text": best_text})

    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in squad_dev_hf_dataset]

    return metric.compute(predictions=predictions, references=references)


def run_federated_training(args):
    seed_everything(args.seed)

    data_dir = Path(args.data_dir) if hasattr(args, "data_dir") and args.data_dir else DATA_DIR

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = os.path.join(
        args.output_dir,
        f"{args.task}_{args.method}_eps{str(args.epsilon).replace('.', 'p')}_bs{args.batch_size}_lra{str(args.lr_a).replace('.', 'p')}_lrb{str(args.lr_b).replace('.', 'p')}_seed{args.seed}_{timestamp}.csv",
    )

    hf_dev_ds = load_dataset("squad")["validation"]
    tokenizer = _ensure_tokenizer()

    noise_multipliers = compute_client_noise_multipliers(
        data_dir=data_dir,
        num_clients=NUM_CLIENTS,
        batch_size=args.batch_size,
        rounds=ROUNDS,
        local_steps=LOCAL_STEPS,
        target_epsilon=args.epsilon,
        delta=DELTA,
    )

    global_model = build_lora_roberta_qa().to(device=DEVICE, dtype=DTYPE)
    server = FederatedServer(global_model)
    clients = make_clients(NUM_CLIENTS, server)
    client_loaders = make_client_loaders(batch_size=args.batch_size, max_len=MAX_LEN)
    client_loaders = attach_dp_to_clients(clients, client_loaders, noise_multipliers=noise_multipliers)
    rng = random.Random(args.seed)
    best_em = 0.0
    best_f1 = 0.0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "mode", "em", "f1", "best_em", "best_f1"])

    for r in range(1, ROUNDS + 1):
        mode = AB_SCHEDULE[(r - 1) % len(AB_SCHEDULE)]
        print(f"\n[Round {r}] mode={mode}")
        active_ids = rng.sample(range(NUM_CLIENTS), ACTIVE_CLIENTS)
        server_lora = server.broadcast()
        for cid in active_ids:
            load_lora_state_dict_(clients[cid].model, server_lora)

        for cid in active_ids:
            train_local_steps(
                cid=cid,
                client=clients[cid],
                dataloader=client_loaders[cid],
                steps=LOCAL_STEPS,
                mode=mode,
            )

        agg_state = server.aggregate_mean_filtered(
            [clients[cid].lora_state() for cid in active_ids],
            mode=mode,
        )

        metrics = evaluate_squad_em_f1(
            model=server.global_model,
            tokenizer=tokenizer,
            squad_dev_hf_dataset=hf_dev_ds,
            batch_size=16,
            max_length=MAX_LEN,
            doc_stride=DOC_STRIDE,
            max_answer_length=30,
            n_best_size=20,
            device=DEVICE,
        )

        em = metrics["exact_match"]
        f1 = metrics["f1"]
        best_em = max(best_em, em)
        best_f1 = max(best_f1, f1)

        print(
            f"[Round {r:03d}] EM={em:.2f} | F1={f1:.2f} | "
            f"agg_keys={len(agg_state) if agg_state else 0}"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([r, mode, em, f1, best_em, best_f1])

        if USE_DP:
            eps_logs = []

            for cid in active_ids:
                pe = clients[cid].privacy_engine
                if pe is not None:
                    eps_logs.append((cid, pe.accountant.get_epsilon(delta=DELTA)))

            if eps_logs:
                eps_str = ", ".join([f"c{cid}: ε={eps:.2f}" for cid, eps in eps_logs])
                print(f"DP spent (δ={DELTA}): {eps_str}")

    print(f"\nFinal best EM={best_em:.2f}, best F1={best_f1:.2f}")
    print(f"Saved log to {log_path}")


def run(args):
    run_federated_training(args)