"""Training utilities: optimizer, train/eval loops, early stopping,
checkpoint manager, seqeval metrics, and plotting.

Exposes:
    TrainConfig            - dataclass for a single training run
    build_optimizer        - AdamW with differential LRs (BERT vs head+CRF)
    build_scheduler        - linear warmup + linear decay
    train_one_epoch        - one pass over the training loader
    evaluate               - loss + seqeval metrics on a loader
    EarlyStopping          - patience-based early stopping
    CheckpointManager      - saves best (monitored metric) + last
    plot_curves            - loss/F1 curves via matplotlib
    run_training           - end-to-end single-config training
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from .config import (
    BATCH_SIZE,
    BERT_LR,
    BERT_NAME,
    CHECKPOINT_DIR,
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE,
    GRAD_CLIP,
    HEAD_LR,
    IGNORE_INDEX,
    MAX_EPOCHS,
    RUNS_DIR,
    SEED,
    SUBWORD_TAG,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from .data import build_dataloaders
from .model import BertNERModel


@dataclass
class TrainConfig:
    """Configuration for a single training run."""
    name: str = "default"
    interface_type: str = "mlp"          # "mlp" | "bilstm"
    hidden_dim: int = 256
    num_lstm_layers: int = 1
    dropout: float = 0.3
    bert_lr: float = BERT_LR
    head_lr: float = HEAD_LR
    weight_decay: float = WEIGHT_DECAY
    warmup_ratio: float = WARMUP_RATIO
    max_epochs: int = MAX_EPOCHS
    early_stop_patience: int = EARLY_STOP_PATIENCE
    early_stop_min_delta: float = EARLY_STOP_MIN_DELTA
    grad_clip: float = GRAD_CLIP
    batch_size: int = BATCH_SIZE
    max_seq_len: int = 64
    seed: int = SEED
    freeze_bert_layers: bool = True
    load_bert_weights: bool = True
    monitor: str = "val_loss"            # "val_loss" | "val_f1"
    monitor_mode: str = "min"            # "min" for loss, "max" for f1

    def out_dir(self, root: Path = CHECKPOINT_DIR) -> Path:
        return root / self.name


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: BertNERModel, cfg: TrainConfig) -> AdamW:
    """AdamW with two parameter groups: lower LR for fine-tuned BERT
    layers, higher LR for the interface (MLP/Bi-LSTM) and CRF."""
    bert_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("bert"):
            bert_params.append(param)
        else:
            head_params.append(param)
    optimizer = AdamW(
        [
            {"params": bert_params, "lr": cfg.bert_lr, "weight_decay": cfg.weight_decay},
            {"params": head_params, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
        ],
        lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
    )
    return optimizer


def build_scheduler(
    optimizer: AdamW, num_training_steps: int, cfg: TrainConfig
):
    """Linear warmup + linear decay schedule (HF transformers)."""
    from transformers import get_linear_schedule_with_warmup

    warmup_steps = max(1, int(cfg.warmup_ratio * num_training_steps))
    return get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )


def _decode_predictions(
    model: BertNERModel,
    batch: Dict[str, torch.Tensor],
    id2tag: List[str],
    device: torch.device,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Run CRF Viterbi decode and align predictions/golds to word level.

    Skips IGNORE_INDEX positions ([CLS]/[SEP]/padding) and SUBWORD_TAG
    ('X') continuation positions so seqeval sees word-level tag lists.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    token_type_ids = batch.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    emissions = outputs["emissions"]
    mask = outputs["mask"]
    decoded = model.decode(emissions, mask)

    x_id = None
    for i, tag in enumerate(id2tag):
        if tag == SUBWORD_TAG:
            x_id = i
            break

    pred_tags: List[List[str]] = []
    gold_tags: List[List[str]] = []
    for i in range(input_ids.size(0)):
        preds = decoded[i]
        golds = labels[i][: len(preds)].tolist()
        p_seq: List[str] = []
        g_seq: List[str] = []
        for p, g in zip(preds, golds):
            if g == IGNORE_INDEX:
                continue
            if x_id is not None and g == x_id:
                continue
            p_seq.append(id2tag[p])
            g_seq.append(id2tag[g])
        pred_tags.append(p_seq)
        gold_tags.append(g_seq)
    return pred_tags, gold_tags


def train_one_epoch(
    model: BertNERModel,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    cfg: TrainConfig,
    log_interval: int = 25,
) -> float:
    """Single training epoch. Returns the mean CRF NLL loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=token_type_ids,
        )
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
        )
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"    step {step + 1:4d}/{len(loader)} loss={loss.item():.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} {elapsed:.1f}s"
            )

    avg_loss = float(total_loss / max(1, n_batches))
    return avg_loss


@torch.no_grad()
def evaluate(
    model: BertNERModel,
    loader: DataLoader,
    id2tag: List[str],
    device: torch.device,
    compute_seqeval: bool = True,
) -> Dict[str, float]:
    """Evaluate loss and (optionally) seqeval metrics on a loader."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds: List[List[str]] = []
    all_golds: List[List[str]] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device, non_blocking=True)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=token_type_ids,
        )
        total_loss += outputs["loss"].item()
        n_batches += 1

        if compute_seqeval:
            preds, golds = _decode_predictions(model, batch, id2tag, device)
            all_preds.extend(preds)
            all_golds.extend(golds)

    result: Dict[str, float] = {"loss": float(total_loss / max(1, n_batches))}

    if compute_seqeval and all_golds:
        result["accuracy"] = float(accuracy_score(all_golds, all_preds))
        result["precision"] = float(precision_score(all_golds, all_preds, average="macro"))
        result["recall"] = float(recall_score(all_golds, all_preds, average="macro"))
        result["f1"] = float(f1_score(all_golds, all_preds, average="macro"))
    return result


class EarlyStopping:
    """Patience-based early stopping on a monitored metric."""

    def __init__(self, mode: str = "min", patience: int = 4, min_delta: float = 1e-4):
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def _improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def __call__(self, value: float) -> bool:
        improved = self._improved(value)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


class CheckpointManager:
    """Saves 'best' (on monitored metric) and 'last' checkpoints.

    Files in `out_dir`:
        best_model.pt   - state_dict + metrics when monitor improved
        last_model.pt   - state_dict of the most recent epoch
        history.json    - per-epoch metrics
    """

    def __init__(self, out_dir: Path, monitor: str = "val_loss", mode: str = "min"):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.history: List[Dict[str, float]] = []

    def update(self, epoch: int, metrics: Dict[str, float], model: BertNERModel, cfg: TrainConfig) -> bool:
        """Save best/last. Returns True if a new best was recorded."""
        record = {"epoch": epoch, **metrics}
        self.history.append(record)

        last_path = self.out_dir / "last_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
                "config": asdict(cfg),
            },
            last_path,
        )

        value = metrics[self.monitor]
        improved = (self.mode == "min" and value < self.best_value) or (
            self.mode == "max" and value > self.best_value
        )
        if improved:
            self.best_value = value
            best_path = self.out_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "config": asdict(cfg),
                },
                best_path,
            )
            return True
        return False

    def dump_history(self) -> None:
        with (self.out_dir / "history.json").open("w") as f:
            json.dump(self.history, f, indent=2)


def plot_curves(history: List[Dict[str, float]], save_path: Path) -> None:
    """Plot train/val loss and val F1 over epochs, save as PNG."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], "s-", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CRF NLL loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "val_f1" in history[0]:
        axes[1].plot(epochs, [h.get("val_f1", 0) for h in history], "o-", color="green")
        axes[1].set_title("Validation F1 (macro)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def run_training(cfg: TrainConfig, device: Optional[torch.device] = None) -> Dict[str, object]:
    """End-to-end training for one config.

    Returns a summary dict with the final history and best metrics.
    """
    set_seed(cfg.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"Training run: {cfg.name}")
    print(f"  interface={cfg.interface_type} hidden={cfg.hidden_dim} "
          f"dropout={cfg.dropout} bert_lr={cfg.bert_lr} head_lr={cfg.head_lr}")
    print(f"  max_epochs={cfg.max_epochs} batch_size={cfg.batch_size} "
          f"early_stop_patience={cfg.early_stop_patience}")
    print(f"  device={device}  monitor={cfg.monitor} ({cfg.monitor_mode})")
    print("=" * 70)

    print("Loading data...")
    train_loader, dev_loader, test_loader, tag2id, id2tag = build_dataloaders(
        batch_size=cfg.batch_size,
        max_len=cfg.max_seq_len,
        num_workers=0,
        seed=cfg.seed,
    )
    num_tags = len(tag2id)
    print(f"  num_tags={num_tags}  train_batches={len(train_loader)} "
          f"dev_batches={len(dev_loader)} test_batches={len(test_loader)}")

    print("Building model...")
    model = BertNERModel(
        num_tags=num_tags,
        interface_type=cfg.interface_type,
        hidden_dim=cfg.hidden_dim,
        num_lstm_layers=cfg.num_lstm_layers,
        dropout=cfg.dropout,
        bert_name=BERT_NAME,
        freeze_bert_layers=cfg.freeze_bert_layers,
        load_bert_weights=cfg.load_bert_weights,
    ).to(device)

    optimizer = build_optimizer(model, cfg)
    num_training_steps = len(train_loader) * cfg.max_epochs
    scheduler = build_scheduler(optimizer, num_training_steps, cfg)

    early = EarlyStopping(mode=cfg.monitor_mode, patience=cfg.early_stop_patience,
                          min_delta=cfg.early_stop_min_delta)
    ckpt = CheckpointManager(cfg.out_dir(), monitor=cfg.monitor, mode=cfg.monitor_mode)

    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.max_epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.max_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, cfg)

        print(f"  Validating...")
        val_metrics = evaluate(model, dev_loader, id2tag, device, compute_seqeval=True)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics.get("f1", 0.0),
            "val_precision": val_metrics.get("precision", 0.0),
            "val_recall": val_metrics.get("recall", 0.0),
            "val_accuracy": val_metrics.get("accuracy", 0.0),
        }
        history.append(record)
        ckpt.update(epoch, record, model, cfg)
        ckpt.dump_history()

        print(
            f"  train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"val_f1={val_metrics.get('f1', 0.0):.4f}  "
            f"val_acc={val_metrics.get('accuracy', 0.0):.4f}"
        )

        monitored = val_metrics["loss"] if cfg.monitor == "val_loss" else val_metrics.get("f1", 0.0)
        improved = early(monitored)
        print(f"  early_stop counter={early.counter}/{early.patience} "
              f"best={early.best:.4f} improved={'yes' if improved else 'no'}")

        if early.should_stop:
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    plot_curves(history, cfg.out_dir() / "curves.png")

    print("\nLoading best checkpoint and evaluating on test set...")
    best_path = cfg.out_dir() / "best_model.pt"
    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    test_metrics = evaluate(model, test_loader, id2tag, device, compute_seqeval=True)
    print(f"  test_loss={test_metrics['loss']:.4f}  "
          f"test_f1={test_metrics.get('f1', 0.0):.4f}  "
          f"test_acc={test_metrics.get('accuracy', 0.0):.4f}")

    preds_all, golds_all = _collect_golds_preds(model, test_loader, id2tag, device)
    report = classification_report(golds_all, preds_all)
    (cfg.out_dir() / "test_report.txt").write_text(report)
    (cfg.out_dir() / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print("Classification report saved to test_report.txt")
    print(report)

    return {
        "config": asdict(cfg),
        "history": history,
        "test_metrics": test_metrics,
        "best_epoch": (
            min(history, key=lambda h: h[cfg.monitor])
            if cfg.monitor_mode == "min"
            else max(history, key=lambda h: h[cfg.monitor])
        )["epoch"],
    }


@torch.no_grad()
def _collect_golds_preds(model, loader, id2tag, device):
    model.eval()
    preds_all, golds_all = [], []
    for batch in loader:
        p, g = _decode_predictions(model, batch, id2tag, device)
        preds_all.extend(p)
        golds_all.extend(g)
    return preds_all, golds_all