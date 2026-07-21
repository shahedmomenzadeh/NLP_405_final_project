"""Phase 3 entry point: Training & Hyperparameter Tuning.

Usage:
    # Single default run (MLP, hidden=256, dropout=0.3)
    python phase3_train.py --run default

    # Tune only one family
    python phase3_train.py --tune mlp
    python phase3_train.py --tune bilstm
    python phase3_train.py --tune all      # both families

    # Single custom config
    python phase3_train.py --run mlp_d03_h128 --interface mlp --hidden 128 --dropout 0.3
"""
from __future__ import annotations

import argparse
import os

# HF cache config MUST run before importing transformers/huggingface_hub.
HF_CACHE_DIR = os.path.abspath("./hf_cache")
try:
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
except FileExistsError:
    pass
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ.setdefault("HF_HUB_CACHE", HF_CACHE_DIR)

import json
import sys

import torch

from src.config import CHECKPOINT_DIR, RUNS_DIR
from src.training import TrainConfig, run_training


TUNING_CONFIGS = {
    "mlp": [
        TrainConfig(name="mlp_h128_d01", interface_type="mlp", hidden_dim=128, dropout=0.1),
        TrainConfig(name="mlp_h256_d03", interface_type="mlp", hidden_dim=256, dropout=0.3),
        TrainConfig(name="mlp_h256_d05", interface_type="mlp", hidden_dim=256, dropout=0.5),
        TrainConfig(name="mlp_h512_d03", interface_type="mlp", hidden_dim=512, dropout=0.3),
    ],
    "bilstm": [
        TrainConfig(name="bilstm_h128_l1", interface_type="bilstm", hidden_dim=128, num_lstm_layers=1, dropout=0.3),
        TrainConfig(name="bilstm_h256_l1", interface_type="bilstm", hidden_dim=256, num_lstm_layers=1, dropout=0.3),
        TrainConfig(name="bilstm_h256_l2", interface_type="bilstm", hidden_dim=256, num_lstm_layers=2, dropout=0.3),
        TrainConfig(name="bilstm_h256_l2_d05", interface_type="bilstm", hidden_dim=256, num_lstm_layers=2, dropout=0.5),
    ],
}


def write_tuning_summary(results, out_path):
    """Write a markdown comparison table of all tuning runs."""
    lines = [
        "# Hyperparameter Tuning Summary",
        "",
        "| Run | Interface | Hidden | Layers | Dropout | Best Epoch | Val Loss | Val F1 | Test Loss | Test F1 | Test Acc |",
        "|-----|----------|--------|--------|---------|------------|----------|--------|-----------|---------|----------|",
    ]
    for r in results:
        cfg = r["config"]
        best = next(h for h in r["history"] if h["epoch"] == r["best_epoch"])
        tm = r["test_metrics"]
        lines.append(
            f"| {cfg['name']} | {cfg['interface_type']} | {cfg['hidden_dim']} | "
            f"{cfg.get('num_lstm_layers', '-')} | {cfg['dropout']} | "
            f"{r['best_epoch']} | {best['val_loss']:.4f} | {best['val_f1']:.4f} | "
            f"{tm['loss']:.4f} | {tm.get('f1', 0):.4f} | {tm.get('accuracy', 0):.4f} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"\nTuning summary written to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 training")
    parser.add_argument("--run", type=str, default=None,
                        help="Run a single named config (use overrides below).")
    parser.add_argument("--tune", type=str, default=None, choices=["mlp", "bilstm", "all"],
                        help="Run a hyperparameter tuning sweep.")
    parser.add_argument("--interface", type=str, default="mlp", choices=["mlp", "bilstm"])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_weights", action="store_true",
                        help="Use random BERT weights (smoke test, no download).")
    args = parser.parse_args()

    if args.run is None and args.tune is None:
        args.run = "default"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_path = RUNS_DIR / "tuning_summary.md"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if args.run is not None:
        cfg = TrainConfig(
            name=args.run,
            interface_type=args.interface,
            hidden_dim=args.hidden,
            num_lstm_layers=args.lstm_layers,
            dropout=args.dropout,
        )
        if args.epochs is not None:
            cfg.max_epochs = args.epochs
        if args.batch is not None:
            cfg.batch_size = args.batch
        if args.seed is not None:
            cfg.seed = args.seed
        if args.no_weights:
            cfg.load_bert_weights = False
        run_training(cfg, device=device)
        return 0

    if args.tune is not None:
        if args.tune == "all":
            configs = TUNING_CONFIGS["mlp"] + TUNING_CONFIGS["bilstm"]
        else:
            configs = TUNING_CONFIGS[args.tune]

        results = []
        for i, cfg in enumerate(configs, 1):
            print(f"\n{'#'*70}\n# TUNING RUN {i}/{len(configs)}: {cfg.name}\n{'#'*70}")
            if args.no_weights:
                cfg.load_bert_weights = False
            result = run_training(cfg, device=device)
            results.append(result)
            
            out_file = RUNS_DIR / "tuning_results.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with out_file.open("w") as f:
                json.dump(results, f, indent=2)

        write_tuning_summary(results, results_path)
        print("\nAll tuning runs complete.")
        return 0


if __name__ == "__main__":
    sys.exit(main())