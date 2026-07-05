"""Phase 2 entry point: Model Architecture verification.

Tests the BERT + MLP/BiLSTM + CRF architecture:
    * Loads data from Phase 1
    * Instantiates models with both interface types
    * Verifies selective freezing of BERT layers
    * Runs forward passes and checks shapes
    * Tests CRF decode (Viterbi)
    * Prints parameter counts (trainable vs frozen)
"""
from __future__ import annotations

import os

# HF cache config MUST run before importing transformers/huggingface_hub.
HF_CACHE_DIR = os.path.abspath("./hf_cache")
try:
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
except FileExistsError:
    pass
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ.setdefault("HF_HUB_CACHE", HF_CACHE_DIR)

import sys

import torch

from src.config import BATCH_SIZE, BERT_NAME, MAX_SEQ_LEN, SEED
from src.data import build_dataloaders
from src.model import BertNERModel

def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> dict:
    """Count total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100.0 * trainable / total if total > 0 else 0.0,
    }


def print_frozen_layers(model: BertNERModel) -> None:
    """Print which BERT layers are frozen vs trainable."""
    print("  BERT layer freezing status:")
    print(f"    embeddings: {'FROZEN' if not any(p.requires_grad for p in model.bert.embeddings.parameters()) else 'trainable'}")
    for i in range(12):
        layer = model.bert.encoder.layer[i]
        is_frozen = not any(p.requires_grad for p in layer.parameters())
        status = "FROZEN" if is_frozen else "trainable"
        print(f"    encoder.layer[{i:2d}]: {status}")
    print()


def test_model_variant(
    variant_name: str,
    model: BertNERModel,
    batch: dict,
    device: torch.device,
    num_tags: int,
) -> None:
    """Test a model variant with a forward pass and decode."""
    print(f"\n{'='*70}")
    print(f"Testing {variant_name}")
    print(f"{'='*70}")

    model.to(device)
    model.eval()

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    print(f"  Input shapes:")
    print(f"    input_ids:      {input_ids.shape}")
    print(f"    attention_mask: {attention_mask.shape}")
    print(f"    labels:         {labels.shape}")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    loss = outputs["loss"]
    emissions = outputs["emissions"]
    mask = outputs["mask"]

    print(f"\n  Forward pass outputs:")
    print(f"    loss:     {loss.item():.4f} (shape: {loss.shape})")
    print(f"    emissions: {emissions.shape} (batch_size, seq_len, num_tags)")
    print(f"    mask:     {mask.shape}")

    decoded = model.decode(emissions, mask)
    print(f"\n  CRF decode (Viterbi):")
    print(f"    Number of sequences decoded: {len(decoded)}")
    print(f"    First sequence length: {len(decoded[0])}")
    print(f"    First 10 tags in first sequence: {decoded[0][:10]}")

    params = count_parameters(model)
    print(f"\n  Parameter counts:")
    print(f"    Total:     {params['total']:>10,}")
    print(f"    Trainable: {params['trainable']:>10,} ({params['trainable_pct']:.2f}%)")
    print(f"    Frozen:    {params['frozen']:>10,}")


def main() -> int:
    set_seed(SEED)

    print("=" * 70)
    print("PHASE 2: Model Architecture")
    print("=" * 70)
    print(f"  BERT_NAME: {BERT_NAME}")
    print(f"  Device:    {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    print("Loading data from Phase 1...")
    train_loader, dev_loader, test_loader, tag2id, id2tag = build_dataloaders(
        batch_size=BATCH_SIZE,
        max_len=MAX_SEQ_LEN,
        num_workers=0,
        seed=SEED,
    )
    num_tags = len(tag2id)
    print(f"  Number of tags: {num_tags}")
    print()

    batch = next(iter(train_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("Instantiating MLP model (random BERT weights, no download)...")
    print("=" * 70)
    model_mlp = BertNERModel(
        num_tags=num_tags,
        interface_type="mlp",
        hidden_dim=256,
        dropout=0.3,
        bert_name=BERT_NAME,
        freeze_bert_layers=True,
        load_bert_weights=False,
    )
    print_frozen_layers(model_mlp)
    test_model_variant("MLP Model", model_mlp, batch, device, num_tags)

    print("\n" + "=" * 70)
    print("Instantiating Bi-LSTM model (random BERT weights, no download)...")
    print("=" * 70)
    model_bilstm = BertNERModel(
        num_tags=num_tags,
        interface_type="bilstm",
        hidden_dim=256,
        num_lstm_layers=1,
        dropout=0.3,
        bert_name=BERT_NAME,
        freeze_bert_layers=True,
        load_bert_weights=False,
    )
    print_frozen_layers(model_bilstm)
    test_model_variant("Bi-LSTM Model", model_bilstm, batch, device, num_tags)

    print("\n" + "=" * 70)
    print("Phase 2 complete.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
