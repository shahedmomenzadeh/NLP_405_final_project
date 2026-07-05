"""Phase 1 entry point: Environment Setup & Data Preparation.

Loads the ATIS dataset, builds the BIO tag vocabulary, tokenizes with
BERT, aligns labels to subword tokens, wraps in a PyTorch Dataset, and
exposes Train/Validation/Test DataLoaders.

Sanity checks at the end:
    * dataset sizes
    * tag distribution (top tags)
    * one aligned example (word -> subwords -> labels)
    * one batch pulled from the train loader, with shapes
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    BERT_NAME,
    FORMAT2_ROOT,
    IGNORE_INDEX,
    MAX_SEQ_LEN,
    PAD_TAG,
    SEED,
    SUBWORD_TAG,
)
from src.data import (
    ATISDataset,
    build_dataloaders,
    build_tag_vocab,
    encode_samples,
    load_atis_split,
    make_collate_fn,
    save_tag_vocab,
    summarize_tag_distribution,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_library_versions() -> None:
    import transformers
    import seqeval
    import matplotlib
    try:
        from importlib.metadata import version
        seqeval_ver = version("seqeval")
    except Exception:
        seqeval_ver = "unknown"

    print("=" * 70)
    print("PHASE 1: Environment Setup & Data Preparation")
    print("=" * 70)
    print(f"  torch          : {torch.__version__}  (cuda available: {torch.cuda.is_available()})")
    print(f"  transformers   : {transformers.__version__}")
    print(f"  seqeval        : {seqeval_ver}")
    print(f"  matplotlib     : {matplotlib.__version__}")
    try:
        import crf
        from crf import CRF
        print(f"  pytorch-crf    : {crf.__file__.split('site-packages/')[-1]}")
    except ImportError:
        try:
            import pytorch_crf
            from pytorch_crf import CRF
            print(f"  pytorch-crf    : {pytorch_crf.__file__.split('site-packages/')[-1]}")
        except ImportError:
            print("  pytorch-crf    : NOT FOUND")
    print(f"  BERT_NAME      : {BERT_NAME}")
    print(f"  MAX_SEQ_LEN    : {MAX_SEQ_LEN}")
    print(f"  BATCH_SIZE     : {BATCH_SIZE}")
    print(f"  SEED           : {SEED}")
    print()


def demo_alignment(samples, tag2id, id2tag, tokenizer, max_len: int) -> None:
    """Show one full example: tokens, subwords, aligned labels, with
    the subword-to-word mapping."""
    sample = samples[0]
    tokens = sample["tokens"]
    tags = sample["tags"]

    enc = encode_samples([sample], tag2id, tokenizer, max_len=max_len)[0]

    print("Example alignment (first training sample):")
    print(f"  Original tokens : {tokens}")
    print(f"  Original tags   : {tags}")
    subwords = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    print(f"  Subword tokens  : {subwords}")
    label_strs = ["-" if i == IGNORE_INDEX else id2tag[i] for i in enc["labels"]]
    print(f"  Aligned labels  : {label_strs}")
    print(f"  word_ids        : {enc['word_ids']}")
    print()


def demo_batch(batch, tokenizer, id2tag) -> None:
    """Print a decoded batch and tensor shapes."""
    print("First batch from train_loader:")
    for k, v in batch.items():
        print(f"  {k:15s} shape={tuple(v.shape)} dtype={v.dtype}")
    print()

    print("Decoded first sentence in the batch:")
    ids = batch["input_ids"][0].tolist()
    mask = batch["attention_mask"][0].tolist()
    labels = batch["labels"][0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(ids)
    valid_pairs = [
        (tok, id2tag[lbl])
        for tok, m, lbl in zip(tokens, mask, labels)
        if m == 1 and lbl != IGNORE_INDEX
    ]
    for tok, lbl in valid_pairs[:20]:
        print(f"    {tok:20s} -> {lbl}")
    print()


def main() -> int:
    set_seed(SEED)
    print_library_versions()

    print("Building tag vocabulary...")
    tag2id, id2tag = build_tag_vocab(FORMAT2_ROOT / "slot_label.txt")
    print(f"  Tag vocab size: {len(tag2id)}")
    print(f"  PAD id: {tag2id[PAD_TAG]}    X id: {tag2id[SUBWORD_TAG]}")
    print(f"  First 8 tags : {id2tag[:8]}")
    print(f"  Last  3 tags : {id2tag[-3:]}")
    print()

    print("Loading raw splits...")
    train_samples = load_atis_split(FORMAT2_ROOT / "train")
    dev_samples = load_atis_split(FORMAT2_ROOT / "dev")
    test_samples = load_atis_split(FORMAT2_ROOT / "test")
    print()

    print("Tag distribution (train):")
    summarize_tag_distribution(train_samples, top_k=10)
    print()

    print(f"Loading tokenizer: {BERT_NAME}")
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(BERT_NAME)
    print(f"  vocab size: {tokenizer.vocab_size}")
    print(f"  pad_token_id: {tokenizer.pad_token_id} ({tokenizer.pad_token!r})")
    print()

    demo_alignment(train_samples, tag2id, id2tag, tokenizer, MAX_SEQ_LEN)

    print("Wrapping into DataLoaders...")
    train_loader, dev_loader, test_loader, tag2id, id2tag = build_dataloaders(
        batch_size=BATCH_SIZE,
        max_len=MAX_SEQ_LEN,
        num_workers=0,
        seed=SEED,
    )
    print(f"  train batches : {len(train_loader)}")
    print(f"  dev   batches : {len(dev_loader)}")
    print(f"  test  batches : {len(test_loader)}")
    print()

    batch = next(iter(train_loader))
    demo_batch(batch, tokenizer, id2tag)

    print("Saving tag vocabulary to artifacts/...")
    save_tag_vocab(tag2id, id2tag, ARTIFACTS_DIR)
    print(f"  -> {ARTIFACTS_DIR / 'tag2id.json'}")
    print(f"  -> {ARTIFACTS_DIR / 'id2tag.json'}")
    print()

    print("=" * 70)
    print("Phase 1 complete.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
