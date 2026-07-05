"""Data loading, tokenization, alignment, and PyTorch DataLoaders.

Exposes:
    load_atis_split(split_dir)         -> list[dict]  (raw samples)
    build_tag_vocab(slot_label_path)   -> (tag2id, id2tag)
    encode_samples(samples, tag2id, tokenizer, max_len)
                                        -> list[dict]  (tokenized + label-aligned)
    ATISDataset                          -> torch Dataset
    make_collate_fn(tokenizer, pad_label_id)
                                        -> callable
    build_dataloaders(batch_size=...)   -> (train_loader, dev_loader, test_loader,
                                              tag2id, id2tag)
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from transformers import BertTokenizerFast

from .config import (
    BERT_NAME,
    FORMAT2_ROOT,
    IGNORE_INDEX,
    MAX_SEQ_LEN,
    PAD_TAG,
    PAD_TAG_ID,
    SEED,
    SUBWORD_TAG,
)

_NON_ASCII = re.compile(r"[^\x00-\x7F]")


def is_english(text: str) -> bool:
    """Heuristic English filter. Returns True if the string contains
    no non-ASCII characters. ATIS is already English; this is a safeguard
    for the multilingual MultiATIS-style datasets referenced in the guide."""
    return _NON_ASCII.search(text) is None


def load_atis_split(split_dir: Path) -> List[Dict]:
    """Load a Format2 ATIS split.

    Each line of seq.in / seq.out / label corresponds to one sentence.
    Returns a list of {"tokens", "tags", "intent"} dicts. Samples whose
    sentence contains non-ASCII characters are dropped (English filter).
    No stop-word removal or text cleaning is applied.
    """
    seq_in_path = split_dir / "seq.in"
    seq_out_path = split_dir / "seq.out"
    label_path = split_dir / "label"

    with seq_in_path.open(encoding="utf-8") as f:
        in_lines = [line.rstrip("\n") for line in f]
    with seq_out_path.open(encoding="utf-8") as f:
        out_lines = [line.rstrip("\n") for line in f]
    with label_path.open(encoding="utf-8") as f:
        label_lines = [line.rstrip("\n") for line in f]

    if not (len(in_lines) == len(out_lines) == len(label_lines)):
        raise ValueError(
            f"Length mismatch in {split_dir}: "
            f"seq.in={len(in_lines)}, seq.out={len(out_lines)}, "
            f"label={len(label_lines)}"
        )

    samples: List[Dict] = []
    dropped = 0
    for tokens_line, tags_line, intent in zip(in_lines, out_lines, label_lines):
        tokens = tokens_line.split() if tokens_line else []
        tags = tags_line.split() if tags_line else []
        if len(tokens) != len(tags):
            raise ValueError(
                f"Token/tag count mismatch in {split_dir}: "
                f"{tokens_line!r} vs {tags_line!r}"
            )
        sentence = " ".join(tokens)
        if not is_english(sentence):
            dropped += 1
            continue
        samples.append({"tokens": tokens, "tags": tags, "intent": intent})

    print(
        f"  Loaded {split_dir.name}: {len(samples)} samples "
        f"(dropped {dropped} non-English)"
    )
    return samples


def build_tag_vocab(slot_label_path: Path) -> Tuple[Dict[str, int], List[str]]:
    """Build tag2id / id2tag mappings.

    PAD=0 (reserved), then the original slot_label.txt tags in the order
    they appear (which already starts with PAD/UNK/<sos>/<eos>/O), then
    a special 'X' tag for subword continuations.
    """
    with slot_label_path.open(encoding="utf-8") as f:
        tags = [line.strip() for line in f if line.strip()]

    if tags[0] != PAD_TAG:
        tags = [PAD_TAG] + tags

    if SUBWORD_TAG not in tags:
        tags.append(SUBWORD_TAG)

    tag2id: Dict[str, int] = {t: i for i, t in enumerate(tags)}
    id2tag: List[str] = list(tags)

    assert tag2id[PAD_TAG] == PAD_TAG_ID, "PAD must be at index 0"
    return tag2id, id2tag


def encode_samples(
    samples: Sequence[Dict],
    tag2id: Dict[str, int],
    tokenizer: BertTokenizerFast,
    max_len: int = MAX_SEQ_LEN,
) -> List[Dict]:
    """Tokenize each sample with BertTokenizerFast and align the BIO
    labels to the resulting subword tokens.

    Alignment rules:
      * First subword of a word   -> original tag id
      * Remaining subwords        -> 'X' id
      * Special tokens ([CLS],
        [SEP]) and padding       -> IGNORE_INDEX (-100)
    """
    encodings: List[Dict] = []
    oov_count = 0
    for sample in samples:
        tokens = sample["tokens"]
        tags = sample["tags"]

        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
            padding=False,
        )

        word_ids = enc.word_ids()
        aligned_labels: List[int] = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(IGNORE_INDEX)
            elif word_idx != previous_word_idx:
                tag = tags[word_idx]
                if tag not in tag2id:
                    oov_count += 1
                    tag = "O"
                aligned_labels.append(tag2id[tag])
            else:
                aligned_labels.append(tag2id[SUBWORD_TAG])
            previous_word_idx = word_idx

        encodings.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "token_type_ids": enc.get("token_type_ids", [0] * len(enc["input_ids"])),
                "labels": aligned_labels,
                "word_ids": word_ids,
            }
        )

    if oov_count:
        print(f"  Warning: {oov_count} OOV tags mapped to 'O'")
    return encodings


class ATISDataset(Dataset):
    """Wraps a list of tokenized encodings into a torch Dataset.

    Each item is a dict of Long tensors: input_ids, attention_mask,
    token_type_ids, labels.
    """

    def __init__(self, encodings: Sequence[Dict]):
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        e = self.encodings[idx]
        return {
            "input_ids": torch.tensor(e["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(e["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(e["token_type_ids"], dtype=torch.long),
            "labels": torch.tensor(e["labels"], dtype=torch.long),
        }


def make_collate_fn(tokenizer: BertTokenizerFast, pad_label_id: int = IGNORE_INDEX):
    """Build a collate_fn that pads input_ids / attention_mask /
    token_type_ids via tokenizer.pad and pads labels with pad_label_id
    so the loss can ignore padded positions.
    """
    pad_token_id = tokenizer.pad_token_id

    def collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(b["input_ids"].size(0) for b in batch)
        out = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
            "labels": [],
        }
        for b in batch:
            n = b["input_ids"].size(0)
            pad_n = max_len - n
            out["input_ids"].append(
                torch.cat([b["input_ids"], torch.full((pad_n,), pad_token_id, dtype=torch.long)])
            )
            out["attention_mask"].append(
                torch.cat([b["attention_mask"], torch.zeros(pad_n, dtype=torch.long)])
            )
            out["token_type_ids"].append(
                torch.cat([b["token_type_ids"], torch.zeros(pad_n, dtype=torch.long)])
            )
            out["labels"].append(
                torch.cat([b["labels"], torch.full((pad_n,), pad_label_id, dtype=torch.long)])
            )
        return {k: torch.stack(v, dim=0) for k, v in out.items()}

    return collate_fn


def build_dataloaders(
    batch_size: int = 32,
    max_len: int = MAX_SEQ_LEN,
    num_workers: int = 0,
    seed: int = SEED,
):
    """End-to-end pipeline: load data, build vocab, tokenize, wrap,
    return DataLoaders plus tag vocab.

    Returns: (train_loader, dev_loader, test_loader, tag2id, id2tag)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    slot_label_path = FORMAT2_ROOT / "slot_label.txt"
    tag2id, id2tag = build_tag_vocab(slot_label_path)
    print(f"  Tag vocab size: {len(tag2id)} (PAD=0, X={tag2id[SUBWORD_TAG]})")

    print("Loading splits...")
    train_samples = load_atis_split(FORMAT2_ROOT / "train")
    dev_samples = load_atis_split(FORMAT2_ROOT / "dev")
    test_samples = load_atis_split(FORMAT2_ROOT / "test")

    print(f"Loading tokenizer: {BERT_NAME}")
    tokenizer = BertTokenizerFast.from_pretrained(BERT_NAME)

    print("Tokenizing & aligning labels...")
    train_enc = encode_samples(train_samples, tag2id, tokenizer, max_len=max_len)
    dev_enc = encode_samples(dev_samples, tag2id, tokenizer, max_len=max_len)
    test_enc = encode_samples(test_samples, tag2id, tokenizer, max_len=max_len)

    collate_fn = make_collate_fn(tokenizer, pad_label_id=IGNORE_INDEX)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        ATISDataset(train_enc),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        generator=g,
    )
    dev_loader = DataLoader(
        ATISDataset(dev_enc),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        ATISDataset(test_enc),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, dev_loader, test_loader, tag2id, id2tag


def save_tag_vocab(tag2id: Dict[str, int], id2tag: List[str], out_dir: Path) -> None:
    """Persist tag vocabulary to disk for later phases."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "tag2id.json").open("w", encoding="utf-8") as f:
        json.dump(tag2id, f, indent=2, ensure_ascii=False)
    with (out_dir / "id2tag.json").open("w", encoding="utf-8") as f:
        json.dump(id2tag, f, indent=2, ensure_ascii=False)


def summarize_tag_distribution(samples: Sequence[Dict], top_k: int = 10) -> None:
    """Print the top-k most common tags across a sample list."""
    counter = Counter()
    for s in samples:
        counter.update(s["tags"])
    print(f"  Total tag tokens: {sum(counter.values())}")
    print(f"  Unique tags seen: {len(counter)}")
    print(f"  Top-{top_k} tags:")
    for tag, n in counter.most_common(top_k):
        print(f"    {tag:30s} {n}")
