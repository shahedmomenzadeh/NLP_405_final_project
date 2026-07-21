"""Phase 5: LLM Experimentation (Part 2).

Steps:
  1. Stratified sampling: pick 20 diverse representative test samples.
  2. BERT baseline: run the best fine-tuned model on those 20 samples.
  3. For every LLM in LLM_MODELS (configurable):
       a. Zero-shot: one prompt per sample, no conversational memory.
       b. Few-shot: same, with k labeled training-set examples.
       c. Compute seqeval F1.
       d. Save per-sample predictions and per-model metrics.
  4. Aggregate everything into a single summary.json + bar chart.

Usage:
    # Run all LLMs in src.config.LLM_MODELS
    python phase5_llm.py

    # Run a single named LLM (model must be loaded in LM Studio)
    python phase5_llm.py --model qwen3-vl-4b-instruct

    # Run a custom subset
    python phase5_llm.py --models qwen3-vl-4b-instruct,google/gemma-3-4b-it

    # BERT only / LLM only
    python phase5_llm.py --skip_llm
    python phase5_llm.py --skip_bert
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from seqeval.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    BERT_NAME,
    CHECKPOINT_DIR,
    FEWSHOT_K,
    FORMAT2_ROOT,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODELS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    MAX_SEQ_LEN,
    PHASE5_DIR,
    SEED,
    SUBSET_SIZE,
)
from src.data import (
    ATISDataset,
    build_tag_vocab,
    encode_samples,
    load_atis_split,
    make_collate_fn,
)
from src.llm import LMStudioClient, parse_bio_list, sanitize_model_name
from src.model import BertNERModel
from src.prompts import (
    SYSTEM_PROMPT,
    build_few_shot_prompt,
    build_zero_shot_prompt,
)
from src.stratify import stratified_sample, summarise_subset


def load_data():
    tag2id, id2tag = build_tag_vocab(FORMAT2_ROOT / "slot_label.txt")
    train = load_atis_split(FORMAT2_ROOT / "train")
    dev = load_atis_split(FORMAT2_ROOT / "dev")
    test = load_atis_split(FORMAT2_ROOT / "test")
    return train, dev, test, tag2id, id2tag


def pick_few_shot_examples(train: List[dict], k: int, seed: int) -> List[dict]:
    """Pick k diverse training examples for few-shot prompting.

    Picks one short, one medium, and one long example to teach the
    LLM the format across sentence lengths.
    """
    from src.stratify import length_bin
    rng = random.Random(seed)
    by_bin = {"short": [], "medium": [], "long": []}
    for ex in train:
        by_bin[length_bin(len(ex["tokens"]))].append(ex)
    chosen = []
    for b in ("short", "medium", "long"):
        if not by_bin[b]:
            continue
        rng.shuffle(by_bin[b])
        chosen.append(by_bin[b][0])
    while len(chosen) < k:
        rng.shuffle(train)
        for ex in train:
            if ex not in chosen:
                chosen.append(ex)
                break
    return chosen[:k]


def run_bert_on_subset(
    subset: List[dict], tag2id: dict, id2tag: List[str], ckpt_path: Path, device
) -> List[dict]:
    """Run the fine-tuned BERT model on each subset sample."""
    print(f"\nLoading BERT model from {ckpt_path}...")
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(BERT_NAME)

    num_tags = len(tag2id)
    model = BertNERModel(
        num_tags=num_tags,
        interface_type="bilstm",
        hidden_dim=256,
        dropout=0.3,
        bert_name=BERT_NAME,
        freeze_bert_layers=True,
        load_bert_weights=True,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    encodings = encode_samples(subset, tag2id, tokenizer, max_len=MAX_SEQ_LEN)
    collate_fn = make_collate_fn(tokenizer, pad_label_id=-100)

    from torch.utils.data import DataLoader
    loader = DataLoader(ATISDataset(encodings), batch_size=8, shuffle=False, collate_fn=collate_fn)

    id2tag_safe = list(id2tag)
    from src.config import IGNORE_INDEX, SUBWORD_TAG
    x_id = id2tag_safe.index(SUBWORD_TAG) if SUBWORD_TAG in id2tag_safe else None

    all_pred_strs: List[List[str]] = []
    all_gold_strs: List[List[str]] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        decoded = model.decode(out["emissions"], out["mask"])
        labels = batch["labels"]
        for i in range(input_ids.size(0)):
            preds = decoded[i]
            golds = labels[i][: len(preds)].tolist()
            p_seq, g_seq = [], []
            for p, g in zip(preds, golds):
                if g == IGNORE_INDEX:
                    continue
                if x_id is not None and g == x_id:
                    continue
                p_seq.append(id2tag_safe[p])
                g_seq.append(id2tag_safe[g])
            all_pred_strs.append(p_seq)
            all_gold_strs.append(g_seq)

    results = []
    for sample, pred, gold in zip(subset, all_pred_strs, all_gold_strs):
        results.append({
            "sentence": " ".join(sample["tokens"]),
            "tokens": sample["tokens"],
            "gold_tags": gold,
            "pred_tags": pred,
        })
    return results


def call_llm_on_subset(
    subset: List[dict],
    client: LMStudioClient,
    all_tags: List[str],
    mode: str,
    few_shot_examples: List[dict] = None,
) -> List[dict]:
    """Call the LLM on every sample. One prompt per sample, no memory."""
    results = []
    for i, sample in enumerate(subset, 1):
        sentence = " ".join(sample["tokens"])
        if mode == "zero":
            user_prompt = build_zero_shot_prompt(sentence, all_tags)
        elif mode == "few":
            user_prompt = build_few_shot_prompt(sentence, all_tags, few_shot_examples)
        else:
            raise ValueError(f"unknown mode: {mode}")

        t0 = time.time()
        try:
            raw = client.chat(SYSTEM_PROMPT, user_prompt)
            pred = parse_bio_list(raw, n_tokens=len(sample["tokens"]))
        except Exception as e:
            print(f"  LLM error on sample {i}: {e}")
            raw = f"ERROR: {e}"
            pred = ["O"] * len(sample["tokens"])
        dt = time.time() - t0

        results.append({
            "sentence": sentence,
            "tokens": sample["tokens"],
            "gold_tags": list(sample["tags"]),
            "pred_tags": pred,
            "raw_output": raw,
            "latency_s": round(dt, 2),
        })
        print(f"  [{mode:>3}] {i:2d}/{len(subset)}  {dt:5.1f}s  "
              f"tokens={len(sample['tokens']):2d}  pred[0:5]={pred[:5]}")
    return results


def compute_metrics(results: List[dict]) -> dict:
    golds = [r["gold_tags"] for r in results]
    preds = [r["pred_tags"] for r in results]
    try:
        return {
            "accuracy": float(accuracy_score(golds, preds)),
            "precision_macro": float(precision_score(golds, preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(golds, preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(golds, preds, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(golds, preds, average="micro", zero_division=0)),
            "n_samples": len(results),
        }
    except Exception as e:
        print(f"  metric error: {e}")
        return {"error": str(e), "n_samples": len(results)}


def make_bar_chart(
    bert_m: dict,
    per_model_results: Dict[str, Dict[str, dict]],
    out_path: Path,
    n_samples: int,
) -> None:
    """One bar per method, grouped as: BERT, then per-LLM zero+few."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels: List[str] = ["BERT"]
    f1s: List[float] = [bert_m.get("f1_macro", 0)]
    accs: List[float] = [bert_m.get("accuracy", 0)]
    colors: List[str] = ["tab:blue"]

    llm_colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    for i, (model_name, res) in enumerate(per_model_results.items()):
        c = llm_colors[i % len(llm_colors)]
        z = res.get("zero_metrics", {})
        f = res.get("few_metrics", {})
        labels.append(f"{model_name}\nzero-shot")
        f1s.append(z.get("f1_macro", 0))
        accs.append(z.get("accuracy", 0))
        colors.append(c)
        labels.append(f"{model_name}\nfew-shot")
        f1s.append(f.get("f1_macro", 0))
        accs.append(f.get("accuracy", 0))
        colors.append(c)

    fig, ax = plt.subplots(figsize=(max(9, 2.0 * len(labels)), 5))
    x = range(len(labels))
    bars = ax.bar(x, f1s, color=colors, alpha=0.85)
    for bar, f1, acc in zip(bars, f1s, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"F1={f1:.3f}\nAcc={acc:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("F1 (macro)")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Slot filling on {n_samples} stratified test samples")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_summary(
    out_dir: Path,
    subset: List[dict],
    bert_metrics: dict,
    bert_results: List[dict],
    per_model_results: Dict[str, Dict[str, dict]],
    few_shot_examples: List[dict],
    models_to_run: List[dict],
    subset_indices: List[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "subset.json").open("w") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)
    with (out_dir / "bert_predictions.json").open("w") as f:
        json.dump(bert_results, f, indent=2, ensure_ascii=False)
    with (out_dir / "few_shot_examples.json").open("w") as f:
        json.dump(few_shot_examples, f, indent=2, ensure_ascii=False)

    # Per-model files
    metrics_table: Dict[str, dict] = {"bert": bert_metrics}
    for model_name, res in per_model_results.items():
        safe = sanitize_model_name(model_name)
        with (out_dir / f"llm_{safe}_zero.json").open("w") as f:
            json.dump(res.get("zero_results", []), f, indent=2, ensure_ascii=False)
        with (out_dir / f"llm_{safe}_few.json").open("w") as f:
            json.dump(res.get("few_results", []), f, indent=2, ensure_ascii=False)
        metrics_table[f"{model_name}::zero"] = res.get("zero_metrics", {})
        metrics_table[f"{model_name}::few"] = res.get("few_metrics", {})

    summary = {
        "subset": summarise_subset(subset, list(range(len(subset)))),
        "metrics": metrics_table,
        "config": {
            "models_requested": [m["name"] for m in models_to_run],
            "n_samples": len(subset),
            "k_fewshot": len(few_shot_examples),
        },
        "subset_indices_in_test": subset_indices,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote outputs to {out_dir}/")


def select_models(args) -> List[dict]:
    """Resolve which LLMs to run based on CLI flags / defaults."""
    if args.model:
        return [{"name": args.model, "base_url": LLM_BASE_URL}]
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        available = {m["name"]: m for m in LLM_MODELS}
        chosen = []
        for name in wanted:
            if name in available:
                chosen.append(available[name])
            else:
                # Allow ad-hoc names with the default base URL
                chosen.append({"name": name, "base_url": LLM_BASE_URL})
        return chosen
    return list(LLM_MODELS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=SUBSET_SIZE)
    parser.add_argument("--k_fewshot", type=int, default=FEWSHOT_K)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ckpt", type=str,
                        default=str(CHECKPOINT_DIR / "bilstm_h256_l1" / "best_model.pt"))
    parser.add_argument("--skip_bert", action="store_true")
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--out", type=str, default=str(PHASE5_DIR))
    parser.add_argument("--model", type=str, default=None,
                        help="Run a single LLM by name (overrides LLM_MODELS).")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of LLM names to run.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    models_to_run = [] if args.skip_llm else select_models(args)
    if not args.skip_llm and not models_to_run:
        print("LLM_MODELS is empty and no --model/--models given. Aborting.")
        return 1

    print("=" * 70)
    print("PHASE 5: LLM Experimentation (Part 2)")
    print(f"  LLMs:        {[m['name'] for m in models_to_run] or '(skipped)'}")
    print(f"  n_samples:   {args.n_samples}   k_fewshot: {args.k_fewshot}")
    print(f"  ckpt:        {args.ckpt}")
    print(f"  out_dir:     {out_dir}")
    print("=" * 70)

    print("\nLoading ATIS data...")
    train, dev, test, tag2id, id2tag = load_data()
    all_tags = [t for t in id2tag if t not in ("PAD", "X")]
    print(f"  train={len(train)} dev={len(dev)} test={len(test)}  num_tags={len(tag2id)}")

    print(f"\nStratified sampling of {args.n_samples} test samples...")
    subset_indices = stratified_sample(test, n_total=args.n_samples, seed=args.seed)
    subset = [test[i] for i in subset_indices]
    summary = summarise_subset(subset, list(range(len(subset))))
    print(f"  length bins : {summary['length_bins']}")
    print(f"  intents     : {summary['n_intent_classes']} distinct")
    print(f"  entity types: {summary['n_distinct_entity_types']} distinct")
    print(f"  sample sizes: min={min(len(s['tokens']) for s in subset)} "
          f"max={max(len(s['tokens']) for s in subset)} "
          f"avg={sum(len(s['tokens']) for s in subset)/len(subset):.1f}")

    few_shot_examples = pick_few_shot_examples(train, k=args.k_fewshot, seed=args.seed)
    print(f"\nFew-shot examples ({len(few_shot_examples)}):")
    for i, ex in enumerate(few_shot_examples, 1):
        print(f"  {i}. {' '.join(ex['tokens'])}")
        print(f"     -> {' '.join(ex['tags'])}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_results: List[dict] = []
    bert_metrics: dict = {"f1_macro": 0, "accuracy": 0, "n_samples": len(subset)}
    if not args.skip_bert:
        print(f"\nRunning BERT baseline on {len(subset)} samples...")
        bert_results = run_bert_on_subset(subset, tag2id, id2tag, Path(args.ckpt), device)
        bert_metrics = compute_metrics(bert_results)
        print(f"  BERT  F1(macro)={bert_metrics.get('f1_macro', 0):.4f}  "
              f"Acc={bert_metrics.get('accuracy', 0):.4f}")

    per_model_results: Dict[str, Dict[str, dict]] = {}

    for spec in models_to_run:
        name = spec["name"]
        base_url = spec["base_url"]
        print("\n" + "#" * 70)
        print(f"# LLM: {name}  @  {base_url}")
        print("#" * 70)

        client = LMStudioClient(
            base_url=base_url,
            model=name,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            timeout=LLM_TIMEOUT,
        )

        # Verify the model is reachable before running the full eval
        try:
            test_resp = client.chat(SYSTEM_PROMPT, "Reply with the single word: OK")
            print(f"  Server reachable. Sanity reply (first 60 chars): {test_resp[:60]!r}")
        except Exception as e:
            print(f"  Cannot reach {name} @ {base_url}: {e}")
            print(f"  Skipping {name}. Make sure it is loaded in LM Studio.")
            per_model_results[name] = {
                "zero_results": [], "few_results": [],
                "zero_metrics": {"error": str(e), "f1_macro": 0, "accuracy": 0, "n_samples": 0},
                "few_metrics":  {"error": str(e), "f1_macro": 0, "accuracy": 0, "n_samples": 0},
            }
            continue

        zero_results: List[dict] = []
        zero_metrics: dict = {"f1_macro": 0, "accuracy": 0, "n_samples": len(subset)}
        few_results: List[dict] = []
        few_metrics: dict = {"f1_macro": 0, "accuracy": 0, "n_samples": len(subset)}

        try:
            print(f"\n  Zero-shot on {len(subset)} samples...")
            zero_results = call_llm_on_subset(subset, client, all_tags, mode="zero")
            zero_metrics = compute_metrics(zero_results)
            print(f"  Zero  F1(macro)={zero_metrics.get('f1_macro', 0):.4f}  "
                  f"Acc={zero_metrics.get('accuracy', 0):.4f}")

            print(f"\n  Few-shot on {len(subset)} samples...")
            few_results = call_llm_on_subset(
                subset, client, all_tags, mode="few", few_shot_examples=few_shot_examples
            )
            few_metrics = compute_metrics(few_results)
            print(f"  Few   F1(macro)={few_metrics.get('f1_macro', 0):.4f}  "
                  f"Acc={few_metrics.get('accuracy', 0):.4f}")
        except Exception as e:
            print(f"  Error while running {name}: {e}")
            zero_metrics.setdefault("error", str(e))
            few_metrics.setdefault("error", str(e))

        per_model_results[name] = {
            "zero_results": zero_results,
            "few_results": few_results,
            "zero_metrics": zero_metrics,
            "few_metrics": few_metrics,
        }

    write_summary(
        out_dir=out_dir,
        subset=subset,
        bert_metrics=bert_metrics,
        bert_results=bert_results,
        per_model_results=per_model_results,
        few_shot_examples=few_shot_examples,
        models_to_run=models_to_run,
        subset_indices=subset_indices,
    )

    make_bar_chart(bert_metrics, per_model_results, out_dir / "f1_comparison.png", len(subset))
    print(f"Wrote {out_dir / 'f1_comparison.png'}")

    print("\n" + "=" * 70)
    print("Phase 5 complete.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())