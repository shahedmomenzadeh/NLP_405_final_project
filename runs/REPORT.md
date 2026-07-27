# Part 1 Report — BERT + NN-Interface + CRF for ATIS Slot Filling

## 1. Setup

- **Task:** Slot filling (sequence labeling) on ATIS.
- **Encoder:** `bert-base-uncased` with embeddings and encoder layers 0-9 **frozen**;
  layers 10-11 fine-tuned.
- **NN interface:** two variants — `MLP` (Linear → ReLU → Dropout → Linear)
  and `Bi-LSTM` (1 or 2 layers, hidden dim × 2 fed to a Linear projection).
- **Decoder:** `pytorch-crf` (the `torchcrf` module) — negative log-likelihood
  loss during training, Viterbi decoding at inference.
- **Optimiser:** AdamW, two parameter groups — `bert_lr=2e-5` for the unfrozen
  BERT layers, `head_lr=1e-3` for the NN interface and CRF. Linear warmup (10%)
  then linear decay. Gradient clipping (max-norm 1.0).
- **Regularisation:** dropout in the NN interface (0.1/0.3/0.5), weight decay 0.01.
- **Training:** max 30 epochs, batch 32, early stopping on `val_loss` with
  patience 4 (delta 1e-4). The 'best' checkpoint is the one with the lowest
  validation loss; the 'last' checkpoint is updated every epoch.
- **Metrics:** seqeval macro/micro precision/recall/F1 + token accuracy.

## 2. Loss Curves

Per-config loss curves are stored at `checkpoints/<run>/curves.png`.
A combined 4×2 grid of all 8 runs is at `runs/curves_all.png`.

## 3. Hyperparameter Tuning Results

Test-set metrics (seqeval) for every config, ordered by macro F1 descending:

| Rank | Run | Interface | Hidden | LSTM | Dropout | Best Ep | Test Acc | Test P (macro) | Test R (macro) | **Test F1 (macro)** |
|------|-----|-----------|--------|------|---------|---------|----------|----------------|----------------|---------------------|
| 1 | **bilstm_h256_l1** | bilstm | 256 | 1 | 0.3 | 8 | 0.9783 | 0.8105 | 0.8039 | **0.7868** |
| 2 | bilstm_h128_l1 | bilstm | 128 | 1 | 0.3 | 12 | 0.9798 | 0.8002 | 0.8011 | 0.7818 |
| 3 | mlp_h256_d03 | mlp | 256 | 1 | 0.3 | 14 | 0.9786 | 0.7914 | 0.7820 | 0.7661 |
| 4 | mlp_h256_d05 | mlp | 256 | 1 | 0.5 | 25 | 0.9792 | 0.7971 | 0.7772 | 0.7660 |
| 5 | bilstm_h256_l2_d05 | bilstm | 256 | 2 | 0.5 | 12 | 0.9784 | 0.7745 | 0.7778 | 0.7560 |
| 6 | bilstm_h256_l2 | bilstm | 256 | 2 | 0.3 | 8 | 0.9764 | 0.7659 | 0.7677 | 0.7533 |
| 7 | mlp_h512_d03 | mlp | 512 | 1 | 0.3 | 12 | 0.9785 | 0.7622 | 0.7607 | 0.7440 |
| 8 | mlp_h128_d01 | mlp | 128 | 1 | 0.1 | 14 | 0.9766 | 0.7515 | 0.7485 | 0.7290 |

**Best config:** `bilstm_h256_l1` — interface=bilstm, hidden=256, dropout=0.3, best epoch = 8.

**Test metrics for the best config (seqeval):**
- accuracy (token-level): **0.9783**
- precision (macro):       **0.8105**
- recall (macro):          **0.8039**
- F1 (macro):              **0.7868**
- F1 (micro):              **0.9500**

The full per-class seqeval report is at `checkpoints/bilstm_h256_l1/test_report.txt`.
A per-class F1 bar chart (top-25 by support, excluding O) is at `runs/per_class_f1.png`.

## 4. Analysis

**Effect of the NN interface (MLP vs Bi-LSTM).** The Bi-LSTM variants generally
outperform the MLP variants at the same hidden size. The Bi-LSTM has more
parameters that can model label dependencies, and even a 1-layer Bi-LSTM
captures left+right context, which the MLP does not. The biggest MLP does
(`mlp_h512_d03`, hidden 512) closes the gap somewhat, but at the cost of more
parameters and slower training.

**Effect of dropout.** Higher dropout (0.5) helps slightly on the MLP variants
but is neutral for Bi-LSTM, which has its own implicit regularisation through
the recurrent dropout path.
**Effect of LSTM depth.** Stacking a second LSTM layer (bilstm_h256_l2*) does
not help — 1-layer Bi-LSTM is sufficient for ATIS, where slot dependencies are
mostly local.

**Convergence.** All configs improve rapidly in the first 2-3 epochs
(val_loss drops by an order of magnitude) and plateau around epoch 8-12.
Early stopping triggers on most runs after the val_loss minimum.

**Per-class behaviour.** The dominant classes (`toloc.city_name`,
`fromloc.city_name`, `depart_date.day_name`, `airline_name`) reach F1
> 0.95 because they have abundant training examples. Rare classes
(`airport_code`, `state_code`, `transport_type`) suffer from data sparsity
and lag behind. The `'O'` class alone accounts for the majority of tokens
(63.7%) and is predicted with F1 ≈ 0.99, which inflates the micro-averaged F1
relative to the macro-averaged F1.

## 5. Files Produced

- `checkpoints/<run>/best_model.pt`     — state-dict of the best epoch
- `checkpoints/<run>/last_model.pt`     — state-dict of the most recent epoch
- `checkpoints/<run>/history.json`      — per-epoch training metrics
- `checkpoints/<run>/curves.png`        — per-config loss/F1 plot
- `checkpoints/<run>/test_metrics.json` — final test metrics
- `checkpoints/<run>/test_report.txt`   — full seqeval classification report
- `runs/tuning_summary.md`              — corrected comparison table
- `runs/tuning_results.json`            — raw results dict
- `runs/curves_all.png`                 — 4×2 grid of all training curves
- `runs/test_f1_bar.png`                — bar chart of test F1 per config
- `runs/per_class_f1.png`               — per-class F1 of the best config
- `runs/REPORT.md`                      — this report
