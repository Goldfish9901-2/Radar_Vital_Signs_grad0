# Model Training and Evaluation

This folder contains the training and evaluation entry points for the current CycleFormer-based radar HR task.

- `train_model.py`: train CycleFormer, TCN, Transformer, PatchTST, TimesNet, or the retained HeartTimeMixer.
- `evaluate_model.py`: evaluate a trained checkpoint on any exported dataset split.
- `evaluate_signal_baselines.py`: evaluate FFT/STFT baselines without training.
- `adapt_source_free.py`: run pseudo-label adaptation on unlabeled target-domain windows.
- `datasets.py`: read samples generated under `training_exports`.

## Main Model: CycleFormer

```bash
python3 src/training/train_model.py \
  --model cycleformer \
  --datasets FTU \
  --export-dir training_exports \
  --output-dir model_outputs/cycleformer_ftu_source \
  --epochs 80 \
  --batch-size 32 \
  --d-model 64 \
  --d-ff 128 \
  --num-layers 2 \
  --nhead 4
```

## Representative Deep Baselines

```bash
python3 src/training/train_model.py \
  --model tcn \
  --datasets FTU \
  --export-dir training_exports \
  --output-dir model_outputs/tcn_ftu_source \
  --epochs 80 \
  --batch-size 32
```

Use the same template with:

```text
transformer
patchtst
timesnet
```

## FFT/STFT Baselines

```bash
python3 src/training/evaluate_signal_baselines.py \
  --method fft \
  --export-dir training_exports \
  --target-datasets PhysDrive \
  --split test \
  --output-json model_outputs/signal_baselines/fft_physdrive_test.json
```

```bash
python3 src/training/evaluate_signal_baselines.py \
  --method stft \
  --export-dir training_exports \
  --target-datasets PhysDrive \
  --split test \
  --output-json model_outputs/signal_baselines/stft_physdrive_test.json
```

## Source Only Evaluation

```bash
python3 src/training/evaluate_model.py \
  --model-dir model_outputs/cycleformer_ftu_source \
  --export-dir training_exports \
  --target-datasets PhysDrive \
  --split test \
  --batch-size 128
```

## Pseudo-label Adaptation

```bash
python3 src/training/adapt_source_free.py \
  --source-model-dir model_outputs/cycleformer_ftu_source \
  --export-dir training_exports \
  --target-datasets PhysDrive \
  --adapt-split train \
  --eval-split test \
  --output-dir model_outputs/cycleformer_ftu_to_physdrive_pseudo \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.0001 \
  --temporal-window 5
```

The adapted checkpoint is saved as:

```text
model_outputs/cycleformer_ftu_to_physdrive_pseudo/best.pt
```

## Metrics

Evaluation output includes:

- `overall`: global MAE, RMSE, Pearson `r`, and tolerance hit rates.
- `by_dataset`: metrics by dataset.
- `by_participant`: metrics by participant/session id.
- `by_group`: metrics by manifest group key.
- `within_5bpm_percent` and `within_10bpm_percent`: practical tolerance rates.

## Useful Options

- `--datasets FTU`: train on one dataset. Omit it to train on all exported datasets.
- `--time-only`: disable the frequency-domain branch.
- `--limit-batches N`: debug quickly with only N batches per epoch.
- `--seed N`: change the random seed.
- `--patience N`: early stopping patience.
