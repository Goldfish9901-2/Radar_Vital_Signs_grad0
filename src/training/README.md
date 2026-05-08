# Model Training and Evaluation

This folder contains the model training and evaluation entry points:

- `train_model.py`: train HeartTimeMixer, TCN, or Transformer.
- `evaluate_model.py`: evaluate a trained checkpoint on any exported dataset split, including cross-dataset testing.
- `datasets.py`: read samples generated under `training_exports`.

Run commands inside the `radar_dev` container from `/Radar_Vital_Signs`.

## HeartTimeMixer

FTU training:

```bash
python3 /Radar_Vital_Signs/src/training/train_model.py \
  --model heart_timemixer \
  --datasets FTU \
  --export-dir /Radar_Vital_Signs/training_exports \
  --output-dir /Radar_Vital_Signs/model_outputs/htm_ftu_test \
  --epochs 80 \
  --batch-size 32 \
  --d-model 32 \
  --d-ff 64 \
  --e-layers 1 \
  --dropout 0.15 \
  --down-sampling-layers 3
```

All exported datasets:

```bash
python3 /Radar_Vital_Signs/src/training/train_model.py \
  --model heart_timemixer \
  --export-dir /Radar_Vital_Signs/training_exports \
  --output-dir /Radar_Vital_Signs/model_outputs/heart_timemixer_all \
  --epochs 80 \
  --batch-size 64 \
  --d-model 32 \
  --d-ff 64 \
  --e-layers 1 \
  --dropout 0.2 \
  --down-sampling-layers 3
```

## TCN

Recommended FTU baseline:

```bash
python3 /Radar_Vital_Signs/src/training/train_model.py \
  --model tcn \
  --datasets FTU \
  --export-dir /Radar_Vital_Signs/training_exports \
  --output-dir /Radar_Vital_Signs/model_outputs/tcn_ftu_test \
  --epochs 80 \
  --batch-size 32 \
  --hidden-channels 32 \
  --num-blocks 3 \
  --kernel-size 7 \
  --dropout 0.2
```

Larger TCN:

```bash
python3 /Radar_Vital_Signs/src/training/train_model.py \
  --model tcn \
  --datasets FTU \
  --export-dir /Radar_Vital_Signs/training_exports \
  --output-dir /Radar_Vital_Signs/model_outputs/tcn_ftu_balanced \
  --epochs 80 \
  --batch-size 32 \
  --hidden-channels 48 \
  --num-blocks 4 \
  --kernel-size 7 \
  --dropout 0.2
```

Time-domain only TCN:

```bash
python3 /Radar_Vital_Signs/src/training/train_model.py \
  --model tcn \
  --datasets FTU \
  --export-dir /Radar_Vital_Signs/training_exports \
  --output-dir /Radar_Vital_Signs/model_outputs/tcn_ftu_balanced_time_only \
  --epochs 80 \
  --batch-size 32 \
  --hidden-channels 48 \
  --num-blocks 4 \
  --kernel-size 7 \
  --dropout 0.2 \
  --time-only
```

## Transformer

FTU training:

```bash
python3 /Radar_Vital_Signs/src/training/train_model.py \
  --model transformer \
  --datasets FTU \
  --export-dir /Radar_Vital_Signs/training_exports \
  --output-dir /Radar_Vital_Signs/model_outputs/transformer_ftu_balanced \
  --epochs 80 \
  --batch-size 32 \
  --d-model 48 \
  --d-ff 128 \
  --num-layers 2 \
  --nhead 4 \
  --dropout 0.2
```

## Evaluation

Evaluate the checkpoint on its own test split:

```bash
python3 /Radar_Vital_Signs/src/training/evaluate_model.py \
  --model-dir /Radar_Vital_Signs/model_outputs/tcn_ftu_test \
  --export-dir /Radar_Vital_Signs/training_exports \
  --target-datasets FTU \
  --split test \
  --batch-size 128
```

Cross-dataset evaluation, for example FTU-trained TCN tested on PhysDrive:

```bash
python3 /Radar_Vital_Signs/src/training/evaluate_model.py \
  --model-dir /Radar_Vital_Signs/model_outputs/tcn_ftu_test \
  --export-dir /Radar_Vital_Signs/training_exports \
  --target-datasets PhysDrive \
  --split test \
  --batch-size 128
```

Cross-dataset evaluation, for example FTU-trained TCN tested on BGT60TR13C:

```bash
python3 /Radar_Vital_Signs/src/training/evaluate_model.py \
  --model-dir /Radar_Vital_Signs/model_outputs/htm_ftu_test \
  --export-dir /Radar_Vital_Signs/training_exports \
  --target-datasets BGT60TR13C \
  --split test \
  --batch-size 128
```

Evaluation output includes:

- `overall`: global loss, MAE, and tolerance hit rates.
- `by_dataset`: metrics by dataset.
- `by_participant`: metrics by participant/session id.
- `by_group`: metrics by manifest group key.
- `within_3bpm_percent`: percentage of predictions with absolute error <= 3 BPM.
- `within_5bpm_percent`: percentage of predictions with absolute error <= 5 BPM.

The JSON is saved automatically under the model output directory, for example:

```text
/Radar_Vital_Signs/model_outputs/tcn_ftu_balanced_small/eval_test_PhysDrive.json
```

## Useful Options

- `--datasets FTU`: train on one dataset. Omit it to train on all exported datasets.
- `--time-only`: disable the frequency-domain branch.
- `--limit-batches N`: debug quickly with only N batches per epoch.
- `--seed N`: change the random seed.
- `--patience N`: early stopping patience.
