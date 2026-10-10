"""Remote CPU-only diagnostics and eight-subject leave-one-out Ridge probes."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def run(args, cwd=None):
    print("$", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def main():
    work = Path("/kaggle/working")
    if not work.is_dir() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("This runner only executes on Kaggle")
    archives = list(Path("/kaggle/input").rglob("radar_vital_signs_code.zip"))
    if archives:
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(work / "code")
        code = work / "code/Radar_Vital_Signs_grad0"
    else:
        code = next(p.parent for p in Path("/kaggle/input").rglob("validate_bgt60_diagnostic.py"))
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    run([sys.executable, "-m", "uv", "pip", "install", "--system", "numpy", "scipy", "pandas", "h5py", "matplotlib", "scikit-learn"])
    run([sys.executable, "validate_adc_rda_synthetic.py"], cwd=code)
    run([sys.executable, "validate_bgt60_diagnostic.py"], cwd=code)
    out = work / "diagnostics"
    cmd = [sys.executable, "-m", "src.data.diagnose_bgt60_adc", "--input-root", "/kaggle/input", "--output-dir", out]
    calibration = code / "kaggle/bgt60_calibration.json"
    if calibration.exists():
        cmd += ["--calibration", calibration]
    run(cmd, cwd=code)
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    results = {}
    for path in out.glob("*_probe.npz"):
        data = np.load(path)
        valid = np.isfinite(data["y"])
        x, y, subjects = data["x"][valid], data["y"][valid], data["subjects"][valid]
        folds = []
        for subject in sorted(set(subjects.tolist())):
            train, test = subjects != subject, subjects == subject
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.))
            model.fit(x[train], y[train])
            p = model.predict(x[test])
            r = float(np.corrcoef(p, y[test])[0, 1]) if p.std() > .01 and y[test].std() > .01 else None
            folds.append({"subject": subject, "windows": int(test.sum()),
                          "mae_bpm": float(np.abs(p-y[test]).mean()),
                          "prediction_std_bpm": float(p.std()), "pearson_r": r,
                          "train_mean_constant_mae_bpm": float(np.abs(y[train].mean()-y[test]).mean())})
        errors = np.array([f["mae_bpm"] for f in folds])
        rng = np.random.default_rng(42)
        ci = np.quantile(rng.choice(errors, (10000, len(errors)), replace=True).mean(axis=1), [.025, .975])
        results[path.stem] = {"folds": folds, "subject_macro_mae_bpm": float(np.mean([f["mae_bpm"] for f in folds])),
                              "subject_bootstrap_macro_mae_95pct_ci": ci.tolist(),
                              "alpha": 10, "evaluation": "leave-one-subject-out; all eight subjects; scaler fitted on training subjects"}
    (out / "linear_probes.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    with (out / "REPORT.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## Leave-one-subject-out spectral Ridge probes\n\nFixed alpha=10; each scaler uses training subjects only. Confidence intervals bootstrap the eight subject MAEs, not overlapping windows.\n\n| Candidate | Subject macro MAE | Subject bootstrap 95% CI |\n| --- | ---: | --- |\n")
        for name, result in results.items():
            lo, hi = result["subject_bootstrap_macro_mae_95pct_ci"]
            stream.write(f'| {name} | {result["subject_macro_mae_bpm"]:.3f} | {lo:.3f}–{hi:.3f} |\n')
    print("CPU diagnostics complete; no neural network training", flush=True)


if __name__ == "__main__":
    main()
