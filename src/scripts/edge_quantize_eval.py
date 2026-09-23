"""Does a validation-fitted temperature survive int8 quantization?

The audit shows that one scalar fitted on validation data returns
foundation-model calibration error to the supervised range in float32. On-device
inference runs in int8, which changes the logits. This script tests whether the
*locked* temperature and the *locked* abstention thresholds still hold after
quantization.

Protocol, and the reason this is a result rather than an engineering step:

  * The temperature is NOT re-fitted on int8 outputs. Re-fitting would be the
    same error the validation lock exists to prevent, one layer down. The
    float32 value is applied unchanged.
  * The abstention thresholds are set once on the float32 selection split and
    applied unchanged to both precisions.
  * The int8 scale-selection pass -- confusingly also called "calibration" in
    ONNX Runtime, and unrelated to probability calibration -- uses fit-split
    data only. Held-out trials never touch it.

Prerequisites (see requirements-edge.txt):
    pip install onnx onnxruntime

The checkpoint must be a TorchScript archive so this script needs no knowledge
of the model class:
    torch.jit.script(model).save("model.ts")
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

TARGET_COVERAGES = (0.4, 0.6, 0.8, 0.9)


def _require(mod: str):
    try:
        return __import__(mod)
    except ImportError as exc:
        raise SystemExit(
            f"{mod} is not installed. Run:  pip install -r requirements-edge.txt"
        ) from exc


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(labels.size), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def coverage_thresholds(conf: np.ndarray, targets=TARGET_COVERAGES) -> dict[float, float]:
    """Confidence cut delivering each target coverage on the split it is fit on."""
    return {t: float(np.quantile(conf, 1.0 - t)) for t in targets}


def delivered(conf: np.ndarray, thresholds: dict[float, float]) -> dict[str, float]:
    return {f"cov@{t}": float((conf >= thr).mean()) for t, thr in thresholds.items()}


def export_onnx(ts_path: Path, sample: np.ndarray, out: Path) -> Path:
    torch = _require("torch")
    model = torch.jit.load(str(ts_path), map_location="cpu").eval()
    dummy = torch.from_numpy(sample[:1]).float()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    return out


def quantize(fp32: Path, fit_X: np.ndarray, out: Path) -> Path:
    _require("onnxruntime")
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static

    class FitSplitReader(CalibrationDataReader):
        """Feeds fit-split trials only. Held-out data must never appear here."""

        def __init__(self, X: np.ndarray, batch: int = 8):
            self.batches = [X[i:i + batch].astype(np.float32)
                            for i in range(0, len(X), batch)]
            self.i = 0

        def get_next(self):
            if self.i >= len(self.batches):
                return None
            b = self.batches[self.i]
            self.i += 1
            return {"input": b}

    quantize_static(
        model_input=str(fp32), model_output=str(out),
        calibration_data_reader=FitSplitReader(fit_X),
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
    )
    return out


def run_onnx(model_path: Path, X: np.ndarray, batch: int = 32) -> tuple[np.ndarray, float]:
    ort = _require("onnxruntime")
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    outs, t0 = [], time.perf_counter()
    for i in range(0, len(X), batch):
        outs.append(sess.run(None, {name: X[i:i + batch].astype(np.float32)})[0])
    elapsed = time.perf_counter() - t0
    return np.concatenate(outs, axis=0), elapsed / max(len(X), 1) * 1e3  # ms/trial


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="TorchScript archive: torch.jit.script(model).save(...)")
    ap.add_argument("--fit", type=Path, required=True,
                    help="npz with X (and y) for the fit split; used for int8 scales")
    ap.add_argument("--selection", type=Path, required=True,
                    help="npz for the selection split; abstention thresholds are set here")
    ap.add_argument("--heldout", type=Path, required=True, help="npz for the held-out session")
    ap.add_argument("--temperature", type=float, required=True,
                    help="the validation-fitted temperature, applied unchanged to both")
    ap.add_argument("--out", type=Path, default=Path("results") / "edge")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(
            f"No checkpoint at {args.checkpoint}.\n"
            "The training runner does not currently save weights. Add, after the\n"
            "best-epoch checkpoint is selected on validation:\n"
            "    torch.jit.script(model).save(out_dir / 'model.ts')\n"
            "then retrain one subject."
        )
    args.out.mkdir(parents=True, exist_ok=True)

    def load(p: Path):
        d = np.load(p, allow_pickle=True)
        return d["X"].astype(np.float32), d["y"].astype(int)

    fit_X, _ = load(args.fit)
    sel_X, sel_y = load(args.selection)
    held_X, held_y = load(args.heldout)

    fp32 = export_onnx(args.checkpoint, fit_X, args.out / "model_fp32.onnx")
    int8 = quantize(fp32, fit_X, args.out / "model_int8.onnx")

    sel_logits_fp32, _ = run_onnx(fp32, sel_X)
    held_fp32, ms_fp32 = run_onnx(fp32, held_X)
    held_int8, ms_int8 = run_onnx(int8, held_X)

    T = args.temperature
    # Thresholds are set once, on float32 selection-split confidences, and are
    # then frozen. Neither precision gets its own thresholds.
    thr = coverage_thresholds(softmax(sel_logits_fp32, T).max(axis=1))

    rows = {}
    for tag, logits, ms in (("fp32", held_fp32, ms_fp32), ("int8", held_int8, ms_int8)):
        probs = softmax(logits, T)          # locked temperature, not re-fitted
        conf = probs.max(axis=1)
        rows[tag] = {
            "accuracy": float((probs.argmax(axis=1) == held_y).mean()),
            "brier": brier(probs, held_y),
            "ece_locked_T": ece(probs, held_y),
            "ece_uncalibrated": ece(softmax(logits, 1.0), held_y),
            "ms_per_trial": ms,
            "model_bytes": (fp32 if tag == "fp32" else int8).stat().st_size,
            **delivered(conf, thr),
        }

    rows["delta_int8_minus_fp32"] = {
        k: rows["int8"][k] - rows["fp32"][k]
        for k in rows["fp32"] if isinstance(rows["fp32"][k], float)
    }
    rows["_protocol"] = {
        "temperature_locked_from_validation": T,
        "temperature_refit_on_int8": False,
        "thresholds_from": "float32 selection split, applied unchanged to both",
        "int8_scale_calibration_data": "fit split only",
        "target_coverages": list(TARGET_COVERAGES),
    }
    (args.out / "edge_quantization.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({k: v for k, v in rows.items() if not k.startswith("_")}, indent=2))
    print(f"\nWrote {args.out / 'edge_quantization.json'}")


if __name__ == "__main__":
    main()
