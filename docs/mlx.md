# MLX, MPS, and why we train with PyTorch

You asked about training with **MLX** (Apple's Metal-native array framework). Here's the reasoning
behind the current choice and what an MLX track would take.

## PyTorch has no MLX backend

On Apple Silicon, PyTorch's GPU backend is **MPS** (Metal Performance Shaders) — it runs on the
same M-series GPU/Metal stack that MLX targets. There is **no** `torch.device("mlx")`; MLX is a
separate framework with its own API. So "use the M-series GPU for training" is delivered by
**PyTorch + MPS**, which is what `tamiltts/train.py` uses (it auto-detects MPS).

## Why not train in MLX directly

The project goals are *small + ONNX + fast CPU inference + Python **and** Rust SDKs*:

- MLX has **no native ONNX export**. A trained MLX model would need a hand-written
  MLX→PyTorch/ONNX bridge to reach `onnxruntime` / the Rust `ort` SDK.
- There is **no mature, proven "train VITS in MLX" recipe**; it would be custom research code.

Choosing MLX would trade a clean, proven export path for an experimental one — without making
inference any faster (inference is CPU/ONNX, not MLX).

## If you still want an MLX track later

A reasonable experimental path, kept out of the default pipeline:

1. Implement the VITS **inference generator** in MLX, matching layer shapes to the PyTorch model.
2. Train (or fine-tune) in MLX on the M-series GPU.
3. Port weights into the PyTorch VITS and run the existing `tamiltts.export_onnx` to get ONNX.

The seam to validate is weight-transfer fidelity. Until then, **PyTorch+MPS → ONNX** is the
supported route.
