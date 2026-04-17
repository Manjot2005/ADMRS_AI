"""
model_optimization.py — ONNX Model Quantization & Export
Converts the trained PyTorch U-Net to ONNX format for edge deployment.
Supports Raspberry Pi, drones, and low-power devices.

Run: python model_optimization.py
"""

import logging
import numpy as np
from pathlib import Path

import torch
import segmentation_models_pytorch as smp

from config import (
    MODEL_PATH, MODELS_DIR,
    IN_CHANNELS, NUM_CLASSES, ENCODER_NAME, TILE_SIZE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ONNX_PATH     = MODELS_DIR / "unet_forest_optimized.onnx"
ONNX_INT8_PATH = MODELS_DIR / "unet_forest_int8.onnx"


def export_to_onnx(pytorch_model_path: Path, onnx_path: Path) -> Path:
    """
    Export PyTorch U-Net to ONNX format.
    ONNX models are framework-agnostic and run on CPU/GPU/edge devices.
    """
    log.info("Loading PyTorch model...")
    device = torch.device("cpu")

    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        activation=None,
    )

    if pytorch_model_path.exists():
        model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
        log.info(f"Loaded weights from {pytorch_model_path}")
    else:
        log.warning(f"No weights found at {pytorch_model_path} — exporting untrained model (for demo)")

    model.eval()

    # Dummy input for tracing
    dummy_input = torch.randn(1, IN_CHANNELS, TILE_SIZE, TILE_SIZE)

    log.info(f"Exporting to ONNX: {onnx_path} ...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,    # Optimize constant expressions
        input_names=["satellite_tile"],
        output_names=["segmentation_logits"],
        dynamic_axes={
            "satellite_tile":       {0: "batch_size"},
            "segmentation_logits":  {0: "batch_size"},
        },
        verbose=False,
    )

    # Check file size
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    log.info(f"ONNX model exported: {size_mb:.1f} MB")

    return onnx_path


def quantize_onnx(input_path: Path, output_path: Path) -> Path:
    """
    Apply INT8 dynamic quantization to ONNX model.
    Reduces model size by ~4x, speeds up inference on CPU.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        log.info("Applying INT8 dynamic quantization ...")
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            optimize_model=True,
        )

        orig_mb = input_path.stat().st_size / (1024 * 1024)
        quant_mb = output_path.stat().st_size / (1024 * 1024)
        reduction = (1 - quant_mb / orig_mb) * 100

        log.info(f"Original: {orig_mb:.1f} MB → Quantized: {quant_mb:.1f} MB ({reduction:.0f}% reduction)")
        return output_path

    except ImportError:
        log.warning("onnxruntime.quantization not available — skipping INT8 quantization")
        return input_path


def benchmark_onnx(onnx_path: Path, n_runs: int = 50) -> dict:
    """
    Benchmark ONNX model inference speed.
    """
    import time
    try:
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(onnx_path), sess_options)

        dummy = np.random.randn(1, IN_CHANNELS, TILE_SIZE, TILE_SIZE).astype(np.float32)
        input_name = session.get_inputs()[0].name

        # Warm-up
        for _ in range(5):
            session.run(None, {input_name: dummy})

        # Benchmark
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            session.run(None, {input_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(times)
        p95_ms = np.percentile(times, 95)
        fps    = 1000 / avg_ms

        log.info(f"Inference: avg={avg_ms:.1f}ms | p95={p95_ms:.1f}ms | {fps:.1f} tiles/sec")

        return {
            "avg_ms": round(avg_ms, 1),
            "p95_ms": round(p95_ms, 1),
            "fps":    round(fps, 1),
            "model":  onnx_path.name,
        }

    except Exception as e:
        log.error(f"Benchmark failed: {e}")
        return {}


def run_onnx_inference(onnx_path: Path, tile: np.ndarray) -> np.ndarray:
    """
    Run inference using ONNX runtime (no PyTorch required on edge devices).

    Args:
        onnx_path : Path to .onnx model file
        tile      : (H, W, 4) float32 array

    Returns:
        mask : (H, W) uint8 binary mask (0=forest, 1=deforested)
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name

    # Prepare input
    x = tile.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1,4,H,W)
    logits = session.run(None, {input_name: x})[0]               # (1,2,H,W)
    mask   = np.argmax(logits, axis=1).squeeze().astype(np.uint8) # (H,W)

    return mask


def main():
    log.info("=" * 60)
    log.info("Model Optimization: PyTorch → ONNX → INT8 Quantization")
    log.info("=" * 60)

    # Step 1: Export to ONNX
    onnx_path = export_to_onnx(MODEL_PATH, ONNX_PATH)

    # Step 2: Quantize
    quant_path = quantize_onnx(onnx_path, ONNX_INT8_PATH)

    # Step 3: Benchmark both
    log.info("\nBenchmarking FP32 ONNX model:")
    fp32_stats = benchmark_onnx(onnx_path)

    if quant_path != onnx_path:
        log.info("\nBenchmarking INT8 quantized model:")
        int8_stats = benchmark_onnx(quant_path)

    log.info("\n" + "=" * 60)
    log.info("✅ Optimization complete")
    log.info(f"   FP32 model : {ONNX_PATH}")
    log.info(f"   INT8 model : {ONNX_INT8_PATH}")
    log.info("   Deploy either file on Raspberry Pi / drone with:")
    log.info("   pip install onnxruntime")
    log.info("   from model_optimization import run_onnx_inference")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
