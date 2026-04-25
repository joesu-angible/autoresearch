"""Export LCNet student model from best.pt to ONNX format.

Usage:
  python export_onnx.py                           # Export best.pt -> lcnet_student.onnx
  python export_onnx.py --checkpoint swa_best.pt  # Custom checkpoint
  python export_onnx.py --output model.onnx       # Custom output name
  python export_onnx.py --quantize                 # Also produce quantized INT8 version
"""

import argparse
import torch
import torch.nn.functional as functional
from pathlib import Path
from loguru import logger

from train import LCNet, LCNET_SCALE, SE_START_BLOCK, SE_REDUCTION, ACTIVATION, KERNEL_SIZES
from prepare import EMBEDDING_DIM, IMAGE_SIZE, TEACHER_REGISTRY


# Teacher used during training (needed to reconstruct model architecture)
TEACHER = "dinov3_ft"


class LCNetExport(torch.nn.Module):
    """Wrapper that exposes only the encode path for ONNX export.

    Takes [B, 3, 224, 224] input, returns [B, 256] L2-normalized embeddings.
    Default V1-compatible single-output module.
    """

    def __init__(self, model: LCNet) -> None:
        super().__init__()
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.stem_act = model.stem_act
        self.blocks = model.blocks
        self.conv_head = model.conv_head
        self.head_act = model.head_act
        self.gap = model.gap
        self.proj = model.proj

    def _summary(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_act(self.bn1(self.conv_stem(x)))
        x = self.blocks(x)
        x = self.head_act(self.conv_head(x))
        return self.gap(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summary = self._summary(x)
        emb = functional.normalize(self.proj(summary), p=2, dim=1)
        return emb


class LCNetProductnessExport(LCNetExport):
    """Two-output ONNX export: (embedding [B,256], productness_score [B] in [0,1]).

    Wraps a ProductnessLCNet checkpoint. The embedding output is byte-equivalent
    to the V1 LCNetExport for the same backbone weights.
    """

    def __init__(self, model: LCNet, productness_head: torch.nn.Module) -> None:
        super().__init__(model)
        self.productness_head = productness_head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        summary = self._summary(x)
        emb = functional.normalize(self.proj(summary), p=2, dim=1)
        score = torch.sigmoid(self.productness_head(summary).squeeze(-1))
        return emb, score


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LCNet student to ONNX")
    parser.add_argument("--checkpoint", type=str, default="best.pt",
                        help="Path to checkpoint (default: best.pt)")
    parser.add_argument("--output", type=str, default="lcnet_student.onnx",
                        help="Output ONNX path (default: lcnet_student.onnx)")
    parser.add_argument("--quantize", action="store_true",
                        help="Also produce quantized INT8 version")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument(
        "--include-productness",
        action="store_true",
        help="Export 2-output ONNX (embedding + productness_score). "
             "Requires a checkpoint trained with USE_PRODUCTNESS_CLS=True.",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    # Rebuild model with same architecture as training
    teacher_dims = {TEACHER: TEACHER_REGISTRY[TEACHER]["embedding_dim"]}
    if args.include_productness:
        from train_v2 import ProductnessLCNet
        model = ProductnessLCNet(
            scale=LCNET_SCALE,
            se_start_block=SE_START_BLOCK,
            se_reduction=SE_REDUCTION,
            activation=ACTIVATION,
            kernel_sizes=KERNEL_SIZES,
            embedding_dim=EMBEDDING_DIM,
            device="cpu",
            teacher_dims=teacher_dims,
        )
    else:
        model = LCNet(
            scale=LCNET_SCALE,
            se_start_block=SE_START_BLOCK,
            se_reduction=SE_REDUCTION,
            activation=ACTIVATION,
            kernel_sizes=KERNEL_SIZES,
            embedding_dim=EMBEDDING_DIM,
            device="cpu",
            teacher_dims=teacher_dims,
        )

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    combined = ckpt.get("combined_metric", 0)
    recall = ckpt.get("recall_at_1", 0)
    cosine = ckpt.get("mean_cosine", 0)
    logger.info(f"Loaded {ckpt_path}: epoch={epoch}, combined={combined:.4f}, "
                f"recall@1={recall:.4f}, cosine={cosine:.4f}")

    # Wrap for export (encode-only path; or 2-output if --include-productness)
    if args.include_productness:
        export_model = LCNetProductnessExport(model, model.productness_head)
        output_names = ["embedding", "productness_score"]
        dynamic_axes = {
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"},
            "productness_score": {0: "batch_size"},
        }
    else:
        export_model = LCNetExport(model)
        output_names = ["embedding"]
        dynamic_axes = {
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        }
    export_model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    # Verify output
    with torch.no_grad():
        out = export_model(dummy)
    if args.include_productness:
        emb_out, score_out = out
        logger.info(
            f"Embedding shape: {emb_out.shape}, norm: {emb_out.norm(dim=1).item():.4f}; "
            f"productness_score shape: {score_out.shape}, value: {score_out.item():.4f}"
        )
        assert emb_out.shape == (1, EMBEDDING_DIM), f"Expected embedding (1, {EMBEDDING_DIM}), got {emb_out.shape}"
        assert abs(emb_out.norm(dim=1).item() - 1.0) < 0.01, "Embedding not L2-normalized"
        assert score_out.shape == (1,), f"Expected score shape (1,), got {score_out.shape}"
        assert 0.0 <= score_out.item() <= 1.0, f"Productness score out of [0,1]: {score_out.item()}"
    else:
        logger.info(f"Output shape: {out.shape}, norm: {out.norm(dim=1).item():.4f}")
        assert out.shape == (1, EMBEDDING_DIM), f"Expected (1, {EMBEDDING_DIM}), got {out.shape}"
        assert abs(out.norm(dim=1).item() - 1.0) < 0.01, "Output not L2-normalized"

    # Count params
    total_params = sum(p.numel() for p in export_model.parameters())
    logger.info(f"Model params: {total_params:,} ({total_params/1e6:.2f}M)")

    # Export ONNX (use dynamo=False for compatibility without onnxscript)
    output_path = Path(args.output)
    torch.onnx.export(
        export_model,
        dummy,
        str(output_path),
        opset_version=args.opset,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Exported: {output_path} ({size_mb:.1f} MB)")

    # Verify ONNX
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(output_path))
    ort_outputs = sess.run(None, {"input": dummy.numpy()})

    # Compare PyTorch vs ONNX
    if args.include_productness:
        emb_diff = np.abs(emb_out.numpy() - ort_outputs[0]).max()
        score_diff = np.abs(score_out.numpy() - ort_outputs[1]).max()
        logger.info(f"PyTorch vs ONNX max diff — embedding: {emb_diff:.6f}, productness: {score_diff:.6f}")
        assert emb_diff < 1e-4, f"ONNX embedding verification failed: max diff {emb_diff}"
        assert score_diff < 1e-4, f"ONNX productness verification failed: max diff {score_diff}"
    else:
        diff = np.abs(out.numpy() - ort_outputs[0]).max()
        logger.info(f"PyTorch vs ONNX max diff: {diff:.6f}")
        assert diff < 1e-4, f"ONNX verification failed: max diff {diff}"
    logger.info("ONNX verification passed")

    # Quantize if requested
    if args.quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quant_path = output_path.with_suffix(".quant.onnx")
        quantize_dynamic(
            str(output_path),
            str(quant_path),
            weight_type=QuantType.QUInt8,
        )
        quant_size = quant_path.stat().st_size / 1024 / 1024
        logger.info(f"Quantized: {quant_path} ({quant_size:.1f} MB, "
                     f"{size_mb/quant_size:.1f}x compression)")

        # Verify quantized
        sess_q = ort.InferenceSession(str(quant_path))
        ort_q_out = sess_q.run(None, {"input": dummy.numpy()})[0]
        diff_q = np.abs(out.numpy() - ort_q_out).max()
        logger.info(f"PyTorch vs Quantized ONNX max diff: {diff_q:.6f}")

    print(f"\n{'='*50}")
    print(f"EXPORT COMPLETE")
    print(f"  ONNX:       {output_path} ({size_mb:.1f} MB)")
    if args.quantize:
        print(f"  Quantized:  {quant_path} ({quant_size:.1f} MB)")
    print(f"  Input:      [B, 3, {IMAGE_SIZE}, {IMAGE_SIZE}]")
    print(f"  Output:     [B, {EMBEDDING_DIM}] (L2-normalized)")
    print(f"  Params:     {total_params:,}")
    print(f"={'='*50}")


if __name__ == "__main__":
    main()
