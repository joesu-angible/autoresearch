"""T9 verification: ONNX export — default V1-compatible + opt-in productness mode.

CPU-only. Runs the ONNX export wrapper classes (LCNetExport, LCNetProductnessExport)
end-to-end against torch.onnx.export and onnxruntime, checking:
  - Default mode: 1 output named 'embedding'; matches PyTorch within 1e-4.
  - Productness mode: 2 outputs ['embedding', 'productness_score']; both match
    PyTorch within 1e-4; score lives in [0, 1].
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
ort = pytest.importorskip("onnxruntime")

from train_v2 import ProductnessLCNet  # noqa: E402
from train import LCNet, LCNET_SCALE, SE_START_BLOCK, SE_REDUCTION, ACTIVATION, KERNEL_SIZES  # noqa: E402
from prepare import EMBEDDING_DIM, IMAGE_SIZE  # noqa: E402
from export_onnx import LCNetExport, LCNetProductnessExport  # noqa: E402

TEACHER_DIMS = {"dinov3_ft": 1024}
ATOL = 1e-4


def _build_lcnet():
    return LCNet(
        scale=LCNET_SCALE,
        se_start_block=SE_START_BLOCK,
        se_reduction=SE_REDUCTION,
        activation=ACTIVATION,
        kernel_sizes=KERNEL_SIZES,
        embedding_dim=EMBEDDING_DIM,
        device="cpu",
        teacher_dims=TEACHER_DIMS,
    ).eval()


def _build_productness_lcnet():
    return ProductnessLCNet(
        scale=LCNET_SCALE,
        se_start_block=SE_START_BLOCK,
        se_reduction=SE_REDUCTION,
        activation=ACTIVATION,
        kernel_sizes=KERNEL_SIZES,
        embedding_dim=EMBEDDING_DIM,
        device="cpu",
        teacher_dims=TEACHER_DIMS,
        productness_hidden=64,
    ).eval()


def _export(wrapper: torch.nn.Module, output_names: list[str], path: Path) -> None:
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    dyn = {"input": {0: "batch_size"}}
    for name in output_names:
        dyn[name] = {0: "batch_size"}
    torch.onnx.export(
        wrapper,
        dummy,
        str(path),
        opset_version=17,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dyn,
        dynamo=False,
    )


def test_default_export_single_output(tmp_path: Path):
    model = _build_lcnet()
    wrapper = LCNetExport(model).eval()
    onnx_path = tmp_path / "default.onnx"
    _export(wrapper, ["embedding"], onnx_path)

    sess = ort.InferenceSession(str(onnx_path))
    outputs = sess.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].name == "embedding"

    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        torch_out = wrapper(dummy).numpy()
    ort_out = sess.run(None, {"input": dummy.numpy()})[0]
    # PyTorch ↔ ONNX equivalence is the export contract under test
    assert np.allclose(torch_out, ort_out, atol=ATOL), (
        f"max diff {np.abs(torch_out - ort_out).max():.6f}"
    )
    assert ort_out.shape == (2, EMBEDDING_DIM)


def test_productness_export_two_outputs(tmp_path: Path):
    model = _build_productness_lcnet()
    wrapper = LCNetProductnessExport(model, model.productness_head).eval()
    onnx_path = tmp_path / "productness.onnx"
    _export(wrapper, ["embedding", "productness_score"], onnx_path)

    sess = ort.InferenceSession(str(onnx_path))
    out_names = [o.name for o in sess.get_outputs()]
    assert out_names == ["embedding", "productness_score"]

    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        torch_emb, torch_score = wrapper(dummy)
    ort_emb, ort_score = sess.run(None, {"input": dummy.numpy()})
    assert np.allclose(torch_emb.numpy(), ort_emb, atol=ATOL)
    assert np.allclose(torch_score.numpy(), ort_score, atol=ATOL)
    # productness score is sigmoid → [0, 1]
    assert (ort_score >= 0.0).all() and (ort_score <= 1.0).all()


def test_productness_default_path_not_polluted(tmp_path: Path):
    """The default LCNetExport class must NOT add a productness output even when
    given a checkpoint with productness_head present."""
    model = _build_productness_lcnet()
    wrapper = LCNetExport(model).eval()  # default wrapper, ignores head
    onnx_path = tmp_path / "default_from_productness.onnx"
    _export(wrapper, ["embedding"], onnx_path)

    sess = ort.InferenceSession(str(onnx_path))
    assert len(sess.get_outputs()) == 1
    assert sess.get_outputs()[0].name == "embedding"
