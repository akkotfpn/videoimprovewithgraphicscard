"""Otimizações da v0.3: modelos uint8 com pós-processamento no grafo, lote, orçamento de memória e
plano por vídeo (resolução de trabalho + modelo)."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from engine.core.adapt import choose_job, detail_psnr, half_psnr
from engine.core.hardware import Backend, GPUInfo, HardwareProfile, Vendor, choose_plan
from engine.core.pipeline import Pipeline, PipelineConfig
from engine.models import realesrgan
from engine.models.onnx_writer import sr_model_u8
from engine.models.pth_loader import state_dict
from engine.stages.upscale import OnnxUpscale

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "_weights"
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg ausente")


def needs(name):
    return pytest.mark.skipif(not (WEIGHTS / name).exists(), reason="rode: python -m engine models install all")


def _plan(gpu, providers):
    h = HardwareProfile("test", "cpu", 32, 31830, [gpu], None, list(providers) + ["CPUExecutionProvider"])
    return choose_plan(h)


def _ultra_5070():
    return _plan(GPUInfo("NVIDIA GeForce RTX 5070", Vendor.NVIDIA, 12227, (12, 0), cuda_index=0),
                 ["CUDAExecutionProvider"])


# ---------------- plano por vídeo ----------------
def test_1080p_true_hd_on_5070_uses_light_model():
    job = choose_job(_ultra_5070(), 1920, 1080, 2, work="original")
    assert job.model == "realesr_general_x4v3" and job.sr_scale == 2 and job.work_div == 1


def test_1080p_half_detail_runs_ai_at_540p_and_4x():
    job = choose_job(_ultra_5070(), 1920, 1080, 2, work="half")
    assert (job.work_w, job.work_h, job.sr_scale, job.out_w, job.out_h) == (960, 540, 4, 3840, 2160)
    assert job.model == "realesr_general_x4v3"


def test_1x_restore_with_half_work_uses_ai():
    job = choose_job(_ultra_5070(), 1920, 1080, 1, work="half")
    assert job.sr_scale == 2 and (job.out_w, job.out_h) == (1920, 1080) and job.model is not None


def test_4x_never_combines_with_half_work():
    job = choose_job(_ultra_5070(), 1920, 1080, 4, work="half")
    assert job.work_div == 1 and job.sr_scale == 4


def test_economy_always_compact_and_forced_model_checks_scale():
    p = _plan(GPUInfo("NVIDIA GeForce GTX 1660 SUPER", Vendor.NVIDIA, 6144, (7, 5)), ["DmlExecutionProvider"])
    assert choose_job(p, 720, 480, 2, work="original").model == "realesr_general_x4v3"
    with pytest.raises(ValueError):
        choose_job(p, 720, 480, 4, work="original", model="realesrgan_x2plus")


def test_plan_memory_budget_from_vram():
    p = _ultra_5070()
    assert 6000 < p.mem_budget_mb < 8000 and 10000 < p.mem_limit_mb < 11000


def test_half_psnr_separates_sd_upscaled_from_native_detail():
    from PIL import Image
    rng = np.random.default_rng(0)
    native = rng.integers(0, 256, (216, 384)).astype(np.float32)
    small = Image.fromarray(rng.integers(0, 256, (108, 192)).astype(np.uint8))
    upscaled = np.asarray(small.resize((384, 216), Image.BICUBIC)).astype(np.float32)
    # ruído branco ampliado ainda tem energia perto do limite da meia resolução; o limiar real (40 dB)
    # é conferido com vídeo no teste seguinte
    assert half_psnr(native) < 20 and half_psnr(upscaled) > half_psnr(native) + 10


@needs_ffmpeg
def test_detail_detection_on_upscaled_clip(tmp_path):
    clip = tmp_path / "sd_em_hd.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=10",
                    "-t", "1", "-vf", "noise=alls=10:allf=t,scale=1280:720:flags=bicubic",
                    "-c:v", "libx264", "-crf", "18", str(clip)], check=True)
    assert detail_psnr(str(clip), 1280, 720, 1.0, samples=2) > 40


# ---------------- estágio uint8 ----------------
def _u8(w=130, h=90, n=3):
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([np.sin(xx / 7) * 90 + 120, np.cos(yy / 5) * 90 + 120, (xx + yy) % 256], -1)
    return np.clip(base[None] + rng.normal(0, 4, (n, h, w, 3)), 0, 255).astype(np.uint8)


def test_whole_tiles_and_tiny_budget_are_identical_for_nearest(tmp_path):
    sr_model_u8(tmp_path / "n.onnx", 4)
    frames = _u8()
    ref = frames.repeat(4, 1).repeat(4, 2)
    whole = OnnxUpscale(tmp_path / "n.onnx", 4, Backend.CPU, tile=0)
    tiles = OnnxUpscale(tmp_path / "n.onnx", 4, Backend.CPU, tile=40)
    tiny = OnnxUpscale(tmp_path / "n.onnx", 4, Backend.CPU, mem_budget_mb=1)
    assert whole.uint8_io and whole.model_scale == 4
    assert np.array_equal(whole.process(frames), ref)
    assert np.array_equal(tiles.process(frames), ref) and tiles.tiles is not None
    assert np.array_equal(tiny.process(frames), ref) and tiny.tiles is not None


def test_2x_from_4x_model_is_box_filter_inside_graph(tmp_path):
    sr_model_u8(tmp_path / "n.onnx", 4)
    frames = _u8(n=2)
    out = OnnxUpscale(tmp_path / "n.onnx", 2, Backend.CPU, tile=0).process(frames)
    assert out.shape == (2, 180, 260, 3)
    assert np.array_equal(out, frames.repeat(2, 1).repeat(2, 2))  # nearest 4x + média 2x2 = nearest 2x


def test_several_frames_per_call(tmp_path):
    sr_model_u8(tmp_path / "n.onnx", 2)
    st = OnnxUpscale(tmp_path / "n.onnx", 2, Backend.CPU, mem_budget_mb=512, max_batch=8)
    st.process(_u8(n=5))
    assert st.tiles is None and st.per_run == 8


def test_odd_sizes_are_padded_and_cropped(tmp_path):
    sr_model_u8(tmp_path / "n.onnx", 2)
    frames = _u8(w=131, h=91, n=2)
    for tile in (0, 50):
        out = OnnxUpscale(tmp_path / "n.onnx", 2, Backend.CPU, tile=tile).process(frames)
        assert np.array_equal(out, frames.repeat(2, 1).repeat(2, 2))


def test_fp16_overflow_switches_to_fp32(tmp_path):
    sr_model_u8(tmp_path / "m.fp16.onnx", 2, fp16=True, broken=True)
    sr_model_u8(tmp_path / "m.onnx", 2)
    st = OnnxUpscale(tmp_path / "m.fp16.onnx", 2, Backend.CPU, tile=0)
    frames = _u8(n=1)
    out = st.process(frames)
    assert st.model_path.name == "m.onnx" and np.array_equal(out, frames.repeat(2, 1).repeat(2, 2))


@needs("RealESRGAN_x2plus.pth")
def test_x2plus_matches_numpy_reference(tmp_path):
    import onnxruntime as ort
    sd = state_dict(WEIGHTS / "RealESRGAN_x2plus.pth")
    assert realesrgan.model_info(sd) == ("rrdbnet", 2)
    realesrgan.export(sd, tmp_path / "x2.onnx")
    fr = np.random.default_rng(0).integers(0, 256, (1, 6, 8, 3), dtype=np.uint8)
    y = ort.InferenceSession(str(tmp_path / "x2.onnx"), providers=["CPUExecutionProvider"]).run(
        ["out"], {"frames": fr, "out_scale": np.array([1, 1, 1, 1], np.float32)})[0]
    ref = realesrgan.run_rrdbnet_numpy(sd, fr[0].transpose(2, 0, 1).astype(np.float32) / 255)
    ref = np.clip(np.round(ref * 255), 0, 255).transpose(1, 2, 0)
    assert y.shape == (1, 12, 16, 3) and np.abs(y[0].astype(int) - ref).max() <= 1


@needs("realesr-general-x4v3.pth")
def test_compact_u8_matches_float_reference(tmp_path):
    sd = state_dict(WEIGHTS / "realesr-general-x4v3.pth")
    realesrgan.export(sd, tmp_path / "c.onnx")
    fr = _u8(24, 20, 1)
    out = OnnxUpscale(tmp_path / "c.onnx", 4, Backend.CPU, tile=0).process(fr)[0]
    ref = realesrgan.run_compact_numpy(sd, fr[0].transpose(2, 0, 1).astype(np.float32) / 255)
    ref = np.clip(np.round(ref * 255), 0, 255).transpose(1, 2, 0)
    assert np.abs(out.astype(int) - ref).max() <= 1


@needs_ffmpeg
def test_pipeline_with_half_work_size(tmp_path):
    import json
    clip = tmp_path / "hd.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10",
                    "-f", "lavfi", "-i", "sine=duration=1", "-t", "1", "-c:v", "libx264", "-c:a", "pcm_s16le",
                    "-shortest", str(clip)], check=True)
    sr_model_u8(tmp_path / "n.onnx", 4)
    out = tmp_path / "o.mkv"
    Pipeline(PipelineConfig(clip, out, [OnnxUpscale(tmp_path / "n.onnx", 4, Backend.CPU)],
                            work_size=(160, 120))).run()
    d = json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(out)],
                                  capture_output=True, text=True).stdout)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (640, 480)
    assert any(s["codec_type"] == "audio" for s in d["streams"])
