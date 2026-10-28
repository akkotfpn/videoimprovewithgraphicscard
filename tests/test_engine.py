import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from engine.core.hardware import Backend, GPUInfo, HardwareProfile, Preset, Vendor, choose_plan
from engine.core.pipeline import Pipeline, PipelineConfig, probe
from engine.core.tiling import process_tiled
from engine.stages.interpolate import BlendInterpolate
from engine.stages.upscale import ResizeUpscale


def hw(gpu=None, ort=(), vulkan=False, torch=None, hwaccels=()):
    return HardwareProfile("test", "cpu", 16, 32000, [gpu] if gpu else [], torch,
                           list(ort) + ["CPUExecutionProvider"], vulkan, list(hwaccels))


# ---------------- escolha de perfil ----------------
def test_rtx5090_tensorrt_is_ultra_fp8():
    g = GPUInfo("NVIDIA GeForce RTX 5090", Vendor.NVIDIA, 32607, (12, 0))
    p = choose_plan(hw(g, ["TensorrtExecutionProvider", "CUDAExecutionProvider"], hwaccels=["cuda"]))
    assert (p.preset, p.backend, p.precision, p.tile_size) == (Preset.ULTRA, Backend.TENSORRT, "fp8", 0)
    assert g.arch == "Blackwell (RTX 50)" and p.nvenc


def test_rtx4070_without_tensorrt_falls_to_cuda_fp16():
    g = GPUInfo("NVIDIA GeForce RTX 4070", Vendor.NVIDIA, 12282, (8, 9))
    p = choose_plan(hw(g, ["CUDAExecutionProvider"]))
    assert p.preset == Preset.ULTRA and p.backend == Backend.CUDA and p.precision == "fp16"
    assert p.tile_size == 768


def test_rtx3060_8gb_is_balanced():
    g = GPUInfo("NVIDIA GeForce RTX 3060", Vendor.NVIDIA, 8192, (8, 6))
    p = choose_plan(hw(g, ["CUDAExecutionProvider"]))
    assert p.preset == Preset.EQUILIBRADO and p.tile_size == 512


def test_amd_on_windows_uses_directml():
    g = GPUInfo("AMD Radeon RX 7800 XT", Vendor.AMD, 16384)
    p = choose_plan(hw(g, ["DmlExecutionProvider"]))
    assert p.preset == Preset.EQUILIBRADO and p.backend == Backend.DIRECTML


def test_old_gpu_with_vulkan_only_is_economy_ncnn():
    g = GPUInfo("NVIDIA GeForce GTX 1050", Vendor.NVIDIA, 2048, (6, 1))
    p = choose_plan(hw(g, vulkan=True))
    assert p.preset == Preset.ECONOMICO and p.backend == Backend.NCNN_VULKAN and p.tile_size <= 256


def test_gpu_without_any_runtime_is_downgraded_to_cpu():
    g = GPUInfo("Intel Arc A770", Vendor.INTEL, 16384)
    p = choose_plan(hw(g))
    assert p.preset == Preset.CPU and p.backend == Backend.CPU


# ---------------- tiling ----------------
def test_tiling_matches_full_frame():
    rng = np.random.default_rng(0)
    img = rng.random((100, 130, 3), dtype=np.float32)
    up = lambda t: t.repeat(2, 0).repeat(2, 1)
    assert np.allclose(process_tiled(img, up, 2, tile=32, overlap=8), up(img))


# ---------------- interpolação ----------------
def test_interpolation_doubles_exactly_across_batches():
    st = BlendInterpolate(); st.setup(4, 4)
    frames = np.arange(7, dtype=np.uint8)[:, None, None, None].repeat(4, 1).repeat(4, 2).repeat(3, 3) * 10
    out = [st.process(frames[:3]), st.process(frames[3:])]
    out.append(st.flush())
    assert sum(len(o) for o in out) == 14


# ---------------- ponta a ponta ----------------
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg ausente")


@pytest.fixture(scope="module")
def vhs_sample(tmp_path_factory):
    """Simula uma fita: 720x480 entrelaçado, 29.97 fps, 3 s, com ruído e áudio estéreo."""
    out = tmp_path_factory.mktemp("v") / "fita.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=720x480:rate=30000/1001:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3:sample_rate=48000",
        "-vf", "noise=alls=20:allf=t,tinterlace=mode=interleave_top,fps=30000/1001,setfield=tff",
        "-c:v", "libx264", "-flags", "+ildct+ilme", "-pix_fmt", "yuv420p",
        "-ac", "2", "-c:a", "pcm_s16le", "-shortest", str(out)], check=True)
    return out


def _streams(path):
    d = json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-count_frames",
                                   "-show_streams", str(path)], capture_output=True, text=True).stdout)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    a = next(s for s in d["streams"] if s["codec_type"] == "audio")
    return v, a


@needs_ffmpeg
def test_probe_detects_interlaced(vhs_sample):
    info = probe(vhs_sample)
    assert info.interlaced and info.has_audio and info.width == 720


@needs_ffmpeg
def test_pipeline_upscale_interpolate_keeps_audio_sync(vhs_sample, tmp_path):
    src_v, src_a = _streams(vhs_sample)
    out = tmp_path / "restaurado.mkv"
    Pipeline(PipelineConfig(vhs_sample, out, [ResizeUpscale(2), BlendInterpolate()],
                            batch_size=4, pre_filters=["hqdn3d=3:2:6:4"])).run()
    v, a = _streams(out)
    assert (v["width"], v["height"]) == (1440, 960)
    assert int(v["nb_read_frames"]) == 2 * int(src_v["nb_read_frames"])
    from fractions import Fraction
    # MKV usa base de tempo em ms, então 60000/1001 pode voltar como 19001/317 (= 59,94)
    assert abs(float(Fraction(v["r_frame_rate"])) - 60000 / 1001) < 0.01
    assert v.get("color_space") == "bt709"
    assert a["codec_name"] == "pcm_s16le"  # áudio copiado sem recompressão
    # duração do vídeo e do áudio iguais (tolerância de 1 frame)
    vd = int(v["nb_read_frames"]) * 1001 / 60000
    ad = float(a.get("duration") or 0) or float(src_a.get("duration") or 3.0)
    assert abs(vd - ad) < 1 / 29.97 + 1e-3
