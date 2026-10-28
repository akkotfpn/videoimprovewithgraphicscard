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
    assert (p.preset, p.backend, p.precision, p.tile_size) == (Preset.ULTRA, Backend.TENSORRT, "fp16", 0)
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


# ---------------- ONNX Runtime de verdade (modelos mínimos gerados no teste) ----------------
from engine.stages.upscale import OnnxUpscale  # noqa: E402
from engine.models.onnx_writer import sr_model  # noqa: E402


@pytest.fixture
def models(tmp_path):
    sr_model(tmp_path / "sr.onnx", 4)
    sr_model(tmp_path / "sr.fp16.onnx", 4, fp16=True, broken=True)
    return tmp_path


def _frames(n=2, h=100, w=130):
    return np.random.default_rng(1).integers(0, 255, (n, h, w, 3), dtype=np.uint8)


def test_onnx_tiled_matches_reference_and_probes_scale(models):
    st = OnnxUpscale(models / "sr.onnx", 4, Backend.CPU, tile=32, overlap=8)
    assert st.model_scale == 4
    f = _frames()
    out = st.process(f)
    assert out.shape == (2, 400, 520, 3)
    assert np.array_equal(out, f.repeat(4, 1).repeat(4, 2))  # tiles costurados sem emenda


def test_onnx_4x_model_with_2x_output(models):
    st = OnnxUpscale(models / "sr.onnx", 2, Backend.CPU, tile=64)
    assert st.process(_frames(1)).shape == (1, 200, 260, 3)


def test_fp16_nan_falls_back_to_fp32(models):
    st = OnnxUpscale(models / "sr.fp16.onnx", 4, Backend.CPU, tile=0)
    out = st.process(_frames(1, 20, 24))
    assert st.dtype == np.float32 and st.model_path.name == "sr.onnx"
    assert out.shape == (1, 80, 96, 3)


def test_fp16_nan_without_fp32_sibling_raises(tmp_path):
    sr_model(tmp_path / "only.fp16.onnx", 2, fp16=True, broken=True)
    st = OnnxUpscale(tmp_path / "only.fp16.onnx", 2, Backend.CPU, tile=0)
    with pytest.raises(RuntimeError, match="NaN"):
        st.process(_frames(1, 8, 8))


# ---------------- PCs medianos e a 5070 ----------------
def test_rtx5070_is_ultra():
    g = GPUInfo("NVIDIA GeForce RTX 5070", Vendor.NVIDIA, 12227, (12, 0))
    p = choose_plan(hw(g, ["CUDAExecutionProvider"]))
    assert p.preset == Preset.ULTRA and p.backend == Backend.CUDA and p.tile_size == 768


def test_rtx5070_with_only_directml_drops_to_balanced():
    g = GPUInfo("NVIDIA GeForce RTX 5070", Vendor.NVIDIA, 12227, (12, 0))
    p = choose_plan(hw(g, ["DmlExecutionProvider"]))
    assert p.preset == Preset.EQUILIBRADO and p.backend == Backend.DIRECTML


def test_gtx1660_has_no_tensor_cores_and_uses_directml():
    g = GPUInfo("NVIDIA GeForce GTX 1660 SUPER", Vendor.NVIDIA, 6144, (7, 5))
    assert not g.has_tensor_cores
    p = choose_plan(hw(g, ["DmlExecutionProvider"]))
    assert p.preset == Preset.ECONOMICO and p.backend == Backend.DIRECTML
    assert p.sr_model == "realesr_general_x4v3" and p.tile_size == 384


def test_rtx2060_6gb_is_balanced():
    g = GPUInfo("NVIDIA GeForce RTX 2060", Vendor.NVIDIA, 6144, (7, 5))
    assert choose_plan(hw(g, ["CUDAExecutionProvider"])).preset == Preset.EQUILIBRADO


@pytest.mark.parametrize("name,vendor", [
    ("Intel(R) Iris(R) Xe Graphics", Vendor.INTEL),
    ("Intel(R) UHD Graphics 630", Vendor.INTEL),
    ("Intel(R) Arc(TM) Graphics", Vendor.INTEL),        # Meteor Lake integrada
    ("AMD Radeon(TM) Graphics", Vendor.AMD),            # Ryzen APU genérico
    ("AMD Radeon 780M Graphics", Vendor.AMD),
])
def test_integrated_gpus_get_economy_with_shared_memory(name, vendor):
    g = GPUInfo(name, vendor, 128)  # o Windows reporta só a reserva dedicada
    assert g.integrated
    p = choose_plan(hw(g, ["DmlExecutionProvider"]))
    assert p.preset == Preset.ECONOMICO and p.backend == Backend.DIRECTML
    assert 128 <= p.tile_size <= 384


@pytest.mark.parametrize("name,vendor", [
    ("Intel(R) Arc(TM) A770 Graphics", Vendor.INTEL),
    ("Intel(R) Arc(TM) B580 Graphics", Vendor.INTEL),
    ("AMD Radeon RX 6600", Vendor.AMD),
])
def test_dedicated_not_marked_integrated(name, vendor):
    assert not GPUInfo(name, vendor, 8192).integrated


def test_laptop_prefers_dedicated_over_integrated():
    h = hw(None, ["DmlExecutionProvider"])
    h.gpus = [GPUInfo("Intel(R) Iris(R) Xe Graphics", Vendor.INTEL, 128),
              GPUInfo("NVIDIA GeForce GTX 1650", Vendor.NVIDIA, 4096, (7, 5))]
    assert h.best_gpu.name.endswith("1650")



# ---------------- blocos balanceados ----------------
from engine.core.tiling import balanced_tile  # noqa: E402


def test_balanced_tile_for_vhs_frame():
    assert balanced_tile(720, 480, 384) == (240, 360)   # 2x2 blocos, sem sobra
    assert balanced_tile(720, 480, 0) == (0, 0)
    assert balanced_tile(720, 480, 1000) == (480, 720)  # cabe inteiro


def test_cpu_preset_does_not_tile():
    assert choose_plan(hw(None)).tile_size == 0


# ---------------- modelos reais (sem PyTorch) ----------------
import pickle  # noqa: E402
import zipfile  # noqa: E402

from engine.models import catalog, realesrgan  # noqa: E402
from engine.models.pth_loader import load_pth, state_dict  # noqa: E402

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "_weights"
needs_weights = pytest.mark.skipif(not (WEIGHTS / "realesr-general-x4v3.pth").exists(),
                                   reason="rode: python -m engine models install all")


def test_pth_loader_refuses_malicious_pickle(tmp_path):
    class Evil:
        def __reduce__(self):
            import os
            return (os.system, ("echo pwned",))
    bad = tmp_path / "evil.pth"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps(Evil()))
    with pytest.raises(pickle.UnpicklingError):
        load_pth(bad)


@needs_weights
def test_compact_onnx_matches_numpy_reference(tmp_path):
    import onnxruntime as ort
    sd = state_dict(WEIGHTS / "realesr-general-x4v3.pth")
    assert realesrgan.detect_arch(sd) == "srvgg_compact"
    realesrgan.export(sd, tmp_path / "c.onnx")
    img = np.random.default_rng(0).random((3, 10, 12), dtype=np.float32)
    sess = ort.InferenceSession(str(tmp_path / "c.onnx"), providers=["CPUExecutionProvider"])
    y = sess.run(None, {"input": img[None]})[0][0]
    assert y.shape == (3, 40, 48)
    assert np.abs(y - realesrgan.run_compact_numpy(sd, img)).max() < 1e-4


@needs_weights
def test_rrdbnet_exports_and_upscales(tmp_path):
    import onnxruntime as ort
    sd = state_dict(WEIGHTS / "RealESRGAN_x4plus.pth")
    assert realesrgan.detect_arch(sd) == "rrdbnet"
    realesrgan.export(sd, tmp_path / "x.onnx")
    yy, xx = np.mgrid[0:24, 0:24] / 24.0
    img = (np.stack([np.sin(3 * xx), np.cos(2 * yy), xx * yy]) * 0.4 + 0.5).astype(np.float32)
    y = ort.InferenceSession(str(tmp_path / "x.onnx"), providers=["CPUExecutionProvider"]).run(
        None, {"input": img[None]})[0][0]
    psnr = 10 * np.log10(1 / np.mean((np.clip(y, 0, 1) - img.repeat(4, 1).repeat(4, 2)) ** 2))
    assert y.shape == (3, 96, 96) and psnr > 28


@needs_weights
def test_catalog_install_from_local_weights(tmp_path):
    out = catalog.install("realesr_general_x4v3", models_dir=tmp_path, denoise=0.3,
                          weights=[WEIGHTS / "realesr-general-x4v3.pth", WEIGHTS / "realesr-general-wdn-x4v3.pth"])
    assert [p.name for p in out] == ["realesr_general_x4v3.onnx", "realesr_general_x4v3.fp16.onnx"]
    st = OnnxUpscale(out[0], 2, Backend.CPU, tile=0)
    assert st.model_scale == 4
    assert st.process(_frames(1, 24, 32)).shape == (1, 48, 64, 3)



# ---------------- PC real do desenvolvedor: RTX 5070 + Radeon integrada do Ryzen ----------------
def _dev_pc(providers):
    h = hw(None, providers, vulkan=True, hwaccels=["cuda"])
    h.gpus = [GPUInfo("NVIDIA GeForce RTX 5070", Vendor.NVIDIA, 12227, (12, 0)),
              GPUInfo("AMD Radeon(TM) Graphics", Vendor.AMD, 512)]
    h.ram_mb = 31830
    return h


def test_dev_pc_defaults_to_5070():
    p = choose_plan(_dev_pc(["CUDAExecutionProvider"]))
    assert p.preset == Preset.ULTRA and p.backend == Backend.CUDA


def test_dev_pc_igpu_via_directml_simulates_mid_range_pc():
    p = choose_plan(_dev_pc(["DmlExecutionProvider"]), gpu_index=1)
    assert p.preset == Preset.ECONOMICO and p.backend == Backend.DIRECTML
    assert p.sr_model == "realesr_general_x4v3"


def test_dev_pc_igpu_without_directml_warns_about_ncnn():
    p = choose_plan(_dev_pc(["CUDAExecutionProvider"]), gpu_index=1)
    assert p.backend == Backend.NCNN_VULKAN
    assert any("NCNN" in r for r in p.reasons)


def test_invalid_gpu_index():
    with pytest.raises(ValueError):
        choose_plan(_dev_pc([]), gpu_index=5)


def test_hardware_notes_flow_into_plan_reasons():
    h = _dev_pc(["CUDAExecutionProvider"])
    h.notes = ["TensorRT não instalado"]
    assert "TensorRT não instalado" in choose_plan(h).reasons


def test_ultra_without_realbasicvsr_falls_back_to_x4plus(tmp_path, monkeypatch):
    from engine import cli
    (tmp_path / "realesrgan_x4plus.onnx").write_bytes(b"x")
    (tmp_path / "realesr_general_x4v3.onnx").write_bytes(b"x")
    monkeypatch.setenv("VIDEOIMPROVE_MODELS", str(tmp_path))
    p = choose_plan(_dev_pc(["CUDAExecutionProvider"]))
    assert cli.resolve_model(None, p).name.startswith("realesrgan_x4plus")


# ---------------- app: 60 fps por campo, prévia, autoteste ----------------
@needs_ffmpeg
def test_field_rate_deinterlace_doubles_with_real_motion(vhs_sample, tmp_path):
    src_v, _ = _streams(vhs_sample)
    out = tmp_path / "60p.mkv"
    Pipeline(PipelineConfig(vhs_sample, out, [ResizeUpscale(2)], batch_size=4, field_rate=True)).run()
    v, a = _streams(out)
    from fractions import Fraction
    assert abs(float(Fraction(v["r_frame_rate"])) - 60000 / 1001) < 0.01
    assert abs(int(v["nb_read_frames"]) - 2 * int(src_v["nb_read_frames"])) <= 1
    assert a["codec_name"] == "pcm_s16le"


@needs_ffmpeg
def test_cli_preview_writes_same_size_images(vhs_sample, tmp_path):
    from PIL import Image
    from engine import cli
    assert cli.main(["preview", str(vhs_sample), "--out", str(tmp_path), "--time", "1", "--preset", "cpu"]) == 0
    a, b = Image.open(tmp_path / "antes.png"), Image.open(tmp_path / "depois.png")
    assert a.size == b.size == (1440, 960)


@needs_ffmpeg
def test_cli_run_smooth_on_interlaced_uses_field_rate(vhs_sample, tmp_path, capsys):
    from engine import cli
    out = tmp_path / "s.mkv"
    assert cli.main(["run", str(vhs_sample), str(out), "--preset", "cpu", "--smooth", "--json"]) == 0
    events = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    start = next(e for e in events if e["event"] == "start")
    assert start["field_rate"] is True and "blend_x2" not in start["stages"]
    prog = [e for e in events if e["event"] == "progress"]
    assert prog[-1]["total"] == 180 and "eta" in prog[-1]
    assert events[-1]["event"] == "done"


def test_denoise_levels_map_to_filters():
    from engine import cli
    p = choose_plan(hw(None))
    assert cli.denoise_filter("off", p) == []
    assert cli.denoise_filter("strong", p)[0].startswith("nlmeans")
    assert cli.denoise_filter("auto", p) == ["hqdn3d=3:2:6:4"]


def test_plan_device_id_follows_runtime():
    g = GPUInfo("NVIDIA GeForce RTX 5070", Vendor.NVIDIA, 12227, (12, 0), dxgi_index=1, cuda_index=0)
    assert choose_plan(hw(g, ["DmlExecutionProvider"])).device_id == 1
    assert choose_plan(hw(g, ["CUDAExecutionProvider"])).device_id == 0
