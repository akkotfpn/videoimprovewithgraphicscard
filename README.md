# videoimprovewithgraphicscard

Restauração e upscaling de vídeos antigos (fitas VHS digitalizadas, Hi8, filmes caseiros) que se adapta ao hardware: usa TensorRT e modelos pesados numa RTX 50, e cai automaticamente para modelos leves, tiling e filtros clássicos em GPUs antigas ou só na CPU.

> Estado: **fase 0**. A engine em Python já detecta o hardware, escolhe o preset e roda o pipeline completo (desentrelaçar → denoise → upscale → interpolação → encode com áudio original). Os modelos de IA reais entram na fase 1. Veja o [planejamento completo](docs/ARQUITETURA.md).

## Requisitos

- Python 3.11+
- FFmpeg e ffprobe no PATH
- Opcional: `onnxruntime-gpu` (NVIDIA), `onnxruntime-directml` (Windows, qualquer GPU) ou `onnxruntime-openvino` (Intel)

## Uso rápido

```bash
pip install -e ".[dev]"            # ou .[nvidia] / .[directml] / .[openvino]

# o que o programa enxerga na sua máquina e qual preset escolheria
python -m engine detect

# upscale 2x + dobrar FPS (sem modelo: redimensionamento clássico, serve para testar)
python -m engine run fita.avi fita_restaurada.mkv --scale 2 --interpolate

# com um modelo ONNX de super-resolução
python -m engine run fita.avi saida.mkv --scale 4 --model models/realesr-general-x4v3.onnx

# forçar preset e tamanho de tile
python -m engine run fita.avi saida.mkv --preset economico --tile 256
```

Use `.mkv` na saída para copiar o áudio original sem recompressão.

## Presets

| Preset | Para quem | Backend |
|---|---|---|
| Ultra | RTX 40/50 com 12 GB+ | TensorRT FP8/FP16 |
| Equilibrado | GPUs com 8 GB+ | CUDA / DirectML / ROCm / OpenVINO, FP16 |
| Econômico | GPUs antigas ou com pouca VRAM | NCNN-Vulkan / DirectML, tiling |
| CPU | sem GPU utilizável | ONNX Runtime CPU + filtros FFmpeg |

## Testes

```bash
python -m pytest
```

Os testes simulam várias placas (RTX 5090, 4070, 3060, Radeon, GTX 1050, Intel Arc sem runtime) e processam um vídeo VHS sintético entrelaçado com áudio, verificando resolução, contagem de frames, espaço de cor e sincronia do áudio.

## Licença

Código sob [Apache-2.0](LICENSE). Pesos de modelos têm licenças próprias e não fazem parte deste repositório; veja [models/README.md](models/README.md).
