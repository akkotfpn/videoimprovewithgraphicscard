# Arquitetura e planejamento

Restauração e upscaling de vídeos antigos (VHS, Hi8, MiniDV, filmes caseiros) com um pipeline que se adapta à GPU disponível: do topo de linha (RTX 50 com TensorRT) até máquinas só com CPU.

> Lista de modelos reflete o estado do ecossistema open-source em meados de 2026. É uma área que muda rápido: antes de fixar um modelo na release, confira a versão mais nova e a licença dos pesos.

---

## 1. Viabilidade

É viável e as peças existem em código aberto. Os riscos reais não são de IA, são de engenharia:

| Risco | Mitigação |
|---|---|
| Tamanho da distribuição (PyTorch + CUDA passa de 3 GB) | Inferência em **ONNX Runtime**; PyTorch só no desenvolvimento para exportar modelos. Runtimes de GPU baixados sob demanda por fabricante. |
| Licenças de pesos (CodeFormer é não comercial) | Pesos nunca vêm no instalador; download opcional com a licença exibida. |
| Tempo de processamento (2 h de fita no Ultra pode levar horas) | Processamento em segmentos com retomada, fila de jobs, prévia de 10 s antes de processar tudo. |
| Expectativa do usuário (IA "inventa" rostos) | Controle de intensidade da restauração facial e comparação lado a lado. |

---

## 2. Arquitetura do software

### 2.1 Stack recomendada

```
┌──────────────────────────────────────────────────────────┐
│  UI: Tauri 2 (Rust) + React + TypeScript + Vite          │
│  - fila de jobs, prévia antes/depois, presets            │
│  - instala/atualiza "runtime packs" e modelos            │
└───────────────┬──────────────────────────────────────────┘
                │ processo filho (sidecar)
                │ stdin: comandos JSON / stdout: eventos JSON por linha
┌───────────────▼──────────────────────────────────────────┐
│  Engine: Python 3.12 (este repositório, pasta engine/)   │
│  - hardware.py: detecção e plano de execução             │
│  - pipeline.py: FFmpeg ⇄ NumPy ⇄ estágios                │
│  - stages/: SR, rosto, interpolação, denoise             │
├──────────────────────────────────────────────────────────┤
│  Runtimes: ONNX Runtime (TensorRT/CUDA/DirectML/MIGraphX/ │
│            OpenVINO/CPU) · NCNN-Vulkan · FFmpeg           │
└──────────────────────────────────────────────────────────┘
```

**Por que Tauri e não Electron:** instalador de ~10 MB contra ~150 MB, menos RAM ocupada enquanto a GPU trabalha, e o Rust cuida bem de processos filhos e downloads. Electron só valeria a pena se a equipe já tiver muito código Node.

**Por que não PySide6 direto:** funciona e é a alternativa mais simples (tudo em Python, um processo só). O problema é acoplar UI e engine: um travamento do driver de GPU derruba a janela junto. Com sidecar, a UI sobrevive, mostra o erro e oferece tentar com preset mais leve. Se quiser começar mais rápido, PySide6 é aceitável no MVP desde que a engine continue sendo um processo separado.

**Por que Python e não C++ na engine:** todo o ecossistema de modelos está em Python; o trabalho pesado já roda em C++/CUDA dentro do ONNX Runtime, TensorRT e FFmpeg. Python só orquestra. Se no futuro o overhead aparecer no profiler (cópia de frames), dá para mover o laço central para Rust com `ort` + `ffmpeg-next` sem trocar a arquitetura.

### 2.2 Protocolo UI ⇄ engine

Já implementado parcialmente em `engine/cli.py` com `--json`:

```json
{"event": "progress", "done": 1200, "total": 215784, "fps": 14.2}
{"event": "done", "output": "C:/Videos/natal-1994_restaurado.mkv"}
```

Próximos eventos: `stage` (qual estágio está rodando), `preview` (caminho de um PNG de comparação), `warning` (ex.: tile reduzido por falta de VRAM), `error` com código e sugestão.

### 2.3 Pipeline de processamento

```
 entrada.avi
     │
 [FFmpeg decode] ── filtros clássicos na própria decodificação:
     │              bwdif (desentrelaçar), crop do overscan/ruído de troca de cabeça,
     │              hqdn3d/nlmeans (ruído), correção de chroma shift
     │  rawvideo RGB24 por pipe (sem PNG em disco)
     ▼
 [Thread leitora] → fila limitada (lotes de N frames)
     ▼
 [Estágios em sequência na thread principal]
     1. Denoise neural (opcional)       SCUNet / FastDVDnet
     2. Super-resolução                  RealBasicVSR | Real-ESRGAN | compact
     3. Rosto                            detecta → recorta → GFPGAN/CodeFormer → cola com máscara
     4. Interpolação                     RIFE (retém 1 frame entre lotes)
     ▼
 fila limitada → [Thread escritora]
     ▼
 [FFmpeg encode] ← entrada original como 2ª fonte (-map 1:a, -c:a copy)
     │              BT.601 → BT.709, aspect 4:3 preservado
     ▼
 saída.mkv
```

Decisões importantes:

- **Pipe em memória, não extração para disco.** Duas horas de fita são ~216 mil frames. Em PNG isso ocupa centenas de GB. Com pipes, decode, inferência e encode rodam em paralelo e a fila limitada controla o uso de RAM.
- **Sincronia de áudio por construção.** Nenhum frame é descartado e a taxa de saída é exatamente a de entrada vezes o fator de interpolação (fração racional, ex. `60000/1001`). O áudio é copiado do arquivo original. O teste `test_pipeline_upscale_interpolate_keeps_audio_sync` verifica isso.
- **Desentrelaçar antes de tudo.** VHS é entrelaçado; mandar campos misturados para uma rede de SR gera o efeito "pente" ampliado 4×. `bwdif mode=0` mantém 1 frame por frame; `mode=1` (dobra o FPS) é uma opção mais fiel ao movimento original e dispensa a interpolação neural em muitos casos. O padrão-ouro é o QTGMC via VapourSynth, que pode entrar como estágio opcional.
- **Cor e proporção.** Material SD usa matriz BT.601 e pixels não quadrados (720×480 exibido como 4:3). A saída HD precisa ser convertida para BT.709 e marcada com o aspect correto, senão as cores ficam esverdeadas e a imagem esticada. Já implementado.
- **Interpolação por último.** Interpolar antes do SR dobra o custo do SR. Interpolar depois custa mais por frame no RIFE, mas o RIFE é bem mais barato que o SR. No preset Econômico, a ordem pode inverter (interpolar em baixa resolução) para caber no tempo.
- **Processamento em segmentos (roadmap).** Dividir em blocos de ~60 s, codificar cada um em arquivo temporário e concatenar com `-f concat -c copy` no fim. Permite retomar depois de queda de energia ou travamento de driver, e paralelizar entre duas GPUs.

### 2.4 Contrato de um estágio

```python
class Stage(Protocol):
    name: str
    scale: int        # fator espacial
    fps_factor: int   # fator temporal
    def setup(self, width: int, height: int) -> None: ...
    def process(self, batch: np.ndarray) -> np.ndarray: ...  # NHWC uint8 RGB
    def flush(self) -> np.ndarray | None: ...  # frames retidos por estágios temporais
```

Estágios temporais (RIFE, RealBasicVSR, FastDVDnet) guardam frames entre lotes e os devolvem no `flush()`. `BlendInterpolate` em `engine/stages/interpolate.py` é a implementação de referência desse contrato.

---

## 3. Modelos e algoritmos

### 3.1 Super-resolução e nitidez

| Modelo | Uso | Peso | Licença | Preset |
|---|---|---|---|---|
| **RealBasicVSR** | SR de vídeo com memória temporal, treinado para degradações reais. Melhor escolha para VHS: usa frames vizinhos e não "pisca". | pesado | Apache-2.0 | Ultra |
| **Real-ESRGAN x4plus** | SR de imagem, frame a frame. Muito testado, fácil de exportar para ONNX/TensorRT. | médio | BSD-3 | Equilibrado |
| **realesr-general-x4v3** (compact) | SRVGG compacto, com controle de força do denoise. Excelente custo/benefício. | leve | BSD-3 | Econômico / CPU |
| **Real-ESRGAN ncnn-vulkan** | Mesmo modelo, roda em qualquer GPU com Vulkan sem CUDA. | leve | MIT (binário) / BSD-3 (pesos) | Econômico |
| HAT / DAT / SPAN | Arquiteturas mais novas; HAT e DAT têm qualidade alta e custo alto; SPAN é rápido. Bons candidatos a avaliar depois do MVP. | variável | verificar por modelo | Ultra / Equilibrado |
| BSRGAN | Degradação sintética mais agressiva; útil como alternativa para fitas muito ruins. | médio | Apache-2.0 | Equilibrado |

A biblioteca **spandrel** (MIT) carrega pesos de dezenas de arquiteturas de SR com uma só API em PyTorch, o que facilita testar modelos e exportar para ONNX.

Dica prática: para VHS, upscaling de 2× costuma ficar mais natural que 4×. A fita tem cerca de 240 linhas de resolução horizontal real; ampliar 4× força a rede a inventar detalhe. Ofereça 2× como padrão e 4× como opção.

### 3.2 Restauração de rostos

| Modelo | Observações | Licença |
|---|---|---|
| **GFPGAN v1.4** | Rápido, resultado natural, bom padrão. | Apache-2.0 |
| **CodeFormer** | Melhor em rostos muito degradados e tem o controle de fidelidade `w` (0 = mais "inventado", 1 = mais fiel). | **S-Lab License 1.0, uso não comercial** |
| RestoreFormer++ | Alternativa ao CodeFormer. | verificar |
| facexlib (RetinaFace + parsing) | Detecção, alinhamento e máscara para colar o rosto de volta. | MIT |

Pontos de projeto:
- Aplicar **depois** do SR, no frame de saída.
- Suavizar o resultado no tempo (detectar o mesmo rosto em frames vizinhos e misturar), senão o rosto "treme" porque cada frame é restaurado de forma independente.
- Expor um controle de intensidade. Em vídeo de família, um rosto parecido mas não igual ao do parente é pior do que um rosto borrado.
- CodeFormer: não incluir no instalador; download opcional, com a licença exibida.

### 3.3 Interpolação de quadros

| Modelo | Observações | Licença |
|---|---|---|
| **RIFE 4.x (Practical-RIFE)** | Padrão de mercado; rápido, exporta para ONNX/TensorRT. | MIT |
| **rife-ncnn-vulkan** | Fallback para qualquer GPU Vulkan. | MIT |
| bwdif `mode=1` | Sem IA: desentrelaça gerando 59,94 fps reais a partir dos campos. Em VHS, muitas vezes basta. | (FFmpeg, LGPL) |

### 3.4 Ruído, granulação e defeitos de fita

| Técnica | Quando | Custo |
|---|---|---|
| `hqdn3d` (FFmpeg) | Ruído de croma e "chuvisco" leve. | muito baixo |
| `nlmeans` / `nlmeans_vulkan` (FFmpeg 7.x) | Ruído mais pesado; a versão Vulkan roda na GPU. | médio |
| **SCUNet** | Denoise neural cego, bom com ruído misto de fita. | Apache-2.0, médio |
| **FastDVDnet** | Denoise temporal (usa 5 frames), evita "fervura" do ruído. | verificar, médio |
| QTGMC (VapourSynth) | Melhor desentrelaçador disponível, também reduz ruído. | GPL, alto na CPU |
| Correções clássicas | Chroma shift (`chromashift`), crop do ruído de troca de cabeça nas últimas linhas, estabilização leve (`deshake`/`vidstab`), remoção de zumbido de 60 Hz no áudio (`afftdn`, `bandreject`). | baixo |

A ordem recomendada é: desentrelaçar → crop → denoise → SR → rosto → interpolação → cor/nitidez final.

---

## 4. Detecção de hardware e fallback

Implementado em `engine/core/hardware.py`. Rode `python -m engine detect` para ver o resultado.

### 4.1 Sondas, da mais confiável para a menos

1. **PyTorch** (se instalado): VRAM exata e compute capability (CUDA, ROCm, Intel XPU, Apple MPS).
2. **nvidia-smi**: nome, VRAM, driver e compute capability, sem precisar do PyTorch.
3. **rocm-smi**: AMD no Linux.
4. **Registro do Windows**: lê `HardwareInformation.qwMemorySize`, que traz a VRAM real. O `AdapterRAM` do WMI é inteiro de 32 bits e trava em 4 GB, então mente para qualquer placa moderna.
5. **vulkaninfo**: qualquer GPU com driver Vulkan (ignora renderizadores de software como llvmpipe).
6. **ONNX Runtime**: quais Execution Providers estão de fato instalados. Uma RTX 5090 sem o pacote `onnxruntime-gpu` é, para o programa, uma CPU.
7. **FFmpeg `-hwaccels` / `-encoders`**: decide NVENC/AMF/QSV no encode.

As sondas falham em silêncio e o resultado é mesclado, então a mesma placa vista por três fontes vira uma entrada só com o dado mais completo.

### 4.2 Identificação da geração NVIDIA

| Compute capability | Geração | Recursos |
|---|---|---|
| 12.0 | Blackwell (RTX 50) | FP16, FP8, FP4. Exige CUDA 12.8+ e builds com `sm_120`. |
| 8.9 | Ada (RTX 40) | FP16, FP8 |
| 8.6 | Ampere (RTX 30) | FP16, TF32 |
| 7.5 | Turing (RTX 20 / GTX 16) | FP16 |

### 4.3 Presets

| Preset | Condição | Backend | Precisão | Tile | Modelos |
|---|---|---|---|---|---|
| **Ultra** | NVIDIA com FP8 (≥ 8.9) e ≥ 12 GB | TensorRT (ou CUDA FP16 se TRT ausente) | FP8 / FP16 | sem tile se ≥ 16 GB, senão 768 | RealBasicVSR + CodeFormer/GFPGAN + RIFE + SCUNet |
| **Equilibrado** | ≥ 8 GB com FP16 | CUDA / DirectML / ROCm / OpenVINO | FP16 | 512 | Real-ESRGAN x4plus + GFPGAN + RIFE lite + nlmeans |
| **Econômico** | 2–8 GB ou só Vulkan | NCNN-Vulkan / DirectML | FP16 | 256 | compact x4v3 + rife-ncnn + hqdn3d |
| **CPU** | sem GPU ou sem runtime | ONNX Runtime CPU | INT8 / FP32 | 192 | compact x4v3 + filtros FFmpeg |

O usuário pode sempre forçar um preset (`--preset`), e a UI deve mostrar o motivo da escolha automática (campo `reasons` do plano).

### 4.4 Fallback em tempo de execução

A escolha inicial é uma estimativa. Durante a execução:
- **Falta de VRAM:** o estágio ONNX captura o erro, divide o tile pela metade e tenta de novo (já implementado em `OnnxUpscale._frame`).
- **Falha do provider:** se o TensorRT não compilar o engine, cair para CUDA; se CUDA falhar, DirectML; depois CPU. Registrar no log e avisar a UI.
- **Lentidão extrema:** se a estimativa de tempo restante passar de um limite, sugerir preset menor na UI.

### 4.5 Notas sobre FP8 e TensorRT

- O engine TensorRT é compilado por GPU e por versão do driver na primeira execução (pode levar vários minutos). Fica em cache (`trt_engine_cache_path`). A UI precisa avisar "otimizando para sua placa, só na primeira vez".
- FP8 exige quantização com calibração (ex.: NVIDIA TensorRT Model Optimizer) e pode degradar redes de SR, que são sensíveis a precisão. Tratar FP8 como experimental: medir PSNR/SSIM e inspecionar visualmente contra FP16 antes de tornar padrão. FP4 da série 50 não é recomendado para SR.
- Em AMD no Windows, DirectML é o caminho mais estável. ROCm no Windows ainda é limitado; no Linux, ROCm/MIGraphX funcionam bem nas RDNA 3/4 suportadas.

---

## 5. Roadmap

### Fase 0: base (feito neste commit)
- [x] Estrutura do repositório e engine em Python
- [x] Detecção de hardware com várias sondas e escolha de preset
- [x] Pipeline FFmpeg ⇄ NumPy com threads, filas limitadas e áudio sincronizado
- [x] Tiling com sobreposição e redução automática de tile por falta de VRAM
- [x] Estágio ONNX de SR com TensorRT/CUDA/DirectML/MIGraphX/OpenVINO/CPU
- [x] Contrato de estágio temporal (interpolação de referência)
- [x] Conversão BT.601 → BT.709 e preservação de aspect 4:3
- [x] CLI com saída JSON para a futura UI
- [x] Testes, incluindo vídeo VHS sintético de ponta a ponta

### Fase 1: MVP com IA real (2–3 semanas)
- [ ] `tools/export_onnx.py`: exportar Real-ESRGAN x4plus e realesr-general-x4v3 para ONNX (FP32 e FP16, eixos dinâmicos)
- [ ] Gerenciador de modelos: manifesto JSON com URL, SHA-256, licença e preset; download com verificação
- [ ] Estágio RIFE em ONNX
- [ ] Estágio NCNN-Vulkan via binários `realesrgan-ncnn-vulkan` e `rife-ncnn-vulkan` (subprocesso com pipe)
- [ ] Prévia: processar 10 s do meio do vídeo e gerar comparação lado a lado
- [ ] Segmentação com retomada

### Fase 2: rostos e qualidade (3–4 semanas)
- [ ] facexlib + GFPGAN, colagem com máscara e suavização temporal
- [ ] CodeFormer como download opcional com aviso de licença
- [ ] RealBasicVSR para o preset Ultra
- [ ] SCUNet / FastDVDnet
- [ ] Benchmark automático: roda 50 frames em cada preset possível e mede fps reais, ajustando a escolha

### Fase 3: interface desktop (3–4 semanas)
- [ ] Tauri 2 + React em `app/`
- [ ] Fila de jobs, presets, comparação antes/depois com slider
- [ ] Instalador de runtime packs (NVIDIA / AMD / Intel / CPU)
- [ ] Estimativa de tempo e espaço em disco

### Fase 4: TensorRT e distribuição (2–3 semanas)
- [ ] Engines TensorRT FP16 com cache; FP8 experimental
- [ ] CI no GitHub Actions: testes, build Windows/Linux, release com instaladores
- [ ] Documentação de usuário e guia de contribuição

---

## 6. Estrutura do repositório

```
videoimprovewithgraphicscard/
├── engine/                  # motor de processamento (Python)
│   ├── __main__.py          # python -m engine
│   ├── cli.py               # CLI e protocolo JSON para a UI
│   ├── core/
│   │   ├── hardware.py      # detecção de GPU e escolha de preset
│   │   ├── pipeline.py      # FFmpeg ⇄ estágios ⇄ FFmpeg
│   │   └── tiling.py        # tiles com sobreposição
│   └── stages/
│       ├── upscale.py       # OnnxUpscale, ResizeUpscale
│       └── interpolate.py   # contrato de estágio temporal
├── models/                  # manifesto de modelos (pesos ficam fora do git)
├── tests/
├── tools/                   # exportação de modelos para ONNX (fase 1)
├── app/                     # UI Tauri + React (fase 3)
└── docs/
```

---

## 7. Empacotamento e distribuição

### 7.1 Licença do projeto

**Apache-2.0** ou **MIT** para o código. Apache-2.0 tem concessão explícita de patentes, útil num projeto que mexe com codecs e IA. Os pesos dos modelos têm licenças próprias e ficam fora do repositório e do instalador.

### 7.2 Dependências pesadas: "runtime packs"

O instalador base leva só a UI, a engine congelada, FFmpeg e o runtime de CPU (~150 MB). Na primeira execução, a detecção de hardware sugere o pacote certo:

| Pack | Conteúdo | Tamanho aprox. |
|---|---|---|
| NVIDIA | `onnxruntime-gpu` + DLLs CUDA/cuDNN (via wheels `nvidia-*`) + TensorRT opcional | 1–2,5 GB |
| AMD / Intel no Windows | `onnxruntime-directml` | ~50 MB |
| Intel | `onnxruntime-openvino` | ~100 MB |
| Vulkan | binários `*-ncnn-vulkan` | ~30 MB |

Os packs ficam numa pasta própria do app, isolados do sistema, sem conflito com outras instalações de CUDA. As DLLs da NVIDIA podem ser redistribuídas dentro dos termos dos EULAs de CUDA, cuDNN e TensorRT; leia a lista de arquivos redistribuíveis antes da primeira release.

### 7.3 FFmpeg

- Build **LGPL** (por exemplo, os builds automáticos do BtbN) pode ir junto com app de qualquer licença.
- `libx264`/`libx265` exigem build **GPL**. Alternativas sem GPL: NVENC/AMF/QSV (hardware), SVT-AV1 (BSD) e OpenH264 (BSD).
- Recomendação: build LGPL como padrão e encoder de hardware quando disponível; build GPL como download opcional.

### 7.4 Empacotamento

- Engine: **PyInstaller** (mais simples) ou **Nuitka** (mais rápido para iniciar, mais trabalhoso). Alternativa moderna: Python embarcado + `uv` instalando dependências travadas na primeira execução.
- UI: bundler do Tauri gera MSI/NSIS no Windows, AppImage/deb/rpm no Linux, dmg no macOS.
- CI: GitHub Actions com matriz Windows/Linux, testes da engine em CPU, build dos instaladores em tag `v*` e release com checksums.

---

## 8. Próximos passos sugeridos

1. Exportar o `realesr-general-x4v3` para ONNX e validar o `OnnxUpscale` com um trecho real de VHS digitalizado.
2. Rodar `python -m engine detect` no seu PC Windows para confirmar a detecção com uma GPU real.
3. Escolher entre Tauri e PySide6 para a UI antes de começar a fase 3.
