# VideoImprove

Restauração e upscaling de vídeos antigos (fitas VHS digitalizadas, Hi8, filmes caseiros) que roda em **qualquer PC mediano**: vídeo integrado Intel/AMD, GTX 1660 e afins. Em placas de ponta (RTX 40/50), usa modelos mais pesados.

## Baixar e usar (Windows 10/11, 64 bits)

1. Vá em **[Releases](../../releases)** e baixe um dos zips:
   - **VideoImprove-Universal-win64.zip**: qualquer placa de vídeo (Intel, AMD, NVIDIA, inclusive vídeo integrado).
   - **VideoImprove-NVIDIA-win64.zip**: placas NVIDIA RTX ou GTX 16. Mais rápido nelas, download maior. Exige driver NVIDIA 580 ou mais novo.
2. Extraia a pasta e abra o **VideoImprove.exe**. Não precisa instalar nada; funciona até de um pendrive.
3. Arraste os vídeos para a janela, veja a prévia (opcional) e clique em **Restaurar**.

Na primeira vez, o Windows pode avisar "O Windows protegeu o computador" (o programa não tem assinatura digital paga): clique em **Mais informações → Executar assim mesmo**.

A pré-release **dev** é atualizada a cada mudança na branch `main`.

---

## Para desenvolvedores

A engine e a interface são Python (ONNX Runtime + FFmpeg + PySide6). O app portátil é gerado pelo GitHub Actions ([build-windows.yml](.github/workflows/build-windows.yml)) com PyInstaller ([packaging/](packaging/)). Veja o [planejamento](docs/ARQUITETURA.md).

### Rodando do código-fonte no Windows

#### 1. Instalar Python, Git e FFmpeg (uma vez)

No PowerShell:

```powershell
winget install Python.Python.3.12 Git.Git Gyan.FFmpeg
```

**Feche o terminal e abra um novo** antes de continuar. Sem isso, `git` e `py` não são reconhecidos, porque o PATH só atualiza em terminais novos. Use um PowerShell normal, **não como administrador**, e cole os blocos abaixo um de cada vez.

#### 2. Baixar o projeto e criar o ambiente

```powershell
cd $HOME
git clone https://github.com/akkotfpn/videoimprovewithgraphicscard.git
cd videoimprovewithgraphicscard
py -3.12 -m venv .venv
```

Se o `py` não for reconhecido mesmo no terminal novo, use o caminho completo:
`& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv`

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # só na primeira vez
.venv\Scripts\Activate.ps1
```

O prompt passa a começar com `(.venv)`. A partir daí, `python` e `pip` são os do ambiente do projeto. Em cada terminal novo, entre na pasta e rode o `Activate.ps1` de novo.

#### 3. Instalar o runtime da sua placa (escolha **um**)

| Sua placa | Comando |
|---|---|
| NVIDIA RTX (20/30/40/50) | `pip install -e ".[nvidia,gui,dev]"` |
| Qualquer outra no Windows: Intel UHD/Iris/Arc, AMD integrada ou RX, GTX 1650/1660 | `pip install -e ".[directml,gui,dev]"` |
| Sem GPU | `pip install -e ".[cpu,gui,dev]"` |

Interface: `python -m gui`. O resto deste guia usa a linha de comando da engine.

#### 4. Detectar, instalar modelos e medir

```powershell
python -m engine detect                         # o que foi detectado e qual preset
python -m engine models install all             # baixa e converte os modelos (~75 MB, sem PyTorch)
python -m engine bench                          # velocidade do modelo do seu preset
python -m engine bench --model realesr_general_x4v3   # modelo leve (o dos PCs medianos)
python -m engine bench --model realesrgan_x4plus      # modelo pesado
```

O `bench` mostra quantos frames por segundo a sua máquina processa e estima quanto tempo leva **1 hora de fita**.

#### 5. Restaurar um vídeo

```powershell
python -m engine run "C:\Videos\fita.avi" "C:\Videos\fita_restaurada.mkv"
```

Opções úteis:

- `--scale 2` (padrão) ou `--scale 4`. Para VHS, 2× costuma ficar mais natural.
- `--interpolate` dobra o FPS.
- `--preset economico` força um preset.
- `--model realesrgan_x4plus` escolhe o modelo.
- Use `.mkv` na saída para copiar o áudio original sem recompressão.

### Presets

| Preset | Exemplo de placa | Runtime | Modelo padrão |
|---|---|---|---|
| Ultra | RTX 4070/5070 ou superior (FP8, 12 GB+) | TensorRT ou CUDA | Real-ESRGAN x4plus* |
| Equilibrado | RTX 2060/3060, RX 6600+, Arc A750 | CUDA / DirectML / ROCm / OpenVINO | Real-ESRGAN x4plus |
| Econômico | GTX 1650/1660, Intel Iris Xe/UHD, AMD Radeon 680M/780M | DirectML / Vulkan | Real-ESRGAN compact |
| CPU | sem GPU utilizável | ONNX Runtime CPU | filtros clássicos (IA só se pedir) |

\* RealBasicVSR e restauração de rostos entram na fase 2.

## Licença

Código sob [Apache-2.0](LICENSE). Os pesos dos modelos têm licenças próprias e não fazem parte do repositório; veja [models/README.md](models/README.md).

### Testes

```bash
python -m pytest
```

Os testes incluem:
- placas simuladas, da Intel UHD à RTX 5090;
- modelos ONNX reais conferidos contra uma implementação de referência em NumPy;
- proteção contra checkpoints maliciosos;
- um vídeo VHS sintético processado de ponta a ponta, conferindo resolução, contagem de frames, espaço de cor e sincronia do áudio.
