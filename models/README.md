# Modelos

Os pesos e os `.onnx` gerados não vão para o git. Instale com `python -m engine models install all`: o comando baixa os pesos oficiais para `models/_weights/`, confere o SHA-256 e gera `<nome>.onnx` (FP32) e `<nome>.fp16.onnx` nesta pasta. O catálogo fica em `engine/models/catalog.py`.

| Modelo | Tarefa | Licença dos pesos | Pode ir no instalador? |
|---|---|---|---|
| realesr-general-x4v3 | SR leve | BSD-3-Clause | sim |
| RealESRGAN_x4plus | SR | BSD-3-Clause | sim |
| RealBasicVSR | SR de vídeo | Apache-2.0 | sim |
| GFPGAN v1.4 | rosto | Apache-2.0 | sim |
| CodeFormer | rosto | S-Lab License 1.0 (**não comercial**) | não, só download opcional com aviso |
| RIFE 4.x | interpolação | MIT | sim |
| SCUNet | denoise | Apache-2.0 | sim |

Confirme a licença na página oficial de cada projeto antes de cada release; elas podem mudar.
