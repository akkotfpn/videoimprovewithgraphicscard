# Modelos

Os pesos não ficam no git (veja `.gitignore`). Na fase 1, um `manifest.json` aqui vai listar cada modelo com URL, SHA-256, licença e preset, e o app baixa sob demanda.

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
