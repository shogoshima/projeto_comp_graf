"""
Pré-processa texturas:
  1) Converte seahorse PSD → PNG (Pillow lê só a camada composta).
  2) Gera textura procedural para o polvo (não veio textura no asset).
  3) Gera textura procedural para a lanterna (asset não trouxe textura, só .max).

Roda ANTES de tools/convert_assets.py.

Uso:
    python3 tools/prep_textures.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# 1) PSD → PNG
# --------------------------------------------------------------------------- #
def convert_psd_to_png() -> None:
    src = ROOT / "seahorse" / "uploads-files-2037155-texture.psd"
    dst = ROOT / "seahorse" / "seahorse_diffuse.png"
    if not src.exists():
        print(f"[skip] PSD não encontrado: {src}")
        return
    im = Image.open(src)
    im.load()
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    im.save(dst, "PNG")
    print(f"[ok]   PSD → {dst.relative_to(ROOT)} ({im.size[0]}x{im.size[1]})")


# --------------------------------------------------------------------------- #
# 2) Polvo procedural
# --------------------------------------------------------------------------- #
def make_octopus_texture(size: int = 1024) -> None:
    """Gera uma textura de pele de polvo: base púrpura/rosa + manchas mais
    escuras (pintas de polvo) + halo rosado nas bordas via gradiente radial.

    Sem dependência de iluminação (cumpre restrição do projeto).
    """
    rng = np.random.default_rng(seed=42)

    # gradiente vertical: dorso púrpura → ventre rosa pálido
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    t = yy / (size - 1)
    base_dark = np.array([0.45, 0.18, 0.45], dtype=np.float32)   # púrpura
    base_light = np.array([0.95, 0.70, 0.78], dtype=np.float32)  # rosa pálido
    img = base_dark[None, None, :] * (1 - t)[..., None] + base_light[None, None, :] * t[..., None]

    # ruído baixa frequência (manchas) — soma de senos com fases aleatórias
    noise = np.zeros((size, size), dtype=np.float32)
    for freq in (3.0, 6.0, 12.0):
        phase_x = rng.uniform(0, math.tau)
        phase_y = rng.uniform(0, math.tau)
        noise += 0.5 * (
            np.sin(2 * math.pi * freq * xx / size + phase_x)
            * np.cos(2 * math.pi * freq * yy / size + phase_y)
        )
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)

    # mancha vermelho-vinho moduladora
    spot_color = np.array([0.30, 0.05, 0.20], dtype=np.float32)
    spot_mask = (noise > 0.65).astype(np.float32) * 0.55
    img = img * (1 - spot_mask[..., None]) + spot_color[None, None, :] * spot_mask[..., None]

    # ventosas: pequenos círculos claros distribuídos em fileiras
    sucker_color = np.array([1.0, 0.85, 0.85], dtype=np.float32)
    n_rows, n_cols = 14, 22
    for r in range(n_rows):
        cy = (r + 0.5) * size / n_rows
        for c in range(n_cols):
            cx = (c + 0.5 + 0.5 * (r % 2)) * size / n_cols
            radius = 9 + rng.integers(-2, 3)
            y0, y1 = max(0, int(cy - radius)), min(size, int(cy + radius))
            x0, x1 = max(0, int(cx - radius)), min(size, int(cx + radius))
            if y1 <= y0 or x1 <= x0:
                continue
            yy_s, xx_s = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            d2 = (yy_s - cy) ** 2 + (xx_s - cx) ** 2
            mask = np.clip(1.0 - d2 / (radius * radius), 0.0, 1.0) ** 1.5
            img[y0:y1, x0:x1] = (
                img[y0:y1, x0:x1] * (1 - mask[..., None] * 0.7)
                + sucker_color[None, None, :] * (mask[..., None] * 0.7)
            )

    img = np.clip(img, 0.0, 1.0)
    out = (img * 255).astype(np.uint8)
    dst = ROOT / "octopus" / "octopus_diffuse.png"
    Image.fromarray(out, "RGB").save(dst)
    print(f"[ok]   octopus procedural → {dst.relative_to(ROOT)} ({size}x{size})")


# --------------------------------------------------------------------------- #
# 3) Lanterna procedural (corpo metálico preto + lente dourada)
# --------------------------------------------------------------------------- #
def make_flashlight_texture(size: int = 512) -> None:
    """Gera textura para a lanterna: faixas horizontais alternando preto fosco
    e cinza grafite (corpo emborrachado) com uma faixa dourada (anel da lente)
    no topo e detalhes de ranhuras."""
    rng = np.random.default_rng(seed=11)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    t = yy / (size - 1)

    # base: preto fosco no corpo, cinza grafite onde fica o cabo (UVs do .obj
    # mapeiam o eixo cilíndrico no V → bandas horizontais funcionam bem).
    body_dark  = np.array([0.08, 0.08, 0.09], dtype=np.float32)   # plástico preto
    body_light = np.array([0.28, 0.28, 0.30], dtype=np.float32)   # grafite
    img = body_dark[None, None, :] * (1 - t)[..., None] + body_light[None, None, :] * t[..., None]

    # ranhuras finas (cabo emborrachado) — modulação senoidal em V
    grooves = 0.5 + 0.5 * np.sin(2 * math.pi * 28 * t)
    grooves = (grooves ** 4)  # picos finos
    img *= (1.0 - 0.35 * grooves[..., None])

    # ruído suave pra textura de plástico
    noise = rng.normal(0.0, 0.04, size=(size, size)).astype(np.float32)
    img += noise[..., None]

    # anel dourado (bisel da lente) numa faixa estreita perto do topo
    bezel_color = np.array([0.85, 0.65, 0.20], dtype=np.float32)  # dourado
    bezel_mask = ((t > 0.78) & (t < 0.86)).astype(np.float32)
    img = img * (1 - bezel_mask[..., None]) + bezel_color[None, None, :] * bezel_mask[..., None]

    # lente prateada brilhante no topo
    lens_color = np.array([0.75, 0.78, 0.82], dtype=np.float32)
    lens_mask = (t > 0.92).astype(np.float32)
    img = img * (1 - lens_mask[..., None]) + lens_color[None, None, :] * lens_mask[..., None]

    img = np.clip(img, 0.0, 1.0)
    out = (img * 255).astype(np.uint8)
    dst = ROOT / "flashlight" / "flashlight_diffuse.png"
    Image.fromarray(out, "RGB").save(dst)
    print(f"[ok]   flashlight procedural → {dst.relative_to(ROOT)} ({size}x{size})")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    convert_psd_to_png()
    make_octopus_texture(size=1024)
    make_flashlight_texture(size=512)
    return 0


if __name__ == "__main__":
    sys.exit(main())
