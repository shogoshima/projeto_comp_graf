"""
Fase 1 — Pipeline de conversão de assets.

Lê cada modelo de origem, triangula faces (fan-triangulation), reescreve com
nomes de material limpos, gera .mtl mínimo (apenas map_Kd, ignora Normal /
Roughness / Metallic / AO — projeto sem iluminação) e copia a textura
correspondente para assets/_obj_exports/<nome>/.

Pipeline 100% Python (sem Blender). Para casos extremos de n-gons côncavos
o loader em runtime também triangula (fan) por segurança.

Uso:
    python3 tools/convert_assets.py
"""
from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
EXPORTS = ROOT / "assets" / "_obj_exports"


# --------------------------------------------------------------------------- #
# Configuração dos jobs
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    """Receita para conversão de UM modelo."""
    name: str                          # subpasta de saída (assets/_obj_exports/<name>/)
    src_obj: str                       # .obj de entrada (relativo ao ROOT)
    # Mapeamento de material: cada entrada vira (mat_name_no_obj_final, tex_filename_destino, src_path)
    # As chaves são padrões de "usemtl" do .obj original; se o nome do material original
    # contiver `match`, ele é renomeado para `mat_name` e aponta para `tex`.
    # Se houver só uma entrada e match == "*", todos os materiais viram esse.
    materials: list[tuple[str, str, str]] = field(default_factory=list)
    # Optional: usemtl substitute. Se não-vazio, todos `usemtl X` viram `usemtl <default>`.
    force_single_material: str | None = None


JOBS: list[Job] = [
    # ---- Cabana (delimitador) ----
    Job(
        name="hut",
        src_obj="hut/extracted/export/log_hut_stylised.obj",
        materials=[("*", "hut_palette", "hut/extracted/export/color_palette.png")],
        force_single_material="hut_palette",
    ),

    # ---- Barco (externo) ----
    # 4 materiais no original (Black Desert Online): 3 usam "body", 1 usa "ship".
    # Mapeamos: tudo "body" → Body.jpg, "ship" → paddle.jpeg.
    Job(
        name="boat",
        src_obj="boat/uploads-files-6980803-Boat.obj",
        materials=[
            ("body", "boat_body",   "boat/Body.jpg"),
            ("ship", "boat_paddle", "boat/paddle .jpeg"),
        ],
    ),

    # ---- Vara de pesca (interno) ----
    Job(
        name="fishingrod",
        src_obj="fishingrod/obj_extracted/spining01.obj",
        materials=[("*", "rod_diffuse",
                    "fishingrod/tex_extracted/map_Spining01_BaseColor.png")],
        force_single_material="rod_diffuse",
    ),

    # ---- Cavalo-marinho (externo) ----
    # PSD será convertido em PNG por convert_textures.py antes deste script rodar.
    Job(
        name="seahorse",
        src_obj="seahorse/uploads-files-2037155-seahorse.obj",
        materials=[("*", "seahorse_diffuse",
                    "seahorse/seahorse_diffuse.png")],
        force_single_material="seahorse_diffuse",
    ),

    # ---- Polvo (externo, no lago) ----
    # Textura procedural será gerada antes deste script rodar.
    Job(
        name="octopus",
        src_obj="octopus/uploads-files-6224776-octopus_squid_alpha_Biolumines.obj",
        materials=[("*", "octopus_diffuse",
                    "octopus/octopus_diffuse.png")],
        force_single_material="octopus_diffuse",
    ),

    # ---- Árvore laranja (externo) ----
    # 2 materiais (.002 = folhas, .004 = tronco) — ambos no mesmo atlas.
    Job(
        name="tree_orange",
        src_obj="trees/orange/Orange Tree/Orange Tree.obj",
        materials=[("*", "tree_orange_atlas",
                    "trees/orange/Orange Tree/Orange Tree Texture.png")],
        force_single_material="tree_orange_atlas",
    ),

    # ---- Árvore verde (externo) ----
    Job(
        name="tree_green",
        src_obj="trees/green/Green Tree/Green Tree.obj",
        materials=[("*", "tree_green_atlas",
                    "trees/green/Green Tree/Green Tree Texture.png")],
        force_single_material="tree_green_atlas",
    ),

    # ---- Balde (interno) ----
    # 1 material "Bucket" + textura BaseColor 1K extraída do .rar.
    Job(
        name="bucket",
        src_obj="bucket/uploads-files-4654687-Bucket.obj",
        materials=[("*", "bucket_basecolor",
                    "bucket/Textures/1K/Bucket_BaseColor.png")],
        force_single_material="bucket_basecolor",
    ),

    # ---- Lanterna (interno) ----
    # Asset só trouxe .obj + .max (sem textura). Usamos textura procedural
    # gerada por prep_textures.py (faixas pretas/grafite + bisel dourado).
    Job(
        name="flashlight",
        src_obj="flashlight/uploads-files-1894371-flashlight_01.obj",
        materials=[("*", "flashlight_diffuse",
                    "flashlight/flashlight_diffuse.png")],
        force_single_material="flashlight_diffuse",
    ),
]


# --------------------------------------------------------------------------- #
# Triangulação por fan (convexa)
# --------------------------------------------------------------------------- #
def triangulate_face(tokens: list[str]) -> Iterable[tuple[str, str, str]]:
    """Recebe os tokens de uma linha 'f' (sem o 'f') e devolve triangulos
    como tuplas de 3 tokens. Faz fan-triangulation: (v0, vi, vi+1)."""
    n = len(tokens)
    if n < 3:
        return
    if n == 3:
        yield (tokens[0], tokens[1], tokens[2])
        return
    for i in range(1, n - 1):
        yield (tokens[0], tokens[i], tokens[i + 1])


# --------------------------------------------------------------------------- #
# Conversão de UM modelo
# --------------------------------------------------------------------------- #
def convert_one(job: Job) -> None:
    src = ROOT / job.src_obj
    if not src.exists():
        print(f"[skip] {job.name}: {src} não encontrado")
        return

    out_dir = EXPORTS / job.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_obj = out_dir / f"{job.name}.obj"
    out_mtl = out_dir / f"{job.name}.mtl"

    # 1) Copia texturas + monta tabela de materiais finais
    final_materials: list[tuple[str, str]] = []  # (mat_name, tex_filename)
    seen_names: set[str] = set()
    for match, mat_name, tex_path in job.materials:
        tex_src = ROOT / tex_path
        if not tex_src.exists():
            print(f"[warn] {job.name}: textura {tex_src} não existe (vai ficar sem map_Kd)")
            continue
        tex_dst_name = f"{mat_name}{tex_src.suffix.lower()}"
        tex_dst = out_dir / tex_dst_name
        # Sempre converte JPEG/JPG e copia tudo como o mesmo formato local.
        if tex_src.suffix.lower() in (".jpg", ".jpeg", ".png"):
            shutil.copyfile(tex_src, tex_dst)
        else:
            # converte qualquer outro formato suportado pelo Pillow → PNG
            tex_dst_name = f"{mat_name}.png"
            tex_dst = out_dir / tex_dst_name
            Image.open(tex_src).convert("RGBA").save(tex_dst)
        if mat_name not in seen_names:
            final_materials.append((mat_name, tex_dst_name))
            seen_names.add(mat_name)

    # 2) Lê e processa o .obj
    triangles_written = 0
    quads_or_more = 0
    materials_seen_in_obj: set[str] = set()

    with src.open("r", encoding="utf-8", errors="replace") as f_in, \
         out_obj.open("w", encoding="utf-8") as f_out:

        f_out.write(f"# Convertido de {job.src_obj} (Fase 1)\n")
        f_out.write(f"mtllib {job.name}.mtl\n")
        f_out.write(f"o {job.name}\n")

        current_mat: str | None = None
        for raw in f_in:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            head = line.split(maxsplit=1)[0]
            if head in ("v", "vt", "vn"):
                f_out.write(line + "\n")
            elif head == "usemtl":
                orig_mat = line[len("usemtl "):].strip()
                materials_seen_in_obj.add(orig_mat)
                if job.force_single_material:
                    new_mat = job.force_single_material
                else:
                    new_mat = _resolve_material(orig_mat, job.materials)
                if new_mat != current_mat:
                    f_out.write(f"usemtl {new_mat}\n")
                    current_mat = new_mat
            elif head == "f":
                tokens = line.split()[1:]
                if len(tokens) > 3:
                    quads_or_more += 1
                for tri in triangulate_face(tokens):
                    f_out.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")
                    triangles_written += 1
            # ignora s, g, o, mtllib originais

    # 3) Escreve .mtl limpo
    with out_mtl.open("w", encoding="utf-8") as f_mtl:
        f_mtl.write(f"# Materiais simplificados de {job.name}\n")
        for mat_name, tex_filename in final_materials:
            f_mtl.write(f"\nnewmtl {mat_name}\n")
            f_mtl.write("Ka 1.000 1.000 1.000\n")
            f_mtl.write("Kd 1.000 1.000 1.000\n")
            f_mtl.write("Ks 0.000 0.000 0.000\n")
            f_mtl.write("d 1.000\n")
            f_mtl.write("illum 1\n")
            f_mtl.write(f"map_Kd {tex_filename}\n")

    print(f"[ok]   {job.name:14s} → {out_obj.relative_to(ROOT)} "
          f"({triangles_written} tris, {quads_or_more} faces n-gon expandidas, "
          f"{len(materials_seen_in_obj)} mats originais → {len(final_materials)} finais)")


def _resolve_material(orig_mat: str, mat_specs: list[tuple[str, str, str]]) -> str:
    """Encontra qual material final o `orig_mat` deve virar."""
    needle = orig_mat.lower()
    for match, mat_name, _ in mat_specs:
        if match == "*" or match.lower() in needle:
            return mat_name
    # fallback: primeiro
    return mat_specs[0][1] if mat_specs else "default"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    print(f"[init] exportando para {EXPORTS.relative_to(ROOT)}/")
    for job in JOBS:
        convert_one(job)
    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
