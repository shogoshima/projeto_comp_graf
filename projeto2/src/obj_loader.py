"""Parser de Wavefront .obj com suporte a múltiplos materiais.

- Lê v / vt / vn / f / usemtl / mtllib.
- Triangula faces com >3 vértices via fan-triangulation (segurança extra,
  mesmo que tools/convert_assets.py já triangule offline).
- Materiais e seus map_Kd vêm do .mtl correspondente.
- Devolve uma lista de SubMesh, cada uma com seu material e seus vértices
  intercalados (px,py,pz, u,v, nx,ny,nz).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Material:
    name: str
    map_kd: str | None = None        # caminho da textura (.png/.jpg) — relativo ao .mtl
    kd: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class SubMesh:
    """Bloco de triângulos que compartilham um material."""
    material: str
    vertices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    # vertices: shape (N*3, 8) → 8 floats por vértice (px,py,pz,u,v,nx,ny,nz)


def _resolve_idx(tok: str, count: int) -> int:
    """Converte um índice OBJ (1-based, ou negativo relativo) para 0-based.
    `count` é o tamanho atual da lista correspondente (positions/tex/normals)."""
    i = int(tok)
    if i > 0:
        return i - 1
    if i < 0:
        return count + i  # ex: -1 com count=10 → 9 (último elemento)
    return -1


def parse_mtl(mtl_path: Path) -> dict[str, Material]:
    """Lê um .mtl e devolve {name: Material}."""
    materials: dict[str, Material] = {}
    cur: Material | None = None
    if not mtl_path.exists():
        return materials
    base = mtl_path.parent
    with mtl_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            head, *rest = line.split(maxsplit=1)
            arg = rest[0].strip() if rest else ""
            if head == "newmtl":
                cur = Material(name=arg)
                materials[arg] = cur
            elif cur is None:
                continue
            elif head == "Kd":
                parts = arg.split()
                if len(parts) >= 3:
                    cur.kd = (float(parts[0]), float(parts[1]), float(parts[2]))
            elif head == "map_Kd":
                # Wavefront map_Kd pode ter flags (-s, -o, etc) antes do filename.
                # Filename = último token.
                tokens = arg.split()
                cur.map_kd = str((base / tokens[-1]).resolve())
    return materials


def load_obj(obj_path: str | Path) -> tuple[list[SubMesh], dict[str, Material]]:
    """Carrega um .obj e devolve (sub-meshes, materiais)."""
    obj_path = Path(obj_path)
    base = obj_path.parent

    positions: list[tuple[float, float, float]] = []
    tex_coords: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []

    submesh_buckets: dict[str, list[float]] = {}
    cur_mat = "__default__"
    submesh_buckets[cur_mat] = []
    materials: dict[str, Material] = {}

    with obj_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            head, *rest = line.split(maxsplit=1)
            arg = rest[0] if rest else ""
            if head == "v":
                p = arg.split()
                positions.append((float(p[0]), float(p[1]), float(p[2])))
            elif head == "vt":
                p = arg.split()
                # algumas exportações têm w opcional; só usamos u, v
                u = float(p[0])
                v = float(p[1]) if len(p) > 1 else 0.0
                tex_coords.append((u, v))
            elif head == "vn":
                p = arg.split()
                normals.append((float(p[0]), float(p[1]), float(p[2])))
            elif head == "mtllib":
                materials.update(parse_mtl(base / arg.strip()))
            elif head == "usemtl":
                cur_mat = arg.strip()
                submesh_buckets.setdefault(cur_mat, [])
            elif head == "f":
                tokens = arg.split()
                # fan-triangulate
                tri_indices = []
                # OBJ permite índices NEGATIVOS (relativos ao fim da lista corrente):
                # -1 = último vértice já definido. Convertemos para 0-based abaixo.
                np_v, np_t, np_n = len(positions), len(tex_coords), len(normals)
                for tok in tokens:
                    parts = tok.split("/")
                    vi = _resolve_idx(parts[0], np_v) if parts[0] else -1
                    ti = (_resolve_idx(parts[1], np_t)
                          if len(parts) > 1 and parts[1] else -1)
                    ni = (_resolve_idx(parts[2], np_n)
                          if len(parts) > 2 and parts[2] else -1)
                    tri_indices.append((vi, ti, ni))
                if len(tri_indices) < 3:
                    continue
                bucket = submesh_buckets[cur_mat]
                # fan: (0, i, i+1)
                for i in range(1, len(tri_indices) - 1):
                    for vi, ti, ni in (tri_indices[0], tri_indices[i], tri_indices[i + 1]):
                        # posição
                        if 0 <= vi < len(positions):
                            px, py, pz = positions[vi]
                        else:
                            px = py = pz = 0.0
                        # uv
                        if 0 <= ti < len(tex_coords):
                            u, v = tex_coords[ti]
                        else:
                            u = v = 0.0
                        # normal (não usamos no shader, mas guardamos por compat.)
                        if 0 <= ni < len(normals):
                            nx, ny, nz = normals[ni]
                        else:
                            nx = ny = nz = 0.0
                        bucket.extend([px, py, pz, u, v, nx, ny, nz])

    submeshes: list[SubMesh] = []
    for mat_name, buf in submesh_buckets.items():
        if not buf:
            continue
        arr = np.asarray(buf, dtype=np.float32).reshape(-1, 8)
        submeshes.append(SubMesh(material=mat_name, vertices=arr))

    return submeshes, materials
