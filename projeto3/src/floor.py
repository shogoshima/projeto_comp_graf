"""Pisos do cenário — gerados em runtime para cumprir o requisito 6 do PDF
(piso interno diferente do externo) sem depender de assets externos.

Externo: plano enorme texturizado de grama/musgo. Procedural (Pillow → OpenGL).
Interno: disco circular dentro da cabana, texturizado com tábuas de madeira.
Lago: disco circular azulado com leve padrão de ondas (procedural).
"""
from __future__ import annotations

import ctypes
import math

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_FALSE, GL_FLOAT, 
    GL_STATIC_DRAW, GL_TEXTURE0, GL_TEXTURE_2D, GL_TRIANGLES,
    glActiveTexture, glBindBuffer, glBindTexture, glBindVertexArray,
    glBufferData, glDrawArrays, glEnableVertexAttribArray, glGenBuffers,
    glGenVertexArrays, glVertexAttribPointer,
)

from src import transforms as T


VERTEX_STRIDE = 8 * 4  # mantém o mesmo layout do Mesh (px,py,pz,u,v,nx,ny,nz)


# --------------------------------------------------------------------------- #
# Geometrias planares
# --------------------------------------------------------------------------- #
def _quad_with_hole(world_half: float, hole_radius: float, hole_center: tuple[float, float],
                    segments: int = 64, uv_scale: float = 16.0) -> np.ndarray:
    """Retorna um array (N*3, 8) — um plano quadrado [-world_half, world_half]
    com um buraco circular `hole_radius` em volta de `hole_center` (x, z).

    Construído como anel quadrado-para-circular (segmentos triangulares) cuja
    borda externa é o quadrado e a interna é o círculo.
    """
    cx, cz = hole_center
    verts: list[float] = []

    def push(x: float, z: float) -> None:
        u = (x + world_half) / (2 * world_half) * uv_scale
        v = (z + world_half) / (2 * world_half) * uv_scale
        verts.extend([x, 0.0, z, u, v, 0.0, 1.0, 0.0])

    # ponto na borda externa "alinhado" com o ângulo theta
    def outer_for_angle(theta: float) -> tuple[float, float]:
        # interseção do raio em theta com o quadrado [-h, h]
        h = world_half
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        # parametrizamos t ≥ 0; para sair pelo quadrado, t = h / max(|cos|, |sin|)
        t = h / max(abs(cos_t), abs(sin_t), 1e-9)
        x = cx + t * cos_t
        z = cz + t * sin_t
        # clamp para garantir que estamos dentro do quadrado original (centrado na origem)
        x = max(-h, min(h, x))
        z = max(-h, min(h, z))
        return x, z

    for i in range(segments):
        t0 = (i     / segments) * math.tau
        t1 = ((i+1) / segments) * math.tau
        # inner ring (círculo do buraco)
        ix0, iz0 = cx + hole_radius * math.cos(t0), cz + hole_radius * math.sin(t0)
        ix1, iz1 = cx + hole_radius * math.cos(t1), cz + hole_radius * math.sin(t1)
        # outer ring (borda do quadrado)
        ox0, oz0 = outer_for_angle(t0)
        ox1, oz1 = outer_for_angle(t1)
        # 2 triângulos: winding CCW vista de +Y (face superior visível com
        # GL_CULL_FACE=BACK + GL_FRONT_FACE=CCW padrão)
        push(ix0, iz0); push(ox1, oz1); push(ox0, oz0)
        push(ix0, iz0); push(ix1, iz1); push(ox1, oz1)

    return np.asarray(verts, dtype=np.float32).reshape(-1, 8)


def _disk(radius: float, segments: int = 64, y: float = 0.0,
          uv_scale: float = 4.0) -> np.ndarray:
    """Disco circular planar (XZ) na altura y. Centrado na origem."""
    verts: list[float] = []

    def push(x: float, z: float) -> None:
        # mapear UV em coords cartesianas → faz o tile funcionar corretamente
        u = (x / radius + 1.0) * 0.5 * uv_scale
        v = (z / radius + 1.0) * 0.5 * uv_scale
        verts.extend([x, y, z, u, v, 0.0, 1.0, 0.0])

    for i in range(segments):
        t0 = (i     / segments) * math.tau
        t1 = ((i+1) / segments) * math.tau
        x0, z0 = radius * math.cos(t0), radius * math.sin(t0)
        x1, z1 = radius * math.cos(t1), radius * math.sin(t1)
        # CCW vista de +Y → face superior visível com cull_face=BACK
        push(0.0, 0.0); push(x1, z1); push(x0, z0)

    return np.asarray(verts, dtype=np.float32).reshape(-1, 8)


# --------------------------------------------------------------------------- #
# Wrapper de "piso" — VAO/VBO + draw, com transform opcional
# --------------------------------------------------------------------------- #
class _PlanarMesh:
    def __init__(self, vertices: np.ndarray, texture_id: int):
        self.count = vertices.shape[0]
        self.texture_id = texture_id
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.scale    = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(5 * 4))
        glBindVertexArray(0)

    def model_matrix(self) -> np.ndarray:
        return T.trs(tuple(self.position.tolist()),
                     tuple(self.rotation.tolist()),
                     tuple(self.scale.tolist()))

    def draw(self, shader) -> None:
        shader.set_mat4("u_model", self.model_matrix())
        shader.set_vec3("u_kd", 1.0, 1.0, 1.0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        shader.set_int("u_tex", 0)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.count)
        glBindVertexArray(0)


class GrassFloorWithHole(_PlanarMesh):
    """Chão externo (grama/musgo) com um buraco circular para a cabana."""
    def __init__(self, world_half: float, hole_radius: float,
                 hole_center: tuple[float, float] = (0.0, 0.0),
                 segments: int = 64, uv_scale: float = 24.0):
        from pathlib import Path
        from src.texture import load_texture_2d
        verts = _quad_with_hole(world_half, hole_radius, hole_center,
                                segments=segments, uv_scale=uv_scale)
        tex = load_texture_2d(str(Path(__file__).resolve().parent.parent / "assets" / "nature_textures" / "grass.png"))
        super().__init__(verts, tex)


class WaterDisk(_PlanarMesh):
    """Lago — disco com textura de água."""
    def __init__(self, radius: float, segments: int = 96, uv_scale: float = 2.0):
        from pathlib import Path
        from src.texture import load_texture_2d
        verts = _disk(radius=radius, segments=segments, y=0.0, uv_scale=uv_scale)
        tex = load_texture_2d(str(Path(__file__).resolve().parent.parent / "assets" / "nature_textures" / "water.png"))
        super().__init__(verts, tex)
