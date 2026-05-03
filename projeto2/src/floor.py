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
    GL_ARRAY_BUFFER, GL_FALSE, GL_FLOAT, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR,
    GL_REPEAT, GL_RGBA, GL_STATIC_DRAW, GL_TEXTURE0, GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T, GL_TRIANGLES, GL_UNSIGNED_BYTE,
    glActiveTexture, glBindBuffer, glBindTexture, glBindVertexArray,
    glBufferData, glDrawArrays, glEnableVertexAttribArray, glGenBuffers,
    glGenerateMipmap, glGenTextures, glGenVertexArrays, glTexImage2D,
    glTexParameteri, glVertexAttribPointer,
)

from src import transforms as T


VERTEX_STRIDE = 8 * 4  # mantém o mesmo layout do Mesh (px,py,pz,u,v,nx,ny,nz)


# --------------------------------------------------------------------------- #
# Texturas procedurais
# --------------------------------------------------------------------------- #
def _to_gl_texture(rgba: np.ndarray) -> int:
    """Sobe um array (H, W, 4) uint8 como textura 2D mip-mapped, repeat."""
    h, w = rgba.shape[:2]
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 rgba.tobytes())
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


def make_grass_texture(size: int = 512, seed: int = 7) -> int:
    """Grama/musgo: ruído verde-amarronzado tileável."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    base = np.zeros((size, size, 3), dtype=np.float32)
    # cor base verde-musgo
    base[..., 0] = 0.18
    base[..., 1] = 0.34
    base[..., 2] = 0.12

    # ruído tileável: soma de cossenos com freqs inteiras
    n = np.zeros((size, size), dtype=np.float32)
    for f in (2, 5, 11, 23):
        ph_x = rng.uniform(0, math.tau)
        ph_y = rng.uniform(0, math.tau)
        n += np.cos(2 * math.pi * f * xx / size + ph_x) * \
             np.cos(2 * math.pi * f * yy / size + ph_y)
    n = (n - n.min()) / (n.max() - n.min() + 1e-6)
    n = n ** 1.4  # aumenta contraste das pintas

    # tons mais escuros (musgo) onde n é alto
    dark  = np.array([0.10, 0.20, 0.06], dtype=np.float32)
    light = np.array([0.30, 0.55, 0.20], dtype=np.float32)
    base = light[None, None, :] * (1 - n[..., None]) + dark[None, None, :] * n[..., None]

    # poucas pedrinhas / talos amarelados
    spots = rng.random((size, size)) > 0.997
    base[spots] = np.array([0.55, 0.50, 0.20], dtype=np.float32)

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(base * 255.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = 255
    return _to_gl_texture(rgba)


def make_wood_planks_texture(size: int = 512, seed: int = 13) -> int:
    """Tábuas de madeira: faixas horizontais de tons quentes c/ veios."""
    rng = np.random.default_rng(seed)
    H, W = size, size
    img = np.zeros((H, W, 3), dtype=np.float32)

    n_planks = 6
    plank_h = H // n_planks
    plank_tones = [
        (0.55, 0.34, 0.18),
        (0.62, 0.40, 0.22),
        (0.48, 0.30, 0.16),
        (0.58, 0.36, 0.20),
        (0.52, 0.32, 0.18),
        (0.60, 0.38, 0.22),
    ]
    rng.shuffle(plank_tones)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for i in range(n_planks):
        y0 = i * plank_h
        y1 = (i + 1) * plank_h
        tone = np.array(plank_tones[i % len(plank_tones)], dtype=np.float32)
        # veios horizontais: senoide com fase aleatória, modulando o tom
        vein_phase = rng.uniform(0, math.tau)
        vein = 0.07 * np.sin(2 * math.pi * 3 * xx[y0:y1] / W + vein_phase) + \
               0.05 * np.sin(2 * math.pi * 8 * xx[y0:y1] / W + vein_phase * 0.7)
        block = tone[None, None, :] + vein[..., None]
        # ruído fino
        block += rng.normal(scale=0.015, size=block.shape).astype(np.float32)
        img[y0:y1] = block
        # linha escura entre tábuas (juntas)
        img[y0:y0 + 2] *= 0.4
    img[-2:] *= 0.4

    img = np.clip(img, 0.0, 1.0)
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[..., :3] = (img * 255.0).astype(np.uint8)
    rgba[..., 3] = 255
    return _to_gl_texture(rgba)


def make_water_texture(size: int = 512, seed: int = 99) -> int:
    """Água: cyan/azul com ondulações suaves."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    n = np.zeros((size, size), dtype=np.float32)
    for f in (3, 7, 15):
        ph = rng.uniform(0, math.tau)
        n += np.sin(2 * math.pi * f * (xx + yy) / size + ph) * \
             np.cos(2 * math.pi * f * (xx - yy) / size + ph * 0.5)
    n = (n - n.min()) / (n.max() - n.min() + 1e-6)

    dark  = np.array([0.05, 0.20, 0.45], dtype=np.float32)
    light = np.array([0.30, 0.65, 0.85], dtype=np.float32)
    img = dark[None, None, :] * (1 - n[..., None]) + light[None, None, :] * n[..., None]

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = 255
    return _to_gl_texture(rgba)


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


def _quad(half_size: float, y: float = 0.0, uv_scale: float = 32.0) -> np.ndarray:
    h = half_size
    s = uv_scale
    verts = np.array([
        # tri 1
        -h, y, -h,   0, 0,   0, 1, 0,
         h, y, -h,   s, 0,   0, 1, 0,
         h, y,  h,   s, s,   0, 1, 0,
        # tri 2
        -h, y, -h,   0, 0,   0, 1, 0,
         h, y,  h,   s, s,   0, 1, 0,
        -h, y,  h,   0, s,   0, 1, 0,
    ], dtype=np.float32).reshape(-1, 8)
    return verts


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
        verts = _quad_with_hole(world_half, hole_radius, hole_center,
                                segments=segments, uv_scale=uv_scale)
        tex = make_grass_texture()
        super().__init__(verts, tex)


class WoodFloorDisk(_PlanarMesh):
    """Chão interno da cabana — disco circular de tábuas de madeira."""
    def __init__(self, radius: float, segments: int = 64, uv_scale: float = 3.0):
        verts = _disk(radius=radius, segments=segments, y=0.0, uv_scale=uv_scale)
        tex = make_wood_planks_texture()
        super().__init__(verts, tex)


class WaterDisk(_PlanarMesh):
    """Lago — disco azul com padrão de ondas."""
    def __init__(self, radius: float, segments: int = 96, uv_scale: float = 2.0):
        verts = _disk(radius=radius, segments=segments, y=0.0, uv_scale=uv_scale)
        tex = make_water_texture()
        super().__init__(verts, tex)
