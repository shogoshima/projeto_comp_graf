"""Mesh = lista de SubMeshes (cada uma com seu material) num único VAO/VBO.

Cada SubMesh é um intervalo (start, count) dentro do VBO, desenhado com
glDrawArrays. Pipeline moderno apenas: VAO, VBO, glVertexAttribPointer.

Vértices são intercalados: [px, py, pz, u, v, nx, ny, nz] (8 floats).
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_FALSE, GL_FLOAT, GL_STATIC_DRAW, GL_TEXTURE0,
    GL_TEXTURE_2D, GL_TRIANGLES,
    glActiveTexture, glBindBuffer, glBindTexture, glBindVertexArray,
    glBufferData, glDrawArrays, glEnableVertexAttribArray, glGenBuffers,
    glGenVertexArrays, glVertexAttribPointer,
)

from src import obj_loader
from src.texture import load_texture_2d


# tamanho de um vértice em bytes (8 floats × 4 bytes)
VERTEX_STRIDE = 8 * 4


@dataclass
class DrawRange:
    material_name: str
    start: int           # primeiro vértice no VBO
    count: int           # número de vértices (= 3 * num_triangulos)
    texture_id: int      # texture object id
    kd: tuple[float, float, float] = (1.0, 1.0, 1.0)


class Mesh:
    """Um VAO/VBO com vários DrawRanges (um por material)."""

    def __init__(self, draw_ranges: list[DrawRange], vao: int, vbo: int):
        self.draw_ranges = draw_ranges
        self.vao = vao
        self.vbo = vbo

    @classmethod
    def from_obj(cls, obj_path: str | Path) -> "Mesh":
        submeshes, materials = obj_loader.load_obj(obj_path)
        if not submeshes:
            raise RuntimeError(f"{obj_path}: nenhuma sub-mesh encontrada")

        # concatena todos os submesh.vertices num único VBO
        all_arrays = [s.vertices for s in submeshes]
        all_data = np.concatenate(all_arrays, axis=0).astype(np.float32, copy=False)

        # cria VAO/VBO
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, all_data.nbytes, all_data, GL_STATIC_DRAW)

        # location 0: position vec3
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE,
                              ctypes.c_void_p(0))
        # location 1: uv vec2
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, VERTEX_STRIDE,
                              ctypes.c_void_p(3 * 4))
        # location 2: normal vec3 (não usado nos shaders sem iluminação,
        #            mas mantido para futura extensão)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE,
                              ctypes.c_void_p(5 * 4))

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # constrói os DrawRanges
        draw_ranges: list[DrawRange] = []
        cursor = 0
        for s in submeshes:
            n_verts = s.vertices.shape[0]
            mat = materials.get(s.material)
            if mat is None or not mat.map_kd:
                # fallback: textura branca 1x1 (geramos sob demanda no _white_pixel())
                tex_id = _white_pixel()
                kd = mat.kd if mat else (1.0, 1.0, 1.0)
            else:
                tex_id = load_texture_2d(mat.map_kd)
                kd = mat.kd
            draw_ranges.append(DrawRange(
                material_name=s.material,
                start=cursor,
                count=n_verts,
                texture_id=tex_id,
                kd=kd,
            ))
            cursor += n_verts

        return cls(draw_ranges=draw_ranges, vao=vao, vbo=vbo)

    def draw(self, shader) -> None:
        """Desenha todos os DrawRanges, vinculando a textura de cada um."""
        glBindVertexArray(self.vao)
        for r in self.draw_ranges:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, r.texture_id)
            shader.set_int("u_tex", 0)
            shader.set_vec3("u_kd", *r.kd)
            glDrawArrays(GL_TRIANGLES, r.start, r.count)
        glBindVertexArray(0)


# --------------------------------------------------------------------------- #
# Pixel branco para meshes sem textura (fallback)
# --------------------------------------------------------------------------- #
_WHITE_PIXEL: int | None = None


def _white_pixel() -> int:
    global _WHITE_PIXEL
    if _WHITE_PIXEL is not None:
        return _WHITE_PIXEL
    from OpenGL.GL import (
        GL_LINEAR, GL_REPEAT, GL_RGBA, GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
        GL_UNSIGNED_BYTE,
        glGenTextures, glTexImage2D, glTexParameteri,
    )
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    pixel = (ctypes.c_ubyte * 4)(255, 255, 255, 255)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixel)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glBindTexture(GL_TEXTURE_2D, 0)
    _WHITE_PIXEL = tex
    return tex
