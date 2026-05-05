"""SkyBox (cubemap) — pipeline moderno.

Renderiza um cubo de tamanho 1 ao redor da câmera. A "view matrix" é passada
sem translação para o cubo permanecer centrado na câmera.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_CULL_FACE, GL_FALSE, GL_FLOAT,
    GL_LEQUAL, GL_LESS, GL_STATIC_DRAW, GL_TEXTURE0, GL_TEXTURE_CUBE_MAP,
    GL_TRIANGLES,
    glActiveTexture, glBindBuffer, glBindTexture, glBindVertexArray,
    glBufferData, glDepthFunc, glDisable, glDrawArrays,
    glEnableVertexAttribArray, glEnable, glGenBuffers, glGenVertexArrays,
    glVertexAttribPointer,
)

from src.texture import load_cubemap


# 36 vértices (12 triângulos) — cubo unitário centrado na origem.
_CUBE_VERTICES = np.array([
    # +X
     1, -1, -1,   1,  1, -1,   1,  1,  1,
     1, -1, -1,   1,  1,  1,   1, -1,  1,
    # -X
    -1, -1,  1,  -1,  1,  1,  -1,  1, -1,
    -1, -1,  1,  -1,  1, -1,  -1, -1, -1,
    # +Y
    -1,  1, -1,   1,  1, -1,   1,  1,  1,
    -1,  1, -1,   1,  1,  1,  -1,  1,  1,
    # -Y
    -1, -1,  1,   1, -1,  1,   1, -1, -1,
    -1, -1,  1,   1, -1, -1,  -1, -1, -1,
    # +Z
    -1, -1,  1,   1, -1,  1,   1,  1,  1,
    -1, -1,  1,   1,  1,  1,  -1,  1,  1,
    # -Z
     1, -1, -1,  -1, -1, -1,  -1,  1, -1,
     1, -1, -1,  -1,  1, -1,   1,  1, -1,
], dtype=np.float32)


class Skybox:
    def __init__(self, face_paths: list[str | Path]):
        self.cubemap = load_cubemap(face_paths)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, _CUBE_VERTICES.nbytes, _CUBE_VERTICES, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glBindVertexArray(0)

    def draw(self, shader, view_no_translation: np.ndarray, proj: np.ndarray) -> None:
        glDisable(GL_CULL_FACE)
        glDepthFunc(GL_LEQUAL)
        shader.use()
        shader.set_mat4("u_view", view_no_translation)
        shader.set_mat4("u_proj", proj)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.cubemap)
        shader.set_int("u_cubemap", 0)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glBindVertexArray(0)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
