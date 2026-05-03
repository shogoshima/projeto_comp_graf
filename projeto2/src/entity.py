"""Entity = Mesh + transform (posição/rotação/escala) + flags de render.

Cada Entity calcula sua própria matriz de modelo e desenha o mesh dela.
"""
from __future__ import annotations

import numpy as np
from OpenGL.GL import (
    GL_BACK, GL_CULL_FACE,
    glCullFace, glDisable, glEnable,
)

from src import transforms as T
from src.mesh import Mesh


class Entity:
    def __init__(
        self,
        mesh: Mesh,
        position: tuple[float, float, float] = (0, 0, 0),
        rotation: tuple[float, float, float] = (0, 0, 0),  # radianos
        scale: tuple[float, float, float] = (1, 1, 1),
        disable_culling: bool = False,
    ):
        self.mesh = mesh
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.scale    = np.array(scale, dtype=np.float32)
        self.disable_culling = disable_culling

    def model_matrix(self) -> np.ndarray:
        return T.trs(tuple(self.position.tolist()),
                     tuple(self.rotation.tolist()),
                     tuple(self.scale.tolist()))

    def draw(self, shader) -> None:
        if self.disable_culling:
            glDisable(GL_CULL_FACE)
        shader.set_mat4("u_model", self.model_matrix())
        self.mesh.draw(shader)
        if self.disable_culling:
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
