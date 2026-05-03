"""Câmera FPS com clamping de bordas (céu/terreno).

- Mouse controla yaw/pitch (pitch limitado a ±89°).
- Teclado: WASD + Shift/Espaço.
- Posição é restrita a um AABB ((xmin,xmax),(ymin,ymax),(zmin,zmax))
  para cumprir o requisito 9 do PDF: não sair dos limites do céu/terreno.
"""
from __future__ import annotations

import math

import numpy as np

from src import transforms as T


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


class FpsCamera:
    def __init__(
        self,
        position: tuple[float, float, float] = (0.0, 1.7, 5.0),
        yaw_deg: float = -90.0,
        pitch_deg: float = 0.0,
        speed: float = 8.0,
        sensitivity: float = 0.13,
        bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    ):
        self.position = np.array(position, dtype=np.float32)
        self.yaw = yaw_deg
        self.pitch = pitch_deg
        self.speed = speed
        self.sensitivity = sensitivity
        self.bounds = bounds  # ((xmin,xmax),(ymin,ymax),(zmin,zmax)) ou None
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._update_vectors()

    # ---------------------------------------------------------- #
    # vetores derivados
    # ---------------------------------------------------------- #
    def _update_vectors(self) -> None:
        cy, cp = math.cos(math.radians(self.yaw)), math.cos(math.radians(self.pitch))
        sy, sp = math.sin(math.radians(self.yaw)), math.sin(math.radians(self.pitch))
        self.front = _norm(np.array([cy * cp, sp, sy * cp], dtype=np.float32))
        self.right = _norm(np.cross(self.front, self.world_up))
        self.up    = _norm(np.cross(self.right, self.front))

    def view_matrix(self) -> np.ndarray:
        return T.look_at(self.position, self.position + self.front, self.up)

    # ---------------------------------------------------------- #
    # input handlers
    # ---------------------------------------------------------- #
    def process_mouse(self, dx: float, dy: float) -> None:
        self.yaw   += dx * self.sensitivity
        self.pitch -= dy * self.sensitivity  # invertido p/ comportamento natural
        self.pitch = max(-89.0, min(89.0, self.pitch))
        self._update_vectors()

    def process_keyboard(self, dt: float, fwd: int, rgt: int, upd: int) -> None:
        """fwd, rgt, upd ∈ {-1, 0, 1}."""
        if fwd == 0 and rgt == 0 and upd == 0:
            return
        # movimento na horizontal usa front projetado em XZ (estilo FPS clássico)
        flat_front = np.array([self.front[0], 0.0, self.front[2]], dtype=np.float32)
        flat_front = _norm(flat_front) if np.linalg.norm(flat_front) > 1e-6 else self.front
        flat_right = _norm(np.cross(flat_front, self.world_up))
        delta = (flat_front * fwd + flat_right * rgt + self.world_up * upd) * (self.speed * dt)
        self.position = self.position + delta
        self._clamp()

    def _clamp(self) -> None:
        if not self.bounds:
            return
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.bounds
        self.position[0] = max(xmin, min(xmax, float(self.position[0])))
        self.position[1] = max(ymin, min(ymax, float(self.position[1])))
        self.position[2] = max(zmin, min(zmax, float(self.position[2])))
