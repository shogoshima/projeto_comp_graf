"""Helpers de matrizes 4x4 (numpy) — pipeline moderno (sem glMatrix).

Convenção:
- Matrizes coluna-major (compatíveis com OpenGL via `transpose=GL_FALSE`
  quando armazenadas em row-major no numpy: passamos `glUniformMatrix4fv(..., GL_TRUE, ...)`
  ou usamos `.T` antes de enviar; aqui retornamos row-major e o shader_uniform
  usa `transpose=GL_TRUE`).
- Tudo float32.
"""
from __future__ import annotations

import math

import numpy as np

Mat4 = np.ndarray  # shape (4,4) float32
Vec3 = np.ndarray  # shape (3,)  float32


def vec3(x: float, y: float, z: float) -> Vec3:
    return np.array([x, y, z], dtype=np.float32)


def identity() -> Mat4:
    return np.eye(4, dtype=np.float32)


def translation(x: float, y: float, z: float) -> Mat4:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scale(sx: float, sy: float | None = None, sz: float | None = None) -> Mat4:
    if sy is None:
        sy = sx
    if sz is None:
        sz = sx
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def rotation_x(angle_rad: float) -> Mat4:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    m = np.eye(4, dtype=np.float32)
    m[1, 1] =  c; m[1, 2] = -s
    m[2, 1] =  s; m[2, 2] =  c
    return m


def rotation_y(angle_rad: float) -> Mat4:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] =  c; m[0, 2] =  s
    m[2, 0] = -s; m[2, 2] =  c
    return m


def rotation_z(angle_rad: float) -> Mat4:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] =  c; m[0, 1] = -s
    m[1, 0] =  s; m[1, 1] =  c
    return m


def perspective(fov_rad: float, aspect: float, znear: float, zfar: float) -> Mat4:
    """Projeção perspectiva clássica (right-handed, depth [-1, 1])."""
    f = 1.0 / math.tan(fov_rad * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4:
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    s = np.cross(f, up)
    s /= np.linalg.norm(s) + 1e-12
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -float(np.dot(s, eye))
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] =  float(np.dot(f, eye))
    return m


def trs(translation_v: Vec3 | tuple,
        rotation_xyz_rad: Vec3 | tuple,
        scale_v: Vec3 | tuple) -> Mat4:
    """Compõe T * Ry * Rx * Rz * S (ordem que costuma dar resultado intuitivo)."""
    tx, ty, tz = translation_v
    rx, ry, rz = rotation_xyz_rad
    sx, sy, sz = scale_v
    m = translation(tx, ty, tz)
    m = m @ rotation_y(ry)
    m = m @ rotation_x(rx)
    m = m @ rotation_z(rz)
    m = m @ scale(sx, sy, sz)
    return m
