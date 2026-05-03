"""Wrapper de programa GLSL (pipeline moderno).

Compila e linka vertex + fragment shaders, expõe set_* para uniforms.
Sem glBegin/glEnd, sem matriz fixa.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_COMPILE_STATUS, GL_FALSE, GL_FRAGMENT_SHADER, GL_LINK_STATUS, GL_TRUE,
    GL_VERTEX_SHADER,
    glAttachShader, glCompileShader, glCreateProgram, glCreateShader,
    glDeleteShader, glGetProgramInfoLog, glGetProgramiv, glGetShaderInfoLog,
    glGetShaderiv, glGetUniformLocation, glLinkProgram, glShaderSource,
    glUniform1f, glUniform1i, glUniform3fv, glUniformMatrix4fv, glUseProgram,
)


def _compile(stage: int, src: str) -> int:
    sh = glCreateShader(stage)
    glShaderSource(sh, src)
    glCompileShader(sh)
    if glGetShaderiv(sh, GL_COMPILE_STATUS) != GL_TRUE:
        log = glGetShaderInfoLog(sh).decode()
        kind = "VERTEX" if stage == GL_VERTEX_SHADER else "FRAGMENT"
        raise RuntimeError(f"[{kind} shader] compile error:\n{log}")
    return sh


class Shader:
    def __init__(self, vs_src: str, fs_src: str):
        vs = _compile(GL_VERTEX_SHADER, vs_src)
        fs = _compile(GL_FRAGMENT_SHADER, fs_src)
        prog = glCreateProgram()
        glAttachShader(prog, vs)
        glAttachShader(prog, fs)
        glLinkProgram(prog)
        if glGetProgramiv(prog, GL_LINK_STATUS) != GL_TRUE:
            log = glGetProgramInfoLog(prog).decode()
            raise RuntimeError(f"[program] link error:\n{log}")
        glDeleteShader(vs)
        glDeleteShader(fs)
        self.id = prog
        self._loc_cache: dict[str, int] = {}

    @classmethod
    def from_files(cls, vs_path: Path, fs_path: Path) -> "Shader":
        return cls(Path(vs_path).read_text(), Path(fs_path).read_text())

    def use(self) -> None:
        glUseProgram(self.id)

    def _loc(self, name: str) -> int:
        loc = self._loc_cache.get(name)
        if loc is None:
            loc = glGetUniformLocation(self.id, name)
            self._loc_cache[name] = loc
        return loc

    # passamos transpose=GL_TRUE porque numpy é row-major e GL espera col-major
    def set_mat4(self, name: str, m: np.ndarray) -> None:
        loc = self._loc(name)
        if loc < 0:
            return
        glUniformMatrix4fv(loc, 1, GL_TRUE, np.ascontiguousarray(m, dtype=np.float32))

    def set_int(self, name: str, v: int) -> None:
        loc = self._loc(name)
        if loc >= 0:
            glUniform1i(loc, v)

    def set_float(self, name: str, v: float) -> None:
        loc = self._loc(name)
        if loc >= 0:
            glUniform1f(loc, float(v))

    def set_vec3(self, name: str, x: float, y: float, z: float) -> None:
        loc = self._loc(name)
        if loc >= 0:
            glUniform3fv(loc, 1, np.array([x, y, z], dtype=np.float32))
