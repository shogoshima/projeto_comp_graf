"""Carrega texturas 2D e cubemap usando Pillow + OpenGL moderno.

Sem fixed-function: usa apenas glGenTextures/glTexImage2D/etc. Texturas 2D
sobem com mipmaps + filtro linear; cubemap sobe sem mipmaps com clamp_to_edge.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFile
from OpenGL.GL import (
    GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_REPEAT, GL_RGBA,
    GL_TEXTURE_2D, GL_TEXTURE_CUBE_MAP, GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
    GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_R, GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T, GL_UNSIGNED_BYTE,
    glBindTexture, glGenTextures, glGenerateMipmap, glTexImage2D,
    glTexParameteri,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# cache para não recarregar a mesma textura várias vezes
_TEX_CACHE: dict[str, int] = {}


def load_texture_2d(path: str | Path) -> int:
    """Cria uma textura 2D a partir de um arquivo, retornando o id."""
    key = str(Path(path).resolve())
    if key in _TEX_CACHE:
        return _TEX_CACHE[key]

    im = Image.open(path)
    im.load()
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    # OpenGL espera origem no canto inferior esquerdo; PIL no superior.
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    w, h = im.size
    data = im.tobytes()

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)

    _TEX_CACHE[key] = tex
    return tex


# Ordem GL p/ cubemap (positive_x, negative_x, positive_y, ...)
_FACE_TARGETS = (
    GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
)


def load_cubemap(face_paths: list[str | Path]) -> int:
    """Cria um cubemap. `face_paths` na ordem +X, -X, +Y, -Y, +Z, -Z."""
    if len(face_paths) != 6:
        raise ValueError("cubemap precisa de 6 faces")

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex)
    for target, path in zip(_FACE_TARGETS, face_paths):
        im = Image.open(path)
        im.load()
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        # NÃO inverter para cubemap — convenção OpenGL difere de 2D
        w, h = im.size
        glTexImage2D(target, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, im.tobytes())

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
    return tex
