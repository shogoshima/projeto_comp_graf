"""
Computação Gráfica — Projeto 2
Cena 3D: "Pescaria de Luxo" 

Pipeline moderno do OpenGL: VAO/VBO + GLSL + matrizes em numpy.
SEM iluminação (proibido pelo PDF).

Controles:
  Câmera (FPS):
    W A S D            — andar
    Espaço / Shift     — subir / descer
    Mouse              — olhar em volta
    ESC                — sair
  Transformações (regra 7):
    Setas (↑↓←→)       — translação do BARCO no lago
    R / T              — rotação do POLVO
    + / -              — escala do CAVALO-MARINHO
    Z / X              — escala do Barco
  Visualização:
    P                  — toggle wireframe
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import glfw
import numpy as np
from OpenGL.GL import (
    GL_BACK, GL_COLOR_BUFFER_BIT, GL_CULL_FACE, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_FILL, GL_FRONT_AND_BACK, GL_LESS, GL_LINE,
    glClear, glClearColor, glCullFace, glDepthFunc, glEnable, glPolygonMode,
    glViewport,
)

from src import transforms as T
from src.camera import FpsCamera
from src.scene import (
    BOAT_PIVOT_Z, Scene, SKY_HEIGHT, WORLD_HALF, LAKE_CENTER, LAKE_RADIUS, WATER_Y,
)
from src.shader import Shader

ROOT = Path(__file__).resolve().parent
SHADERS = ROOT / "shaders"

WIDTH, HEIGHT = 700, 700
TITLE = "Pescaria de Luxo - Computação Gráfica P2"


class InputState:
    def __init__(self):
        self.first_mouse = True
        self.last_x = WIDTH / 2
        self.last_y = HEIGHT / 2
        self.wireframe = False
        self.p_was_down = False


def make_window():
    if not glfw.init():
        raise RuntimeError("Falha ao inicializar GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    win = glfw.create_window(WIDTH, HEIGHT, TITLE, None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("Falha ao criar janela GLFW")
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    return win


def main() -> int:
    win = make_window()
    inp = InputState()

    print("[init] carregando shaders...")
    basic = Shader.from_files(SHADERS / "basic.vs", SHADERS / "basic.fs")
    sky_shader = Shader.from_files(SHADERS / "skybox.vs", SHADERS / "skybox.fs")

    print("[init] carregando cena...")
    scene = Scene()

    cam = FpsCamera(
        position=(0.0, 2.5, 25.0),
        yaw_deg=-90.0, pitch_deg=-5.0,
        speed=12.0, sensitivity=0.13,
        bounds=((-WORLD_HALF + 2, WORLD_HALF - 2),
                (0.7, SKY_HEIGHT - 5),
                (-WORLD_HALF + 2, WORLD_HALF - 2)),
    )

    glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_DISABLED)

    def on_mouse(window, xpos, ypos):
        if inp.first_mouse:
            inp.last_x, inp.last_y = xpos, ypos
            inp.first_mouse = False
        dx = xpos - inp.last_x
        dy = ypos - inp.last_y
        inp.last_x, inp.last_y = xpos, ypos
        cam.process_mouse(dx, dy)

    def on_resize(window, w, h):
        glViewport(0, 0, w, h)

    glfw.set_cursor_pos_callback(win, on_mouse)
    glfw.set_framebuffer_size_callback(win, on_resize)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glClearColor(0.05, 0.07, 0.10, 1.0)

    # parâmetros das transformações por teclado
    boat_speed = 4.0
    boat_turn_speed = 6.0
    octopus_rot_speed = 1.6
    seahorse_scale_speed = 0.6
    seahorse_min, seahorse_max = 0.15, 1.5

    print("[init] OK. Entrando no loop.")
    last_t = glfw.get_time()
    while not glfw.window_should_close(win):
        now = glfw.get_time()
        dt = now - last_t
        last_t = now

        # ---------------- INPUT ----------------
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(win, True)

        # câmera
        fwd = (glfw.get_key(win, glfw.KEY_W) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_S) == glfw.PRESS)
        rgt = (glfw.get_key(win, glfw.KEY_D) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_A) == glfw.PRESS)
        upd = (glfw.get_key(win, glfw.KEY_SPACE) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS)
        cam.process_keyboard(dt, fwd, rgt, upd)

        # BARCO (translação) — setas
        bdx = (glfw.get_key(win, glfw.KEY_RIGHT) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_LEFT) == glfw.PRESS)
        bdz = (glfw.get_key(win, glfw.KEY_DOWN)  == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_UP)   == glfw.PRESS)
        if bdx or bdz:
            desired_yaw = math.atan2(bdx, bdz)
            current_yaw = float(scene.boat.rotation[1])
            # smooth yaw to avoid pivot snaps when changing direction
            delta = (desired_yaw - current_yaw + math.pi) % (2.0 * math.pi) - math.pi
            yaw_step = boat_turn_speed * dt
            if abs(delta) <= yaw_step:
                new_yaw = desired_yaw
            else:
                new_yaw = current_yaw + math.copysign(yaw_step, delta)

            scene.boat.position[0] += bdx * boat_speed * dt
            scene.boat.position[2] += bdz * boat_speed * dt
            # mantém o barco DENTRO do círculo do lago (não sai pro chão)
            cx, cz = LAKE_CENTER
            dx, dz = scene.boat.position[0] - cx, scene.boat.position[2] - cz
            r = math.sqrt(dx * dx + dz * dz)
            max_r = LAKE_RADIUS - 1.5
            if r > max_r:
                scene.boat.position[0] = cx + dx * (max_r / r)
                scene.boat.position[2] = cz + dz * (max_r / r)
            # vira a "proa" pra direção do movimento
            scene.boat.rotation[1] = new_yaw
            
        # BARCO (escala) — X / Z
        plus = (glfw.get_key(win, glfw.KEY_X) == glfw.PRESS)
        minus = (glfw.get_key(win, glfw.KEY_Z) == glfw.PRESS)
        if plus or minus:
            d = (1 if plus else 0) - (1 if minus else 0)
            new_s = float(scene.boat.scale[0]) + d * 0.8 * dt
            new_s = max(0.05, min(0.5, new_s))
            scene.boat.scale = np.array([new_s, new_s, new_s], dtype=np.float32)

        # POLVO (rotação) — R / T
        srot = (glfw.get_key(win, glfw.KEY_T) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_R) == glfw.PRESS)
        if srot:
            scene.octopus.rotation[1] += srot * octopus_rot_speed * dt

        # CAVALO-MARINHO (escala) — + / -
        plus = (glfw.get_key(win, glfw.KEY_EQUAL) == glfw.PRESS or
                glfw.get_key(win, glfw.KEY_KP_ADD) == glfw.PRESS)
        minus = (glfw.get_key(win, glfw.KEY_MINUS) == glfw.PRESS or
                 glfw.get_key(win, glfw.KEY_KP_SUBTRACT) == glfw.PRESS)
        if plus or minus:
            d = (1 if plus else 0) - (1 if minus else 0)
            new_s = float(scene.seahorse.scale[0]) + d * seahorse_scale_speed * dt
            new_s = max(seahorse_min, min(seahorse_max, new_s))
            scene.seahorse.scale = np.array([new_s, new_s, new_s], dtype=np.float32)

        # WIREFRAME toggle (P)
        p_now = glfw.get_key(win, glfw.KEY_P) == glfw.PRESS
        if p_now and not inp.p_was_down:
            inp.wireframe = not inp.wireframe
            print(f"[wireframe] {'ON' if inp.wireframe else 'OFF'}")
        inp.p_was_down = p_now
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if inp.wireframe else GL_FILL)

        # ---------------- UPDATE ----------------
        scene.update(dt)

        # ---------------- RENDER ----------------
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        w, h = glfw.get_framebuffer_size(win)
        proj = T.perspective(math.radians(60.0), w / max(h, 1), 0.1, 1500.0)
        view = cam.view_matrix()

        # 1) Skybox primeiro (com view sem translação)
        view_no_t = view.copy()
        view_no_t[0, 3] = view_no_t[1, 3] = view_no_t[2, 3] = 0.0
        scene.skybox.draw(sky_shader, view_no_t, proj)

        # 2) Resto da cena (com basic shader)
        basic.use()
        basic.set_mat4("u_proj", proj)
        basic.set_mat4("u_view", view)
        basic.set_int("u_wireframe", 1 if inp.wireframe else 0)
        scene.draw(basic, wireframe=inp.wireframe)

        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
