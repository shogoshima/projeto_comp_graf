"""Cena "Pescaria de Luxo" — monta TODAS as entidades.

Layout (vista de cima, X→leste, Z→sul, Y→cima):

    z=-WORLD_HALF                                                z=+WORLD_HALF
   ┌────────────────────────────────────────────────────────────────────────┐
   │                  ÁRVORES (Orange + Green espalhadas)                   │
   │                                                                        │
   │       (-25, -8) [arv]                                                  │
   │                                                                        │
   │                      ┌────────────┐                                    │
   │   [arv]              │   CABANA   │   [LAGO  centrado em (X+, 0)]      │
   │                      │  (interno) │       barco, polvo, cavalomar.     │
   │                      └────────────┘       → modelos animados/transf.   │
   │       [arv]                                                            │
   │                                                                        │
   │                  ÁRVORES (Orange + Green espalhadas)                   │
   └────────────────────────────────────────────────────────────────────────┘

Convenção dos modelos (para cada exemplo):
- import .obj de assets/_obj_exports/<name>/<name>.obj
- center_xz e/ou floor_y opcionais (tratados antes de exportar via tools)

Transformações por teclado (regra 7):
- Translação  → BARCO (setas ↑↓←→ dentro do lago)
- Rotação     → POLVO (R / T)
- Escala      → BARCO (Z / X)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
from OpenGL.GL import GL_CULL_FACE, glCullFace, glDisable, glEnable, GL_BACK

from src.entity import Entity
from src.floor import GrassFloorWithHole, WaterDisk
from src.mesh import Mesh
from src.skybox import Skybox

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
SKY_DIR = ROOT / "sky"

# limites do mundo (chão / céu / câmera)
WORLD_HALF = 80.0
SKY_HEIGHT = 60.0

# raio externo da cabana (delimitador) — usado pra cavar o buraco no piso de grama.
# Cabana escalada tem largura X ~6m e profundidade Z ~10m; usamos um raio que
# encaixa o disco de madeira na largura X (Z é coberto pelas paredes da cabana).
HUT_FOOTPRINT_RADIUS = 3.5

HUT_CENTER  = (-14.0, 0.0)  # (x, z) — cabana centrada à esquerda do mundo
LAKE_CENTER = ( 14.0, 0.0)  # (x, z) — lago centrado à direita do mundo
LAKE_RADIUS = 18.0
WATER_Y = 0.05               # ligeiramente acima do solo p/ evitar z-fighting
BOAT_Y = WATER_Y           # altura do barco (pode ser animada com a água)
BOAT_PIVOT_Z = -5.0  # pivô do barco (para inclinar melhor) — 5m à frente do centro do modelo


class JumpingFish:
    """Peixe que pula periodicamente fora d'água em arco parabólico."""

    def __init__(self, entity: Entity, lake_cx: float, lake_cz: float,
                 lake_r: float, rng: np.random.Generator,
                 jump_height: float = 3.0, jump_dist: float = 5.0,
                 jump_dur: float = 1.0):
        self.entity = entity
        self.cx, self.cz, self.r = lake_cx, lake_cz, lake_r
        self.rng = rng
        self.jump_h = jump_height
        self.jump_d = jump_dist
        self.jump_dur = jump_dur

        angle = float(rng.uniform(0, math.tau))
        dist = float(rng.uniform(2, lake_r - 4))
        self.x = lake_cx + dist * math.cos(angle)
        self.z = lake_cz + dist * math.sin(angle)
        self.yaw = float(rng.uniform(0, math.tau))

        self.is_jumping = False
        self.timer = float(rng.uniform(0.5, 3.0))
        self.progress = 0.0
        entity.position[1] = WATER_Y - 5.0

    def update(self, dt: float) -> None:
        if self.is_jumping:
            self.progress += dt
            t = min(self.progress / self.jump_dur, 1.0)

            y = WATER_Y + self.jump_h * 4.0 * t * (1.0 - t)

            speed = self.jump_d / self.jump_dur
            self.x += math.cos(self.yaw) * speed * dt
            self.z -= math.sin(self.yaw) * speed * dt

            slope = 4.0 * self.jump_h * (1.0 - 2.0 * t) / self.jump_d
            pitch = math.atan(slope)

            self.entity.position[0] = self.x
            self.entity.position[1] = y
            self.entity.position[2] = self.z
            self.entity.rotation[1] = self.yaw
            self.entity.rotation[2] = pitch

            if self.progress >= self.jump_dur:
                self.is_jumping = False
                self.timer = float(self.rng.uniform(1.5, 4.0))
                self.entity.position[1] = WATER_Y - 5.0
                self.yaw += float(self.rng.uniform(-0.8, 0.8))
        else:
            self.timer -= dt
            if self.timer <= 0:
                end_x = self.x + math.cos(self.yaw) * self.jump_d
                end_z = self.z - math.sin(self.yaw) * self.jump_d
                if (end_x - self.cx) ** 2 + (end_z - self.cz) ** 2 > (self.r - 3.0) ** 2:
                    self.yaw += math.pi
                self.is_jumping = True
                self.progress = 0.0


class Scene:
    def __init__(self):
        # ---------------- Skybox ----------------
        # ordem GL: +X, -X, +Y, -Y, +Z, -Z
        self.skybox = Skybox([
            str(SKY_DIR / "px.png"),
            str(SKY_DIR / "nx.png"),
            str(SKY_DIR / "py.png"),
            str(SKY_DIR / "ny.png"),
            str(SKY_DIR / "pz.png"),
            str(SKY_DIR / "nz.png"),
        ])

        # ---------------- Pisos ----------------
        # Chão externo
        self.outdoor_floor = GrassFloorWithHole(
            world_half=WORLD_HALF,
            hole_radius=HUT_FOOTPRINT_RADIUS,
            hole_center=HUT_CENTER,
            segments=64,
            uv_scale=60.0,
        )

        # Chão interno (tapete)
        CARPET_SCALE = 0.08
        self.indoor_floor = Entity(
            Mesh.from_obj(str(ASSETS / "carpet" / "carpet.obj")),
            position=(HUT_CENTER[0] - 2.0, 0.1, HUT_CENTER[1] + 0.5),
            rotation=(0.0, math.radians(90), 0.0),
            scale=(CARPET_SCALE, CARPET_SCALE, CARPET_SCALE),
        )

        # Lago
        self.lake = WaterDisk(radius=LAKE_RADIUS, segments=96, uv_scale=6.0)
        self.lake.position = np.array([LAKE_CENTER[0], WATER_Y, LAKE_CENTER[1]],
                                      dtype=np.float32)

        # Cabana
        HUT_SCALE = 0.8
        self.hut = Entity(
            Mesh.from_obj(str(ASSETS / "hut" / "hut.obj")),
            position=(HUT_CENTER[0] + 5.6 * HUT_SCALE, 0.0, HUT_CENTER[1] + 0.19 * HUT_SCALE),
            scale=(HUT_SCALE, HUT_SCALE, HUT_SCALE),
            disable_culling=True,
        )

        # ---------------- Modelos EXTERNOS ----------------
        # Barco flutuando na superfície do lago. Translação por teclado.
        BOAT_SCALE = 0.1
        self.boat = Entity(
            Mesh.from_obj(str(ASSETS / "ponyo_boat" / "ponyo_boat.obj")),
            position=(LAKE_CENTER[0] - 4.0, BOAT_Y, LAKE_CENTER[1]),
            rotation=(math.radians(-90), math.radians(0), math.radians(0)),
            scale=(BOAT_SCALE, BOAT_SCALE, BOAT_SCALE),
            pivot=(0.0, 0.0, BOAT_PIVOT_Z),
        )

        # Polvo ao redor do lago, em pé (ligeiramente submerso). Rotação por teclado.
        OCTOPUS_SCALE = 8
        self.octopus = Entity(
            Mesh.from_obj(str(ASSETS / "octopus" / "octopus.obj")),
            position=(LAKE_CENTER[0] + 8.0, WATER_Y + 0.05, LAKE_CENTER[1] - 2.0),
            rotation=(0.0, 0.0, 0.0),
            scale=(OCTOPUS_SCALE, OCTOPUS_SCALE, OCTOPUS_SCALE),
        )

        rng = np.random.default_rng(seed=2024)
        forbidden = [
            (HUT_CENTER[0],  HUT_CENTER[1],  HUT_FOOTPRINT_RADIUS + 8.0),
            (LAKE_CENTER[0], LAKE_CENTER[1], LAKE_RADIUS + 4.0),
        ]

        def far_enough(x: float, z: float) -> bool:
            for fx, fz, fr in forbidden:
                if (x - fx) ** 2 + (z - fz) ** 2 < fr * fr:
                    return False
            return True

        def random_outdoor_pos() -> tuple[float, float] | None:
            for _ in range(100):
                x = float(rng.uniform(-WORLD_HALF + 5, WORLD_HALF - 5))
                z = float(rng.uniform(-WORLD_HALF + 5, WORLD_HALF - 5))
                if far_enough(x, z):
                    return x, z
            return None

        spruce_meshes = [
            Mesh.from_obj(str(ASSETS / "tree_spruce_small_01" / "tree_spruce_small_01.obj")),
            Mesh.from_obj(str(ASSETS / "tree_spruce_tiny_01"  / "tree_spruce_tiny_01.obj")),
        ]
        bush_set_meshes = [
            Mesh.from_obj(str(ASSETS / "bush_average"       / "bush_average.obj")),
            Mesh.from_obj(str(ASSETS / "bush_group_average" / "bush_group_average.obj")),
        ]
        stone_meshes = [
            Mesh.from_obj(str(ASSETS / "stone_average_01"    / "stone_average_01.obj")),
            Mesh.from_obj(str(ASSETS / "stone_group_average" / "stone_group_average.obj")),
        ]

        self.outdoor_props: List[Entity] = []

        for i in range(300):
            pos = random_outdoor_pos()
            if pos is None:
                continue
            x, z = pos
            scale = float(rng.uniform(0.8, 1.0))
            yaw = float(rng.uniform(0, math.tau))
            self.outdoor_props.append(Entity(spruce_meshes[i % 2], position=(x, 0.0, z),
                                      rotation=(0.0, yaw, 0.0), scale=(scale, scale, scale)))

        for i in range(200):
            pos = random_outdoor_pos()
            if pos is None:
                continue
            x, z = pos
            scale = float(rng.uniform(0.8, 1.0))
            yaw = float(rng.uniform(0, math.tau))
            self.outdoor_props.append(Entity(bush_set_meshes[i % 2], position=(x, 0.0, z),
                                      rotation=(0.0, yaw, 0.0), scale=(scale, scale, scale)))

        for i in range(100):
            pos = random_outdoor_pos()
            if pos is None:
                continue
            x, z = pos
            scale = float(rng.uniform(1.0, 2.0))
            yaw = float(rng.uniform(0, math.tau))
            self.outdoor_props.append(Entity(stone_meshes[i % 2], position=(x, 0.0, z),
                                      rotation=(0.0, yaw, 0.0), scale=(scale, scale, scale)))

        # ---------------- Modelos INTERNOS ----------------
        # Mesa
        TABLE_SCALE = 0.025
        self.table = Entity(
            Mesh.from_obj(str(ASSETS / "table" / "table.obj")),
            position=(HUT_CENTER[0] + 1.0, 0.0, HUT_CENTER[1] + 3.5),
            rotation=(0.0, math.radians(0), 0.0),
            scale=(TABLE_SCALE, TABLE_SCALE, TABLE_SCALE),
        )
        
        # Ramen ponyo
        RAMEN_SCALE = 0.003
        self.ramen = Entity(
            Mesh.from_obj(str(ASSETS / "ramen" / "ramen.obj")),
            position=(HUT_CENTER[0] + 1.0, 1.65, HUT_CENTER[1] + 3.5),
            rotation=(0.0, math.radians(0), 0.0),
            scale=(RAMEN_SCALE, RAMEN_SCALE, RAMEN_SCALE),
        )

        # Balde apoiado no chão de madeira, perto da parede oposta à vara.
        BUCKET_SCALE = 0.2
        self.bucket = Entity(
            Mesh.from_obj(str(ASSETS / "bucket_ponyo" / "bucket_ponyo.obj")),
            position=(HUT_CENTER[0] +1.8, -0.45, HUT_CENTER[1] -1.2),
            rotation=(0.0, math.radians(20), 0.0),
            scale=(BUCKET_SCALE, BUCKET_SCALE, BUCKET_SCALE),
        )

        # Lanterna deitada no chão
        FLASHLIGHT_SCALE = 0.02
        self.flashlight = Entity(
            Mesh.from_obj(str(ASSETS / "flashlight" / "flashlight.obj")),
            position=(HUT_CENTER[0] -0.5, 0.3, HUT_CENTER[1] - 3.0),
            rotation=(0.0, math.radians(120), 0.0),
            scale=(FLASHLIGHT_SCALE, FLASHLIGHT_SCALE, FLASHLIGHT_SCALE),
        )

        self.indoor_extras: List[Entity] = [self.bucket, self.flashlight, self.ramen]

        # ---------------- Peixes pulando ----------------
        FISH_SCALE = 0.5
        fish_mesh = Mesh.from_obj(str(ASSETS / "fish" / "pez3.obj"))
        fish_params = [
            dict(jump_height=2.5, jump_dist=4.0, jump_dur=0.9),
            dict(jump_height=3.5, jump_dist=6.0, jump_dur=1.1),
            dict(jump_height=3.0, jump_dist=5.0, jump_dur=1.0),
        ]
        self.jumping_fish: List[JumpingFish] = []
        for params in fish_params:
            ent = Entity(
                fish_mesh,
                scale=(FISH_SCALE, FISH_SCALE, FISH_SCALE),
                disable_culling=True,
            )
            self.jumping_fish.append(
                JumpingFish(ent, LAKE_CENTER[0], LAKE_CENTER[1],
                            LAKE_RADIUS, rng, **params)
            )

        # ---------------- Listas para draw ----------------
        self.outdoor_entities: List[Entity] = [
            self.boat, self.octopus, *self.outdoor_props,
            *[jf.entity for jf in self.jumping_fish],
        ]
        self.indoor_entities: List[Entity] = [
            self.table, *self.indoor_extras,
        ]

    # --------------------------------------------------------------- #
    # Update / draw
    # --------------------------------------------------------------- #
    def update(self, dt: float) -> None:
        """Animações sutis: barco oscila com a água, peixes pulam."""
        t = (self._time if hasattr(self, "_time") else 0.0) + dt
        self._time = t
        # bobbing do barco
        self.boat.position[1] = BOAT_Y + 0.025 * math.sin(t * 0.4)
        self.boat.rotation[0] = math.radians(-90) + math.radians(2.0) * math.sin(t * 1.0)
        # peixes
        for jf in self.jumping_fish:
            jf.update(dt)

    def draw(self, shader, wireframe: bool = False) -> None:
        # pisos
        self.outdoor_floor.draw(shader)
        self.indoor_floor.draw(shader)
        self.lake.draw(shader)

        # cabana (delimitador, com cull off)
        self.hut.draw(shader)

        # demais
        for e in self.outdoor_entities:
            e.draw(shader)
        for e in self.indoor_entities:
            e.draw(shader)
