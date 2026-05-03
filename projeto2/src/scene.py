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
- Escala      → CAVALO-MARINHO (+ / -)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
from OpenGL.GL import GL_CULL_FACE, glCullFace, glDisable, glEnable, GL_BACK

from src.entity import Entity
from src.floor import GrassFloorWithHole, WoodFloorDisk, WaterDisk
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

# Lago — círculo de água deslocado em +X p/ não conflitar com a cabana
LAKE_CENTER = (28.0, 0.0)   # (x, z)
LAKE_RADIUS = 18.0
WATER_Y = 0.05               # ligeiramente acima do solo p/ evitar z-fighting


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
        # Chão externo (grama com buraco circular para a cabana)
        self.outdoor_floor = GrassFloorWithHole(
            world_half=WORLD_HALF,
            hole_radius=HUT_FOOTPRINT_RADIUS,
            hole_center=(0.0, 0.0),
            segments=64,
            uv_scale=24.0,
        )

        # Chão interno (madeira) — disco que preenche o buraco
        self.indoor_floor = WoodFloorDisk(
            radius=HUT_FOOTPRINT_RADIUS - 0.05,  # leve recuo p/ caber dentro da parede
            segments=64,
            uv_scale=3.5,
        )
        self.indoor_floor.position = np.array([0.0, 0.05, 0.0], dtype=np.float32)

        # Lago (disco de água)
        self.lake = WaterDisk(radius=LAKE_RADIUS, segments=96, uv_scale=2.5)
        self.lake.position = np.array([LAKE_CENTER[0], WATER_Y, LAKE_CENTER[1]],
                                      dtype=np.float32)

        # ---------------- Cabana (delimitador, NÃO conta nos 6) ----------------
        # Cabana raw é gigante (~13x13x22): centro em (-5.6, 6.4, -0.2) e ~13m
        # de altura. Escalamos p/ ~5.9m de altura e DESLOCAMOS em +X p/ que o
        # centro do modelo caia na origem (assim o buraco circular do piso casa).
        # culling desligado pra ver paredes por dentro.
        HUT_SCALE = 0.45
        self.hut = Entity(
            Mesh.from_obj(str(ASSETS / "hut" / "hut.obj")),
            position=(5.6 * HUT_SCALE, 0.0, 0.19 * HUT_SCALE),
            scale=(HUT_SCALE, HUT_SCALE, HUT_SCALE),
            disable_culling=True,
        )

        # ---------------- Modelos EXTERNOS ----------------
        # Barco — flutuando na superfície do lago. Translação por teclado.
        # Modelo veio em escala enorme (DAZ Studio em centímetros), rescalando p/
        # caber no lago. Posição em (lake_x, water_level, lake_z).
        self.boat = Entity(
            Mesh.from_obj(str(ASSETS / "boat" / "boat.obj")),
            position=(LAKE_CENTER[0] - 4.0, WATER_Y + 0.1, LAKE_CENTER[1]),
            rotation=(0.0, math.radians(-90), 0.0),
            scale=(0.012, 0.012, 0.012),  # asset gigante (DAZ) → ~3m
        )

        # Polvo — ao redor do lago, em pé (ligeiramente submerso). Rotação por teclado.
        self.octopus = Entity(
            Mesh.from_obj(str(ASSETS / "octopus" / "octopus.obj")),
            position=(LAKE_CENTER[0] + 8.0, WATER_Y + 0.05, LAKE_CENTER[1] - 2.0),
            rotation=(0.0, 0.0, 0.0),
            scale=(8.0, 8.0, 8.0),
        )

        # Cavalo-marinho — perto da margem, escala por teclado.
        # Modelo é "alto" no eixo Y (~9 unidades), então floor_y é ajustado por position.
        self.seahorse = Entity(
            Mesh.from_obj(str(ASSETS / "seahorse" / "seahorse.obj")),
            position=(LAKE_CENTER[0] - 12.0, WATER_Y, LAKE_CENTER[1] + 4.0),
            rotation=(0.0, math.radians(45), 0.0),
            scale=(0.4, 0.4, 0.4),
        )

        # Árvores — gerar instâncias espalhadas.
        self.tree_mesh = Mesh.from_obj(str(ASSETS / "tree" / "tree.obj"))

        rng = np.random.default_rng(seed=2024)
        self.trees: List[Entity] = []
        n_trees_each = 300
        forbidden = [
            (0.0, 0.0, HUT_FOOTPRINT_RADIUS + 4.0),
            (LAKE_CENTER[0], LAKE_CENTER[1], LAKE_RADIUS + 4.0),
        ]
        TREE_Y_OFFSET = -1.17

        def far_enough(x: float, z: float) -> bool:
            for fx, fz, fr in forbidden:
                if (x - fx) ** 2 + (z - fz) ** 2 < fr * fr:
                    return False
            return True

        def random_tree_pos() -> tuple[float, float]:
            for _ in range(50):
                x = float(rng.uniform(-WORLD_HALF + 5, WORLD_HALF - 5))
                z = float(rng.uniform(-WORLD_HALF + 5, WORLD_HALF - 5))
                if far_enough(x, z):
                    return x, z
            return 0.0, 0.0

        for _ in range(n_trees_each):
            x, z = random_tree_pos()
            scale = float(rng.uniform(4.0, 5.0))
            yaw = float(rng.uniform(0, math.tau))
            self.trees.append(Entity(
                self.tree_mesh,
                position=(x, TREE_Y_OFFSET * scale, z),
                rotation=(0.0, yaw, 0.0),
                scale=(scale, scale, scale),
            ))

        # ---------------- Modelos INTERNOS ----------------
        # Vara de pesca — encostada na parede da cabana.
        # Raw é ~1.5m de comprimento: scale 1.0 já dá um cabo realista.
        # Bottom raw ≈ y=-0.37; lift p/ apoiar no piso (y=0.05).
        self.fishingrod = Entity(
            Mesh.from_obj(str(ASSETS / "fishingrod" / "fishingrod.obj")),
            position=(-2.0, 0.45, -1.0),
            rotation=(math.radians(-10), math.radians(30), math.radians(70)),
            scale=(1.0, 1.0, 1.0),
        )

        # Balde — apoiado no chão de madeira, perto da parede oposta à vara.
        # Raw ~0.33x0.49x0.28 (m), base em y=0.
        self.bucket = Entity(
            Mesh.from_obj(str(ASSETS / "bucket" / "bucket.obj")),
            position=(1.8, 0.05, -1.2),
            rotation=(0.0, math.radians(20), 0.0),
            scale=(1.4, 1.4, 1.4),  # ~70cm de altura
        )

        # Lanterna — deitada no chão. Raw enorme (~33x8 no eixo X), scale 0.008
        # produz comprimento ~26cm. Eixo longo no X → rotação Y orienta natural.
        self.flashlight = Entity(
            Mesh.from_obj(str(ASSETS / "flashlight" / "flashlight.obj")),
            position=(-0.5, 0.10, 2.0),  # y elevado p/ apoiar no piso (raio ~3cm)
            rotation=(0.0, math.radians(35), 0.0),
            scale=(0.01, 0.01, 0.01),
        )

        self.indoor_extras: List[Entity] = [self.bucket, self.flashlight]

        # ---------------- Listas para draw ----------------
        self.outdoor_entities: List[Entity] = [
            self.boat, self.octopus, self.seahorse, *self.trees,
        ]
        self.indoor_entities: List[Entity] = [
            self.fishingrod, *self.indoor_extras,
        ]

    # --------------------------------------------------------------- #
    # Update / draw
    # --------------------------------------------------------------- #
    def update(self, dt: float) -> None:
        """Animações sutis: barco oscila com a água."""
        t = (self._time if hasattr(self, "_time") else 0.0) + dt
        self._time = t
        # bobbing do barco
        self.boat.position[1] = WATER_Y + 0.10 + 0.05 * math.sin(t * 1.4)
        self.boat.rotation[2] = math.radians(2.0) * math.sin(t * 1.0)

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
