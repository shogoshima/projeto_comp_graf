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


class SwimmingSeahorse:
    """Cavalo-marinho que nada pelo lago com oscilação senoidal."""

    def __init__(self, entity: Entity, lake_cx: float, lake_cz: float,
                 lake_r: float, swim_speed: float = 2.0,
                 bob_amp: float = 0.6, bob_freq: float = 0.9):
        self.entity = entity
        self.cx, self.cz, self.r = lake_cx, lake_cz, lake_r
        self.speed = swim_speed
        self.bob_amp = bob_amp
        self.bob_freq = bob_freq

        self.x = float(entity.position[0])
        self.z = float(entity.position[2])
        self.yaw = float(entity.rotation[1])
        self.time = 0.0

    def update(self, dt: float) -> None:
        self.time += dt

        self.x += math.cos(self.yaw) * self.speed * dt
        self.z -= math.sin(self.yaw) * self.speed * dt

        dx = self.x - self.cx
        dz = self.z - self.cz
        dist = math.sqrt(dx * dx + dz * dz)
        margin = self.r - 3.0

        if dist > margin:
            target = math.atan2(self.z - self.cz, self.cx - self.x)
            delta = (target - self.yaw + math.pi) % math.tau - math.pi
            self.yaw += delta * min(1.0, 4.0 * dt)
        else:
            self.yaw += math.sin(self.time * 0.5) * 0.4 * dt

        s = float(self.entity.scale[0])
        y = WATER_Y - s * 3.0 + self.bob_amp * math.sin(self.time * self.bob_freq * math.tau)

        self.entity.position[0] = self.x
        self.entity.position[1] = y
        self.entity.position[2] = self.z
        self.entity.rotation[1] = self.yaw


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
            environment=1,
            base_color=(1.0, 0.95, 0.86),
            diffuse=0.82,
            specular=0.12,
            shininess=18.0,
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
            environment=0,
            base_color=(1.0, 0.93, 0.85),
            diffuse=0.78,
            specular=0.35,
            shininess=38.0,
        )

        # Polvo ao redor do lago, em pé (ligeiramente submerso). Rotação por teclado.
        OCTOPUS_SCALE = 8
        self.octopus = Entity(
            Mesh.from_obj(str(ASSETS / "octopus" / "octopus.obj")),
            position=(LAKE_CENTER[0] + 8.0, WATER_Y + 0.05, LAKE_CENTER[1] - 2.0),
            rotation=(0.0, 0.0, 0.0),
            scale=(OCTOPUS_SCALE, OCTOPUS_SCALE, OCTOPUS_SCALE),
            environment=0,
            base_color=(0.98, 0.82, 0.9),
            diffuse=0.72,
            specular=0.45,
            shininess=42.0,
        )

        # Cavalo-marinho perto da margem, escala por teclado.
        SEAHORSE_SCALE = 0.4
        self.seahorse = Entity(
            Mesh.from_obj(str(ASSETS / "seahorse" / "seahorse.obj")),
            position=(LAKE_CENTER[0] - 12.0, WATER_Y, LAKE_CENTER[1] + 4.0),
            rotation=(0.0, math.radians(45), 0.0),
            scale=(SEAHORSE_SCALE, SEAHORSE_SCALE, SEAHORSE_SCALE),
            environment=0,
            base_color=(0.9, 0.98, 1.0),
            diffuse=0.74,
            specular=0.55,
            shininess=58.0,
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
            self.outdoor_props.append(Entity(
                spruce_meshes[i % 2], position=(x, 0.0, z),
                rotation=(0.0, yaw, 0.0), scale=(scale, scale, scale),
                environment=0, base_color=(0.88, 1.0, 0.86),
                diffuse=0.86, specular=0.06, shininess=10.0,
            ))

        for i in range(200):
            pos = random_outdoor_pos()
            if pos is None:
                continue
            x, z = pos
            scale = float(rng.uniform(0.8, 1.0))
            yaw = float(rng.uniform(0, math.tau))
            self.outdoor_props.append(Entity(
                bush_set_meshes[i % 2], position=(x, 0.0, z),
                rotation=(0.0, yaw, 0.0), scale=(scale, scale, scale),
                environment=0, base_color=(0.82, 1.0, 0.78),
                diffuse=0.84, specular=0.05, shininess=8.0,
            ))

        for i in range(100):
            pos = random_outdoor_pos()
            if pos is None:
                continue
            x, z = pos
            scale = float(rng.uniform(1.0, 2.0))
            yaw = float(rng.uniform(0, math.tau))
            self.outdoor_props.append(Entity(
                stone_meshes[i % 2], position=(x, 0.0, z),
                rotation=(0.0, yaw, 0.0), scale=(scale, scale, scale),
                environment=0, base_color=(0.92, 0.92, 0.9),
                diffuse=0.7, specular=0.22, shininess=22.0,
            ))

        # ---------------- Modelos INTERNOS ----------------
        # Mesa
        TABLE_SCALE = 0.025
        self.table = Entity(
            Mesh.from_obj(str(ASSETS / "table" / "table.obj")),
            position=(HUT_CENTER[0] + 1.0, 0.0, HUT_CENTER[1] + 3.5),
            rotation=(0.0, math.radians(0), 0.0),
            scale=(TABLE_SCALE, TABLE_SCALE, TABLE_SCALE),
            environment=1,
            base_color=(1.0, 0.88, 0.76),
            diffuse=0.74,
            specular=0.34,
            shininess=40.0,
        )
        
        # Ramen ponyo
        RAMEN_SCALE = 0.003
        self.ramen = Entity(
            Mesh.from_obj(str(ASSETS / "ramen" / "ramen.obj")),
            position=(HUT_CENTER[0] + 1.0, 1.65, HUT_CENTER[1] + 3.5),
            rotation=(0.0, math.radians(0), 0.0),
            scale=(RAMEN_SCALE, RAMEN_SCALE, RAMEN_SCALE),
            environment=1,
            base_color=(1.0, 0.96, 0.9),
            diffuse=0.72,
            specular=0.48,
            shininess=56.0,
        )

        # Balde apoiado no chão de madeira, perto da parede oposta à vara.
        BUCKET_SCALE = 0.2
        self.bucket = Entity(
            Mesh.from_obj(str(ASSETS / "bucket_ponyo" / "bucket_ponyo.obj")),
            position=(HUT_CENTER[0] - 1, -0.45, HUT_CENTER[1] + 3.5),
            rotation=(0.0, math.radians(20), 0.0),
            scale=(BUCKET_SCALE, BUCKET_SCALE, BUCKET_SCALE),
            environment=1,
            base_color=(0.96, 0.96, 1.0),
            diffuse=0.68,
            specular=0.55,
            shininess=64.0,
        )

        # Lanterna deitada no chão
        FLASHLIGHT_SCALE = 0.02
        self.flashlight = Entity(
            Mesh.from_obj(str(ASSETS / "flashlight" / "flashlight.obj")),
            position=(HUT_CENTER[0] - 1, 0.5, HUT_CENTER[1] + 3.5),
            rotation=(0.0, math.radians(120), - math.radians(75)),
            scale=(FLASHLIGHT_SCALE, FLASHLIGHT_SCALE, FLASHLIGHT_SCALE),
            environment=1,
            base_color=(0.92, 0.94, 1.0),
            diffuse=0.62,
            specular=0.85,
            shininess=96.0,
        )

        # Lâmpada pendurada no centro da cabana.
        LIGHTBULB_SCALE = 0.08
        self.lightbulb = Entity(
            Mesh.from_obj(str(ASSETS / "lightbulb" / "lightbulb.obj")),
            position=(HUT_CENTER[0] + 0.56, -9, HUT_CENTER[1] - 6.2),
            rotation=(0.0, math.radians(0), 0.0),
            scale=(LIGHTBULB_SCALE, LIGHTBULB_SCALE, LIGHTBULB_SCALE),
            environment=1,
            base_color=(1.0, 0.94, 0.74),
            diffuse=0.78,
            specular=0.35,
            shininess=44.0,
        )

        self.indoor_extras: List[Entity] = [
            self.bucket, self.flashlight, self.ramen, self.lightbulb,
        ]

        # ---------------- Cavalo-marinho nadando ----------------
        self.swimming_seahorse = SwimmingSeahorse(
            self.seahorse, LAKE_CENTER[0], LAKE_CENTER[1], LAKE_RADIUS,
        )

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
                environment=0,
                base_color=(0.95, 0.98, 1.0),
                diffuse=0.5,
                specular=0.10,
                shininess=48.0,
                normal_sign=-1.0,
            )
            self.jumping_fish.append(
                JumpingFish(ent, LAKE_CENTER[0], LAKE_CENTER[1],
                            LAKE_RADIUS, rng, **params)
            )

        # ---------------- Listas para draw ----------------
        self.outdoor_entities: List[Entity] = [
            self.boat, self.octopus, self.seahorse, *self.outdoor_props,
            *[jf.entity for jf in self.jumping_fish],
        ]
        self.indoor_entities: List[Entity] = [
            self.table, *self.indoor_extras,
        ]

        # ---------------- Iluminação ----------------
        self.ambient_enabled = True
        self.sun_enabled = True
        self.boat_light_enabled = True
        self.lightbulb_enabled = True
        self.flashlight_enabled = True
        self.ambient_strength = 0.22
        self.day_ambient_strength = 0.22
        self.night_ambient_strength = 0.06
        self.diffuse_gain = 1.0
        self.specular_gain = 1.0
        self.night_mode = False

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
        # cavalo-marinho
        self.swimming_seahorse.update(dt)
        # peixes
        for jf in self.jumping_fish:
            jf.update(dt)

    @staticmethod
    def _entity_local_point(entity: Entity, point: tuple[float, float, float]) -> tuple[float, float, float]:
        p = np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
        world = entity.model_matrix() @ p
        return (float(world[0]), float(world[1]), float(world[2]))

    @staticmethod
    def _entity_local_direction(entity: Entity, direction: tuple[float, float, float]) -> tuple[float, float, float]:
        p0 = entity.model_matrix() @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        p1 = entity.model_matrix() @ np.array([direction[0], direction[1], direction[2], 1.0], dtype=np.float32)
        d = p1[:3] - p0[:3]
        d /= np.linalg.norm(d) + 1e-12
        return (float(d[0]), float(d[1]), float(d[2]))

    def boat_light_position(self) -> tuple[float, float, float]:
        """Posição aproximada da vela dentro do barco.

        Usa coordenadas locais do OBJ e a mesma matriz de modelo do barco para
        acompanhar yaw, pitch, escala, bobbing e o pivô deslocado do casco.
        """
        return self._entity_local_point(self.boat, (0.0, 0.0, 4.0))

    def lightbulb_position(self) -> tuple[float, float, float]:
        return self._entity_local_point(self.lightbulb, (-7.14, 199.6, 85.5))

    def flashlight_position(self) -> tuple[float, float, float]:
        return self._entity_local_point(self.flashlight, (-16.34, 0.0, 0.0))

    def flashlight_direction(self) -> tuple[float, float, float]:
        return self._entity_local_direction(self.flashlight, (-1.0, 0.0, 0.0))

    def skybox_brightness(self) -> float:
        return 0.1 if self.night_mode else 1.0

    def toggle_night_mode(self) -> None:
        self.night_mode = not self.night_mode
        if self.night_mode:
            self.sun_enabled = False
            self.ambient_strength = self.night_ambient_strength
        else:
            self.sun_enabled = True
            self.ambient_strength = self.day_ambient_strength

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def change_ambient(self, delta: float) -> None:
        self.ambient_strength = self._clamp(self.ambient_strength + delta, 0.0, 0.6)
        if not self.night_mode:
            self.day_ambient_strength = self.ambient_strength

    def change_diffuse(self, delta: float) -> None:
        self.diffuse_gain = self._clamp(self.diffuse_gain + delta, 0.0, 2.0)

    def change_specular(self, delta: float) -> None:
        self.specular_gain = self._clamp(self.specular_gain + delta, 0.0, 2.0)

    def apply_lighting_uniforms(self, shader, camera_position) -> None:
        cam = np.asarray(camera_position, dtype=np.float32)
        shader.set_vec3("u_view_pos", float(cam[0]), float(cam[1]), float(cam[2]))

        shader.set_int("u_ambient_enabled", 1 if self.ambient_enabled else 0)
        shader.set_float("u_ambient_strength", self.ambient_strength)
        shader.set_float("u_diffuse_gain", self.diffuse_gain)
        shader.set_float("u_specular_gain", self.specular_gain)

        shader.set_int("u_sun_enabled", 1 if self.sun_enabled else 0)
        shader.set_vec3("u_sun_pos", -120.0, 220.0, -90.0)
        shader.set_vec3("u_sun_color", 1.0, 0.94, 0.82)
        shader.set_float("u_sun_intensity", 2400.0)

        bx, by, bz = self.boat_light_position()
        shader.set_int("u_boat_light_enabled", 1 if self.boat_light_enabled else 0)
        shader.set_vec3("u_boat_light_pos", bx, by, bz)
        shader.set_vec3("u_boat_light_color", 1.0, 0.64, 0.28)
        shader.set_float("u_boat_light_intensity", 10.0)

        lx, ly, lz = self.lightbulb_position()
        shader.set_int("u_lightbulb_enabled", 1 if self.lightbulb_enabled else 0)
        shader.set_vec3("u_lightbulb_pos", lx, ly, lz)
        shader.set_vec3("u_lightbulb_color", 1.0, 0.82, 0.45)
        shader.set_float("u_lightbulb_intensity", 6.0)

        fx, fy, fz = self.flashlight_position()
        fdx, fdy, fdz = self.flashlight_direction()
        shader.set_int("u_flashlight_enabled", 1 if self.flashlight_enabled else 0)
        shader.set_vec3("u_flashlight_pos", fx, fy, fz)
        shader.set_vec3("u_flashlight_dir", fdx, fdy, fdz)
        shader.set_vec3("u_flashlight_color", 0.55, 0.76, 1.0)
        shader.set_float("u_flashlight_intensity", 12.0)
        shader.set_float("u_flashlight_inner_cutoff", math.cos(math.radians(22.0)))
        shader.set_float("u_flashlight_outer_cutoff", math.cos(math.radians(42.0)))

    def draw(self, shader, wireframe: bool = False) -> None:
        # pisos
        self.outdoor_floor.draw(shader)
        self.lake.draw(shader)

        # cabana (delimitador, com cull off)
        self.hut.draw(shader)

        # demais
        for e in self.outdoor_entities:
            e.draw(shader)
        for e in self.indoor_entities:
            e.draw(shader)
