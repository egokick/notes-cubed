import argparse
import ctypes
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pyglet
from pyglet import gl, shapes, sprite
from pyglet.image.buffer import Framebuffer
from pyglet.math import Vec3, Vec4
from pyglet.media import synthesis
from pyglet.window import key, mouse


DATA_DIR = Path(__file__).parent / "data"
CONFIG_PATH = DATA_DIR / "config.json"
FACE_NAMES = ["front", "right", "back", "left", "top", "bottom"]
DEFAULT_BACKGROUNDS = {
    "front": "#1e1e2e",
    "right": "#10314a",
    "back": "#2d1b3f",
    "left": "#3c2f2f",
    "top": "#0f3d3e",
    "bottom": "#3a2a1a",
}
DEFAULT_CONFIG = {
    "last_face": 0,
    "remote": "",
    "backgrounds": {name: {"type": "color", "value": DEFAULT_BACKGROUNDS[name]} for name in FACE_NAMES},
    "font_colors": {name: [255, 255, 255] for name in FACE_NAMES},
}
EDITOR_MARGIN = 80
EDITOR_SCALE = 0.72  # fraction of window size used by editor overlay
AUTOSAVE_SECONDS = 30
ROTATE_SENSITIVITY = 2.0
FACE_PREVIEW_SIZE = 512
FACE_PREVIEW_PADDING = 12
FACE_KEY_WIDTH = 120
FACE_KEY_HEIGHT = 22
FACE_KEY_SPACING = 6
FACE_KEY_MARGIN = 18
AUTO_SPIN_SPEED = 0.35
AUTO_SPIN_TILT_DEGREES = 12.0
SCROLLBAR_WIDTH = 4
SCROLLBAR_MARGIN = 10
SCROLLBAR_MIN_HEIGHT = 24
SETTINGS_PANEL_WIDTH = 320
SETTINGS_PANEL_HEIGHT = 270
SETTINGS_PANEL_PADDING = 12
SETTINGS_SWATCH_SIZE = 14
SETTINGS_SWATCH_GAP = 6
SETTINGS_BOX_WIDTH = 110
SETTINGS_BOX_HEIGHT = 18
SETTINGS_BUTTON_WIDTH = 70
SETTINGS_BUTTON_HEIGHT = 18
FONT_COLOR_PRESETS = [
    (255, 255, 255),
    (240, 220, 170),
    (180, 220, 255),
    (255, 160, 160),
    (180, 255, 200),
    (210, 210, 210),
]
FACE_COLOR_PRESETS = [
    (30, 30, 46),
    (16, 49, 74),
    (45, 27, 63),
    (60, 47, 47),
    (15, 61, 62),
    (58, 42, 26),
]


def parse_color(value, fallback=(40, 40, 50)):
    if isinstance(value, str) and value.startswith("#") and len(value) in (4, 7):
        hex_value = value[1:]
        if len(hex_value) == 3:
            hex_value = "".join([c * 2 for c in hex_value])
        try:
            r = int(hex_value[0:2], 16)
            g = int(hex_value[2:4], 16)
            b = int(hex_value[4:6], 16)
            return r, g, b
        except ValueError:
            return fallback
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return int(value[0]), int(value[1]), int(value[2])
    return fallback


def ensure_data_folder():
    DATA_DIR.mkdir(exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
    for name in FACE_NAMES:
        path = DATA_DIR / f"{name}.txt"
        if not path.exists():
            path.write_text("", encoding="utf-8")


def load_config():
    ensure_data_folder()
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        data = DEFAULT_CONFIG
    # Fill defaults for missing backgrounds
    if "backgrounds" not in data:
        data["backgrounds"] = DEFAULT_CONFIG["backgrounds"]
    for name in FACE_NAMES:
        if name not in data["backgrounds"]:
            data["backgrounds"][name] = {"type": "color", "value": DEFAULT_BACKGROUNDS[name]}
        elif data["backgrounds"][name].get("type") == "image":
            data["backgrounds"][name].setdefault("mode", "scale")
    if "remote" not in data:
        data["remote"] = ""
    if "last_face" not in data:
        data["last_face"] = 0
    if "font_colors" not in data:
        data["font_colors"] = DEFAULT_CONFIG["font_colors"]
    for name in FACE_NAMES:
        if name not in data["font_colors"]:
            data["font_colors"][name] = DEFAULT_CONFIG["font_colors"][name]
    return data


def save_config(config):
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def git_sync(commit_message, remote_url):
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Git not available; skipping sync.")
        return

    try:
        subprocess.run(["git", "init"], check=True, cwd=DATA_DIR, capture_output=True)
    except subprocess.CalledProcessError as exc:
        print(f"git init failed: {exc}")
        return

    status = subprocess.run(["git", "status", "--porcelain"], cwd=DATA_DIR, capture_output=True, text=True)
    if status.stdout.strip() == "":
        return  # nothing to commit

    try:
        subprocess.run(["git", "add", "."], check=True, cwd=DATA_DIR)
        subprocess.run(
            ["git", "commit", "-m", commit_message],
            check=True,
            cwd=DATA_DIR,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"git commit failed: {exc}")
        return

    if remote_url:
        try:
            remotes = subprocess.run(
                ["git", "remote"], cwd=DATA_DIR, capture_output=True, text=True, check=True
            ).stdout.splitlines()
            if "origin" not in remotes:
                subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=DATA_DIR, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"git remote setup failed: {exc}")
            return
        try:
            subprocess.run(["git", "push", "-u", "origin", "main"], cwd=DATA_DIR, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"git push failed: {exc}")


def rotation_matrix_apply(vec, rotation):
    x, y, z = vec
    rotated = rotation @ Vec4(x, y, z, 0.0)
    return rotated.x, rotated.y, rotated.z


def rotation_from_yaw_pitch(yaw_deg, pitch_deg):
    rot_y = pyglet.math.Mat4.from_rotation(math.radians(yaw_deg), Vec3(0, 1, 0))
    rot_x = pyglet.math.Mat4.from_rotation(math.radians(pitch_deg), Vec3(1, 0, 0))
    return rot_y @ rot_x


def orthonormalize_rotation(rotation):
    original_forward = Vec3(*rotation_matrix_apply((0, 0, 1), rotation))
    right = Vec3(*rotation_matrix_apply((1, 0, 0), rotation))
    if right.length() == 0:
        return rotation
    right = right.normalize()
    up = Vec3(*rotation_matrix_apply((0, 1, 0), rotation))
    if up.length() == 0:
        return rotation
    up = (up - right * right.dot(up)).normalize()
    forward = right.cross(up).normalize()
    if original_forward.length() and forward.dot(original_forward) < 0:
        up = up * -1.0
        forward = forward * -1.0
    return pyglet.math.Mat4(
        right.x,
        up.x,
        forward.x,
        0.0,
        right.y,
        up.y,
        forward.y,
        0.0,
        right.z,
        up.z,
        forward.z,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )


class CubeRenderer:
    NORMALS = {
        "front": (0, 0, 1),
        "back": (0, 0, -1),
        "left": (-1, 0, 0),
        "right": (1, 0, 0),
        "top": (0, 1, 0),
        "bottom": (0, -1, 0),
    }
    ORIENTATIONS = {
        "front": (0.0, 0.0),
        "back": (180.0, 0.0),
        "left": (90.0, 0.0),
        "right": (-90.0, 0.0),
        "top": (0.0, 90.0),
        "bottom": (0.0, -90.0),
    }

    def __init__(self, size=1.2, face_alpha=220):
        self.size = float(size)
        self.face_alpha = int(face_alpha)
        self._build_geometry()

    def _build_geometry(self):
        size = self.size
        faces = [
            ("front", [(size, -size, size), (-size, -size, size), (-size, size, size), (size, size, size)]),
            ("back", [(-size, -size, -size), (size, -size, -size), (size, size, -size), (-size, size, -size)]),
            ("left", [(-size, -size, size), (-size, -size, -size), (-size, size, -size), (-size, size, size)]),
            ("right", [(size, -size, -size), (size, -size, size), (size, size, size), (size, size, -size)]),
            ("top", [(-size, size, size), (-size, size, -size), (size, size, -size), (size, size, size)]),
            ("bottom", [(-size, -size, -size), (-size, -size, size), (size, -size, size), (size, -size, -size)]),
        ]
        self.face_verts = {name: verts for name, verts in faces}
        self.face_centers = {
            name: (
                sum(v[0] for v in verts) / 4.0,
                sum(v[1] for v in verts) / 4.0,
                sum(v[2] for v in verts) / 4.0,
            )
            for name, verts in faces
        }
        self.face_order = [name for name, _ in faces]
        self.face_tex_coords = self._build_face_tex_coords()
        positions = []
        indices = []
        edges = []
        for _face_name, verts in faces:
            start_index = len(positions) // 3
            for vx, vy, vz in verts:
                positions.extend([vx, vy, vz])
            indices.extend([start_index, start_index + 1, start_index + 2, start_index, start_index + 2, start_index + 3])
            edges.extend(
                [
                    (start_index, start_index + 1),
                    (start_index + 1, start_index + 2),
                    (start_index + 2, start_index + 3),
                    (start_index + 3, start_index),
                ]
            )
        self.positions = positions
        self.indices = indices
        self.edges = edges

    def _build_face_tex_coords(self):
        face_tex_coords = {}
        for face_name in FACE_NAMES:
            verts = self.face_verts.get(face_name)
            if not verts:
                continue
            yaw, pitch = self.ORIENTATIONS.get(face_name, (0.0, 0.0))
            rot_y = pyglet.math.Mat4.from_rotation(math.radians(yaw), Vec3(0, 1, 0))
            rot_x = pyglet.math.Mat4.from_rotation(math.radians(pitch), Vec3(1, 0, 0))
            rot = rot_y @ rot_x
            rotated = [rot @ Vec4(x, y, z, 1.0) for x, y, z in verts]
            xs = [v.x for v in rotated]
            ys = [v.y for v in rotated]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span_x = max_x - min_x
            span_y = max_y - min_y
            if span_x == 0 or span_y == 0:
                face_tex_coords[face_name] = [0.0, 0.0, 0.0] * 4
                continue
            tex_coords = []
            for v in rotated:
                u = (v.x - min_x) / span_x
                t = (v.y - min_y) / span_y
                tex_coords.extend([float(u), float(t), 0.0])
            face_tex_coords[face_name] = tex_coords
        return face_tex_coords

    def setup_3d(self, window, rotation):
        gl.glEnable(gl.GL_DEPTH_TEST)
        aspect = window.width / float(window.height)
        # pyglet.math.Mat4.perspective_projection signature: (aspect, z_near, z_far, fov_degrees=60)
        window.projection = pyglet.math.Mat4.perspective_projection(aspect, 0.1, 100.0, 60.0)
        window.view = pyglet.math.Mat4.look_at(Vec3(0, 0, 4.0), Vec3(0, 0, 0), Vec3(0, 1, 0)) @ rotation

    def draw(self, backgrounds, rotation):
        raise NotImplementedError("Use draw_textured so text stays visible on faces.")

    def draw_textured(self, face_textures, backgrounds, rotation):
        self.draw_textured_with_alpha(face_textures, backgrounds, rotation, self.face_alpha)

    def draw_textured_with_alpha(self, face_textures, backgrounds, rotation, face_alpha, sort_faces=False):
        # Default pyglet shader samples a texture and adds `colors`, so draw per-face with that face's texture bound.
        indices = [0, 1, 2, 0, 2, 3]
        face_names = self.face_order
        if sort_faces:
            face_names = sorted(
                self.face_order,
                key=lambda name: rotation_matrix_apply(self.face_centers[name], rotation)[2],
            )
        for face_name in face_names:
            if isinstance(face_alpha, dict):
                alpha = int(face_alpha.get(face_name, self.face_alpha))
            else:
                alpha = int(face_alpha)
            tex_coords = self.face_tex_coords.get(face_name) or [0.0, 0.0, 0.0] * 4
            bg_def = (backgrounds or {}).get(face_name) or {"type": "color", "value": DEFAULT_BACKGROUNDS[face_name]}
            if bg_def.get("type") == "color":
                base_color = parse_color(bg_def.get("value"))
            else:
                base_color = parse_color(DEFAULT_BACKGROUNDS[face_name])
            rotated_normal = rotation_matrix_apply(self.NORMALS[face_name], rotation)
            light = max(0.0, rotated_normal[2])
            shade = 0.35 + 0.6 * light
            shaded = (
                int(min(255, base_color[0] * shade)),
                int(min(255, base_color[1] * shade)),
                int(min(255, base_color[2] * shade)),
                alpha,
            )
            colors = list(shaded) * 4
            texture = (face_textures or {}).get(face_name)
            if texture:
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glBindTexture(texture.target, texture.id)
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(770, 771)  # SRC_ALPHA, ONE_MINUS_SRC_ALPHA
            else:
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            verts = self.face_verts[face_name]
            positions = [c for v in verts for c in v]
            pyglet.graphics.draw_indexed(
                4,
                gl.GL_TRIANGLES,
                indices,
                position=("f", positions),
                colors=("Bn", colors),
                tex_coords=("f", tex_coords),
            )

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # Edges (wireframe) for depth cues.
        edge_indices = [i for edge in self.edges for i in edge]
        pyglet.graphics.draw_indexed(
            len(self.positions) // 3,
            gl.GL_LINES,
            edge_indices,
            position=("f", self.positions),
            colors=("Bn", [20, 20, 20, 255] * (len(self.positions) // 3)),
        )


class FaceState:
    def __init__(self, name, path, background_def, font_color=None):
        self.name = name
        self.path = path
        self.background_def = background_def or {"type": "color", "value": DEFAULT_BACKGROUNDS[name]}
        self.font_color = tuple(int(c) for c in (font_color or (255, 255, 255))[:3])
        self.document = pyglet.text.document.UnformattedDocument(self._load_text())
        base_style = {"font_name": "Consolas", "font_size": 14, "color": (*self.font_color, 255)}
        self.document.set_style(0, len(self.document.text), base_style)
        self.layout = None
        self.caret = None
        self._bg_image = None
        self._bg_player = None
        self._bg_sprite = None
        self.preview_dirty = True
        self._preview_last_text = None
        self._preview_texture = None
        self._preview_fbo = None
        self._preview_batch = None
        self._preview_doc = None
        self._preview_layout = None
        self._preview_bg = None
        self._preview_size = None
        self.undo_stack = []
        self.redo_stack = []
        self._history_suspended = False
        self._load_background_asset()

    def _load_text(self):
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def save(self):
        self.path.write_text(self.document.text, encoding="utf-8")

    def bind_layout(self, width, height, window=None):
        self.layout = pyglet.text.layout.IncrementalTextLayout(
            self.document, width=width, height=height, multiline=True, wrap_lines=True
        )
        self.layout.x = EDITOR_MARGIN
        self.layout.y = EDITOR_MARGIN
        self.caret = pyglet.text.caret.Caret(self.layout, color=(*self.font_color, 255), window=window)

    def resize_layout(self, width, height):
        if not self.layout:
            return
        self.layout.width = width
        self.layout.height = height
        self.layout.wrap_lines = True
        self.layout.x = EDITOR_MARGIN
        self.layout.y = EDITOR_MARGIN

    def mark_preview_dirty(self):
        self.preview_dirty = True

    def preview_texture(self):
        return self._preview_texture

    def _ensure_preview_resources(self, size):
        if size <= 0:
            return
        if self._preview_texture is not None and self._preview_size == size:
            return
        self._preview_size = size
        self._preview_texture = pyglet.image.Texture.create(size, size)
        self._preview_fbo = Framebuffer()
        self._preview_fbo.attach_texture(self._preview_texture)
        self._preview_batch = pyglet.graphics.Batch()
        self._preview_bg = shapes.Rectangle(
            0,
            0,
            size,
            size,
            color=(0, 0, 0),
            batch=self._preview_batch,
        )
        self._preview_bg.opacity = 0
        self._preview_doc = pyglet.text.document.UnformattedDocument("")
        self._preview_doc.set_style(
            0,
            0,
            {"font_name": "Consolas", "font_size": 14, "color": (*self.font_color, 255)},
        )
        self._preview_layout = pyglet.text.layout.TextLayout(
            self._preview_doc,
            x=FACE_PREVIEW_PADDING,
            y=FACE_PREVIEW_PADDING,
            width=size - FACE_PREVIEW_PADDING * 2,
            height=size - FACE_PREVIEW_PADDING * 2,
            multiline=True,
            wrap_lines=True,
            batch=self._preview_batch,
        )

    def render_preview_to_texture(self, window, size):
        self._ensure_preview_resources(size)
        if self._preview_size is None:
            return
        preview_text = self.document.text
        if preview_text != self._preview_last_text:
            self._preview_last_text = preview_text
            # Keep previews snappy by truncating.
            max_chars = 2000
            if len(preview_text) > max_chars:
                preview_text = preview_text[:max_chars] + "\n…"
            self._preview_doc.text = preview_text
            self._preview_doc.set_style(
                0,
                len(self._preview_doc.text),
                {"font_name": "Consolas", "font_size": 14, "color": (*self.font_color, 255)},
            )

        prev_view = window.view
        prev_proj = window.projection
        prev_viewport = window.viewport

        try:
            self._preview_fbo.bind()
            gl.glViewport(0, 0, self._preview_size, self._preview_size)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(770, 771)  # SRC_ALPHA, ONE_MINUS_SRC_ALPHA
            gl.glClearColor(0.0, 0.0, 0.0, 0.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            window.projection = pyglet.math.Mat4.orthogonal_projection(
                0, self._preview_size, 0, self._preview_size, -1, 1
            )
            window.view = pyglet.math.Mat4()
            self._preview_batch.draw()
        finally:
            self._preview_fbo.unbind()
            window.viewport = prev_viewport
            window.projection = prev_proj
            window.view = prev_view

        self.preview_dirty = False

    def scroll(self, dy):
        if not self.layout:
            return
        new_view = self.layout.view_y + dy
        new_view = max(0, min(max(0, self.layout.content_height - self.layout.height), new_view))
        self.layout.view_y = new_view

    def background_color(self):
        if self.background_def.get("type") == "color":
            return parse_color(self.background_def.get("value"))
        return parse_color(DEFAULT_BACKGROUNDS[self.name])

    def background_image(self):
        if self._bg_player:
            try:
                return self._bg_player.get_texture()
            except Exception:
                return None
        if self._bg_sprite:
            return self._bg_sprite
        return self._bg_image

    def snapshot(self):
        position = self.caret.position if self.caret else len(self.document.text)
        mark = self.caret.mark if self.caret else None
        return {"text": self.document.text, "position": position, "mark": mark}

    def record_undo(self):
        if self._history_suspended:
            return
        snapshot = self.snapshot()
        if self.undo_stack and self.undo_stack[-1] == snapshot:
            return
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _apply_state(self, state):
        self._history_suspended = True
        text = state["text"]
        self.document.text = text
        self.document.set_style(
            0,
            len(self.document.text),
            {"font_name": "Consolas", "font_size": 14, "color": (*self.font_color, 255)},
        )
        if self.caret:
            self.caret.position = min(state["position"], len(text))
            mark = state["mark"]
            if mark is not None:
                mark = min(mark, len(text))
            self.caret.mark = mark
        self._history_suspended = False
        self.preview_dirty = True

    def undo(self):
        if not self.undo_stack:
            return
        current = self.snapshot()
        state = self.undo_stack.pop()
        self.redo_stack.append(current)
        if len(self.redo_stack) > 100:
            self.redo_stack.pop(0)
        self._apply_state(state)

    def redo(self):
        if not self.redo_stack:
            return
        current = self.snapshot()
        state = self.redo_stack.pop()
        self.undo_stack.append(current)
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)
        self._apply_state(state)

    def background_image_mode(self):
        if self.background_def.get("type") == "image":
            return self.background_def.get("mode", "scale")
        return "scale"

    def set_background_def(self, background_def):
        self.background_def = background_def or {"type": "color", "value": DEFAULT_BACKGROUNDS[self.name]}
        if self.background_def.get("type") == "image":
            self.background_def.setdefault("mode", "scale")
        self._load_background_asset()

    def set_font_color(self, color):
        rgb = tuple(int(c) for c in color[:3])
        self.font_color = rgb
        self.document.set_style(0, len(self.document.text), {"color": (*self.font_color, 255)})
        if self.caret:
            self.caret.color = (*self.font_color, 255)
        self.preview_dirty = True

    def title_text(self):
        text = self.document.text
        if text:
            line = text.splitlines()[0].strip()
            if line:
                return line
        return self.name.capitalize()

    def key_title(self, max_chars=14):
        title = self.title_text()
        if len(title) > max_chars:
            return title[: max(1, max_chars - 3)] + "..."
        return title

    def _load_background_asset(self):
        if self.background_def.get("type") != "image":
            self._bg_image = None
            if self._bg_player:
                try:
                    self._bg_player.pause()
                except Exception:
                    pass
            self._bg_player = None
            self._bg_sprite = None
            return
        path = Path(self.background_def.get("value", ""))
        if not path.exists():
            self._bg_image = None
            if self._bg_player:
                try:
                    self._bg_player.pause()
                except Exception:
                    pass
            self._bg_player = None
            self._bg_sprite = None
            return
        if self._bg_player:
            try:
                self._bg_player.pause()
            except Exception:
                pass
            self._bg_player = None
        self._bg_sprite = None
        try:
            ext = path.suffix.lower()
            if ext in {".gif", ".mp4", ".mov", ".m4v", ".webm", ".avi"}:
                try:
                    source = pyglet.media.load(path.as_posix())
                    player = pyglet.media.Player()
                    player.queue(source)
                    player.volume = 0.0
                    if hasattr(player, "loop"):
                        player.loop = True
                    player.play()
                    self._bg_player = player
                    self._bg_image = None
                    return
                except Exception:
                    self._bg_player = None
            loaded = pyglet.image.load(path.as_posix())
            if isinstance(loaded, pyglet.image.Animation):
                self._bg_sprite = pyglet.sprite.Sprite(loaded)
                self._bg_image = None
            else:
                self._bg_image = loaded
        except Exception:
            self._bg_image = None
            self._bg_sprite = None


class NotesCubedApp(pyglet.window.Window):
    def __init__(self, config):
        screen = self._get_default_screen()
        config_gl = pyglet.gl.Config(alpha_size=8, depth_size=24, double_buffer=True, sample_buffers=1, samples=4)
        super().__init__(
            width=screen.width,
            height=screen.height,
            caption="Notes Cubed",
            resizable=False,
            style=pyglet.window.Window.WINDOW_STYLE_TRANSPARENT,
            config=config_gl,
        )
        self.set_location(0, 0)
        self._make_topmost()

        self.config_data = config
        self.faces = []
        for name in FACE_NAMES:
            face = FaceState(
                name,
                DATA_DIR / f"{name}.txt",
                config["backgrounds"].get(name),
                config["font_colors"].get(name),
            )
            face.bind_layout(self.width - 2 * EDITOR_MARGIN, self.height - 2 * EDITOR_MARGIN, window=self)
            self.faces.append(face)
        self.current_face_index = max(0, min(len(self.faces) - 1, config.get("last_face", 0)))

        self.rotation = self._rotation_for_face(self.faces[self.current_face_index].name)

        self.dragging = False
        self.last_mouse = (0, 0)
        self._drag_vector = None
        self._drag_started_outside_cube = False
        self.edit_mode = True
        self._return_to_edit_when_aligned = False
        self.last_save = time.time()

        self.ui_batch = pyglet.graphics.Batch()
        self.settings_batch = pyglet.graphics.Batch()
        self.editor_rect = None
        self.face_label = None
        self.mode_label = None
        self.toast_label = None
        self.face_key_items = []
        self.spin_key_item = None
        self.auto_spin = False
        self.auto_spin_speed = AUTO_SPIN_SPEED
        self.cog_label = None
        self.settings_panel = None
        self.settings_title = None
        self.settings_labels = {}
        self.font_color_swatches = []
        self.face_color_swatches = []
        self.font_rgb_box = None
        self.font_rgb_label = None
        self.face_rgb_box = None
        self.face_rgb_label = None
        self.image_button = None
        self.image_label = None
        self.image_info_label = None
        self.mode_buttons = []
        self.settings_open = False
        self._settings_input_target = None
        self._settings_input_text = ""
        self._settings_panel_bounds = None
        self._settings_font_rgb_bounds = None
        self._settings_face_rgb_bounds = None
        self._settings_image_bounds = None
        self._pending_edit_click = None
        self._scrollbar_track = None
        self._scrollbar_thumb = None
        self._toast_until = 0.0
        self._pending_screenshot_path = None
        self._mvp_3d = None
        self._cube_alpha_normal = 220
        self._cube_alpha_drag_front = 110
        self._cube_alpha_drag_other = 150
        self._build_editor_overlay()
        self._build_labels()
        self._build_face_keys()
        self.cube = CubeRenderer()

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(True)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)

    def _get_default_screen(self):
        # Pyglet 2.x on Windows exposes Display via pyglet.display, not pyglet.canvas.
        try:
            from pyglet import display

            return display.Display().get_default_screen()
        except Exception:
            try:
                platform = pyglet.window.get_platform()
                display_obj = platform.get_default_display()
                return display_obj.get_default_screen()
            except Exception:
                # Last resort: raise a clear error
                raise RuntimeError("Unable to obtain default screen from pyglet")

    def _build_editor_overlay(self):
        active_face = self.faces[self.current_face_index]
        width = int(self.width * EDITOR_SCALE)
        height = int(self.height * EDITOR_SCALE)
        for face in self.faces:
            face.resize_layout(width, height)
        color = active_face.background_color()
        if self.editor_rect is None:
            self.editor_rect = shapes.Rectangle(
                int((self.width - width) / 2),
                int((self.height - height) / 2),
                width,
                height,
                color=(color[0], color[1], color[2]),
            )
        else:
            self.editor_rect.x = int((self.width - width) / 2)
            self.editor_rect.y = int((self.height - height) / 2)
            self.editor_rect.width = width
            self.editor_rect.height = height
            self.editor_rect.color = color

    def _build_labels(self):
        active_face = self.faces[self.current_face_index]
        if self.face_label is None:
            self.face_label = pyglet.text.Label(
                f"Face: {active_face.name}",
                font_size=12,
                x=20,
                y=self.height - 30,
                color=(255, 255, 255, 200),
                batch=self.ui_batch,
            )
        else:
            self.face_label.text = f"Face: {active_face.name}"
            self.face_label.y = self.height - 30
        if self.mode_label is None:
            self.mode_label = pyglet.text.Label(
                self._mode_text(),
                font_size=11,
                x=20,
                y=self.height - 50,
                color=(200, 200, 200, 180),
                batch=self.ui_batch,
            )
        else:
            self.mode_label.text = self._mode_text()
            self.mode_label.y = self.height - 50

    def _face_key_label_text(self, face_name):
        for face in self.faces:
            if face.name == face_name:
                return face.key_title()
        return face_name.capitalize()

    def _build_face_keys(self):
        if self.spin_key_item is None:
            rect = shapes.Rectangle(
                0,
                0,
                FACE_KEY_WIDTH,
                FACE_KEY_HEIGHT,
                color=(40, 40, 50),
                batch=self.ui_batch,
            )
            label = pyglet.text.Label(
                "⟳",
                font_size=11,
                x=0,
                y=0,
                anchor_x="center",
                anchor_y="center",
                color=(230, 230, 230, 230),
                batch=self.ui_batch,
            )
            self.spin_key_item = {"rect": rect, "label": label}
        if not self.face_key_items:
            for face_name in FACE_NAMES:
                rect = shapes.Rectangle(
                    0,
                    0,
                    FACE_KEY_WIDTH,
                    FACE_KEY_HEIGHT,
                    color=(40, 40, 50),
                    batch=self.ui_batch,
                )
                label = pyglet.text.Label(
                    self._face_key_label_text(face_name),
                    font_size=10,
                    x=0,
                    y=0,
                    anchor_y="center",
                    color=(230, 230, 230, 230),
                    batch=self.ui_batch,
                )
                self.face_key_items.append({"name": face_name, "rect": rect, "label": label})
        self._update_face_keys()

    def _update_labels(self):
        active_face = self.faces[self.current_face_index]
        self.face_label.text = f"Face: {active_face.name}"
        self.face_label.y = self.height - 30
        self.mode_label.text = self._mode_text()
        self.mode_label.y = self.height - 50
        self._update_face_keys()

    def _update_face_keys(self):
        right = self.width - FACE_KEY_MARGIN
        top = self.height - FACE_KEY_MARGIN
        if self._mvp_3d is not None:
            bbox = self._cube_bbox_on_screen(self._mvp_3d)
            if bbox:
                _x0, _y0, x1, y1 = bbox
                desired_left = x1 + FACE_KEY_MARGIN
                right = min(self.width - FACE_KEY_MARGIN, desired_left + FACE_KEY_WIDTH)
                top = min(self.height - FACE_KEY_MARGIN, y1 + FACE_KEY_MARGIN)
        current_name = self.faces[self.current_face_index].name
        face_lookup = {face.name: face for face in self.faces}
        if self.spin_key_item:
            rect = self.spin_key_item["rect"]
            label = self.spin_key_item["label"]
            rect.x = right - FACE_KEY_WIDTH
            rect.y = top - FACE_KEY_HEIGHT
            label.x = rect.x + rect.width / 2
            label.y = rect.y + rect.height / 2
            if self.auto_spin:
                rect.color = (90, 120, 160)
                label.color = (255, 255, 255, 240)
            else:
                rect.color = (40, 40, 50)
                label.color = (230, 230, 230, 220)
        for index, item in enumerate(self.face_key_items):
            rect = item["rect"]
            label = item["label"]
            rect.x = right - FACE_KEY_WIDTH
            rect.y = top - (FACE_KEY_HEIGHT + FACE_KEY_SPACING) * (index + 1) - FACE_KEY_HEIGHT
            label.x = rect.x + 8
            label.y = rect.y + FACE_KEY_HEIGHT / 2
            face_state = face_lookup.get(item["name"])
            if face_state:
                label.text = face_state.key_title()
            if item["name"] == current_name:
                rect.color = (80, 120, 190)
                label.color = (255, 255, 255, 240)
            else:
                rect.color = (40, 40, 50)
                label.color = (230, 230, 230, 220)

    def _mode_text(self):
        if self.edit_mode:
            return "Edit mode (click outside editor to rotate, F12 screenshot)"
        return "Rotate mode (release to snap, F12 screenshot)"

    def _apply_pending_caret(self):
        if not self._pending_edit_click:
            return
        x, y = self._pending_edit_click
        if self._point_in_editor(x, y):
            self._activate_caret(x, y, mouse.LEFT, 0)
        self._pending_edit_click = None

    def _ensure_scrollbar(self):
        if self._scrollbar_track is None:
            self._scrollbar_track = shapes.Rectangle(0, 0, 1, 1, color=(20, 20, 24))
            self._scrollbar_track.opacity = 140
        if self._scrollbar_thumb is None:
            self._scrollbar_thumb = shapes.Rectangle(0, 0, 1, 1, color=(220, 220, 220))
            self._scrollbar_thumb.opacity = 220

    def _update_scrollbar(self, face, pad):
        self._ensure_scrollbar()
        if not face.layout or face.layout.content_height <= face.layout.height + 1:
            self._scrollbar_track.opacity = 0
            self._scrollbar_thumb.opacity = 0
            return
        track_x = self.editor_rect.x + self.editor_rect.width - pad - SCROLLBAR_MARGIN
        track_y = self.editor_rect.y + pad
        track_height = max(1, self.editor_rect.height - pad * 2)
        view_ratio = min(1.0, face.layout.height / float(face.layout.content_height))
        thumb_height = max(SCROLLBAR_MIN_HEIGHT, int(track_height * view_ratio))
        max_scroll = max(1, face.layout.content_height - face.layout.height)
        scroll_ratio = min(1.0, max(0.0, face.layout.view_y / float(max_scroll)))
        thumb_y = track_y + (track_height - thumb_height) * scroll_ratio
        self._scrollbar_track.x = track_x
        self._scrollbar_track.y = track_y
        self._scrollbar_track.width = SCROLLBAR_WIDTH
        self._scrollbar_track.height = track_height
        self._scrollbar_track.opacity = 120
        self._scrollbar_thumb.x = track_x
        self._scrollbar_thumb.y = thumb_y
        self._scrollbar_thumb.width = SCROLLBAR_WIDTH
        self._scrollbar_thumb.height = thumb_height
        self._scrollbar_thumb.opacity = 220

    def _draw_scrollbar(self):
        if self._scrollbar_track and self._scrollbar_track.opacity > 0:
            self._scrollbar_track.draw()
        if self._scrollbar_thumb and self._scrollbar_thumb.opacity > 0:
            self._scrollbar_thumb.draw()

    def _ensure_settings_ui(self):
        if self.cog_label is None:
            self.cog_label = pyglet.text.Label(
                "⚙",
                font_size=12,
                x=0,
                y=0,
                anchor_x="center",
                anchor_y="center",
                color=(240, 240, 240, 220),
                batch=self.settings_batch,
            )
        if self.settings_panel is None:
            self.settings_panel = shapes.Rectangle(
                0,
                0,
                SETTINGS_PANEL_WIDTH,
                SETTINGS_PANEL_HEIGHT,
                color=(18, 18, 24),
                batch=self.settings_batch,
            )
            self.settings_panel.opacity = 235
            self.settings_title = pyglet.text.Label(
                "Face settings",
                font_size=11,
                x=0,
                y=0,
                color=(240, 240, 240, 230),
                batch=self.settings_batch,
            )
            self.settings_labels["font"] = pyglet.text.Label(
                "Font color",
                font_size=9,
                x=0,
                y=0,
                color=(210, 210, 210, 220),
                batch=self.settings_batch,
            )
            self.settings_labels["face"] = pyglet.text.Label(
                "Face color",
                font_size=9,
                x=0,
                y=0,
                color=(210, 210, 210, 220),
                batch=self.settings_batch,
            )
            self.settings_labels["font_rgb"] = pyglet.text.Label(
                "Font RGB",
                font_size=8,
                x=0,
                y=0,
                color=(180, 180, 180, 220),
                batch=self.settings_batch,
            )
            self.settings_labels["face_rgb"] = pyglet.text.Label(
                "Face RGB",
                font_size=8,
                x=0,
                y=0,
                color=(180, 180, 180, 220),
                batch=self.settings_batch,
            )
            self.settings_labels["image"] = pyglet.text.Label(
                "Image",
                font_size=9,
                x=0,
                y=0,
                color=(210, 210, 210, 220),
                batch=self.settings_batch,
            )
            self.settings_labels["mode"] = pyglet.text.Label(
                "Image mode",
                font_size=8,
                x=0,
                y=0,
                color=(180, 180, 180, 220),
                batch=self.settings_batch,
            )
            for color in FONT_COLOR_PRESETS:
                rect = shapes.Rectangle(0, 0, SETTINGS_SWATCH_SIZE, SETTINGS_SWATCH_SIZE, color=color, batch=self.settings_batch)
                rect.opacity = 230
                self.font_color_swatches.append({"color": color, "rect": rect})
            for color in FACE_COLOR_PRESETS:
                rect = shapes.Rectangle(0, 0, SETTINGS_SWATCH_SIZE, SETTINGS_SWATCH_SIZE, color=color, batch=self.settings_batch)
                rect.opacity = 230
                self.face_color_swatches.append({"color": color, "rect": rect})
            self.font_rgb_box = shapes.Rectangle(0, 0, SETTINGS_BOX_WIDTH, SETTINGS_BOX_HEIGHT, color=(35, 35, 45), batch=self.settings_batch)
            self.font_rgb_box.opacity = 220
            self.font_rgb_label = pyglet.text.Label(
                "",
                font_size=8,
                x=0,
                y=0,
                color=(230, 230, 230, 230),
                batch=self.settings_batch,
            )
            self.face_rgb_box = shapes.Rectangle(0, 0, SETTINGS_BOX_WIDTH, SETTINGS_BOX_HEIGHT, color=(35, 35, 45), batch=self.settings_batch)
            self.face_rgb_box.opacity = 220
            self.face_rgb_label = pyglet.text.Label(
                "",
                font_size=8,
                x=0,
                y=0,
                color=(230, 230, 230, 230),
                batch=self.settings_batch,
            )
            self.image_button = shapes.Rectangle(0, 0, SETTINGS_BOX_WIDTH, SETTINGS_BOX_HEIGHT, color=(35, 35, 45), batch=self.settings_batch)
            self.image_button.opacity = 220
            self.image_label = pyglet.text.Label(
                "",
                font_size=8,
                x=0,
                y=0,
                color=(230, 230, 230, 230),
                batch=self.settings_batch,
            )
            self.image_info_label = pyglet.text.Label(
                "",
                font_size=7,
                x=0,
                y=0,
                color=(170, 170, 170, 200),
                batch=self.settings_batch,
            )
            for name in ("crop", "repeat", "scale"):
                rect = shapes.Rectangle(0, 0, SETTINGS_BUTTON_WIDTH, SETTINGS_BUTTON_HEIGHT, color=(40, 40, 55), batch=self.settings_batch)
                rect.opacity = 220
                label = pyglet.text.Label(
                    name,
                    font_size=8,
                    x=0,
                    y=0,
                    color=(220, 220, 220, 220),
                    anchor_x="center",
                    anchor_y="center",
                    batch=self.settings_batch,
                )
                self.mode_buttons.append({"name": name, "rect": rect, "label": label})

    def _cog_hit(self, x, y):
        if not self.cog_label:
            return False
        half_w = self.cog_label.content_width / 2.0
        half_h = self.cog_label.content_height / 2.0
        return (self.cog_label.x - half_w <= x <= self.cog_label.x + half_w) and (
            self.cog_label.y - half_h <= y <= self.cog_label.y + half_h
        )

    def _settings_panel_hit(self, x, y):
        if not self._settings_panel_bounds:
            return False
        x0, y0, x1, y1 = self._settings_panel_bounds
        return x0 <= x <= x1 and y0 <= y <= y1

    def _update_settings_ui(self):
        self._ensure_settings_ui()
        pad = SETTINGS_PANEL_PADDING
        cog_x = self.editor_rect.x + self.editor_rect.width - pad
        cog_y = self.editor_rect.y + self.editor_rect.height - pad
        self.cog_label.x = cog_x
        self.cog_label.y = cog_y
        if not self.settings_open:
            self.settings_panel.opacity = 0
            if self.settings_title:
                self.settings_title.color = (240, 240, 240, 0)
            for label in self.settings_labels.values():
                label.color = (*label.color[:3], 0)
            for swatch in self.font_color_swatches + self.face_color_swatches:
                swatch["rect"].opacity = 0
            for item in (self.font_rgb_box, self.face_rgb_box, self.image_button):
                if item:
                    item.opacity = 0
            for label in (self.font_rgb_label, self.face_rgb_label, self.image_label, self.image_info_label):
                if label:
                    label.color = (*label.color[:3], 0)
            for btn in self.mode_buttons:
                btn["rect"].opacity = 0
                btn["label"].color = (*btn["label"].color[:3], 0)
            return

        panel_x = self.editor_rect.x + self.editor_rect.width - SETTINGS_PANEL_WIDTH - pad
        panel_y = self.editor_rect.y + self.editor_rect.height - SETTINGS_PANEL_HEIGHT - pad
        panel_x = max(self.editor_rect.x + pad, panel_x)
        panel_y = max(self.editor_rect.y + pad, panel_y)
        self.settings_panel.x = panel_x
        self.settings_panel.y = panel_y
        self.settings_panel.width = SETTINGS_PANEL_WIDTH
        self.settings_panel.height = SETTINGS_PANEL_HEIGHT
        self.settings_panel.opacity = 235
        self._settings_panel_bounds = (
            panel_x,
            panel_y,
            panel_x + SETTINGS_PANEL_WIDTH,
            panel_y + SETTINGS_PANEL_HEIGHT,
        )
        x = panel_x + pad
        y = panel_y + SETTINGS_PANEL_HEIGHT - pad - 6
        if self.settings_title:
            self.settings_title.x = x
            self.settings_title.y = y
            self.settings_title.color = (240, 240, 240, 230)
        y -= 20
        self.settings_labels["font"].x = x
        self.settings_labels["font"].y = y
        self.settings_labels["font"].color = (210, 210, 210, 220)
        y -= 14
        for idx, swatch in enumerate(self.font_color_swatches):
            rect = swatch["rect"]
            rect.x = x + idx * (SETTINGS_SWATCH_SIZE + SETTINGS_SWATCH_GAP)
            rect.y = y
            rect.width = SETTINGS_SWATCH_SIZE
            rect.height = SETTINGS_SWATCH_SIZE
            rect.opacity = 230
        y -= SETTINGS_SWATCH_SIZE + 8
        self.settings_labels["font_rgb"].x = x
        self.settings_labels["font_rgb"].y = y + 4
        self.settings_labels["font_rgb"].color = (180, 180, 180, 220)
        self.font_rgb_box.x = x + 70
        self.font_rgb_box.y = y
        self.font_rgb_box.width = SETTINGS_BOX_WIDTH
        self.font_rgb_box.height = SETTINGS_BOX_HEIGHT
        self.font_rgb_box.opacity = 230
        self.font_rgb_label.x = self.font_rgb_box.x + 6
        self.font_rgb_label.y = self.font_rgb_box.y + 4
        y -= SETTINGS_BOX_HEIGHT + 12
        self.settings_labels["face"].x = x
        self.settings_labels["face"].y = y
        self.settings_labels["face"].color = (210, 210, 210, 220)
        y -= 14
        for idx, swatch in enumerate(self.face_color_swatches):
            rect = swatch["rect"]
            rect.x = x + idx * (SETTINGS_SWATCH_SIZE + SETTINGS_SWATCH_GAP)
            rect.y = y
            rect.width = SETTINGS_SWATCH_SIZE
            rect.height = SETTINGS_SWATCH_SIZE
            rect.opacity = 230
        y -= SETTINGS_SWATCH_SIZE + 8
        self.settings_labels["face_rgb"].x = x
        self.settings_labels["face_rgb"].y = y + 4
        self.settings_labels["face_rgb"].color = (180, 180, 180, 220)
        self.face_rgb_box.x = x + 70
        self.face_rgb_box.y = y
        self.face_rgb_box.width = SETTINGS_BOX_WIDTH
        self.face_rgb_box.height = SETTINGS_BOX_HEIGHT
        self.face_rgb_box.opacity = 230
        self.face_rgb_label.x = self.face_rgb_box.x + 6
        self.face_rgb_label.y = self.face_rgb_box.y + 4
        y -= SETTINGS_BOX_HEIGHT + 12
        self.settings_labels["image"].x = x
        self.settings_labels["image"].y = y + 4
        self.settings_labels["image"].color = (210, 210, 210, 220)
        self.image_button.x = x + 50
        self.image_button.y = y
        self.image_button.width = SETTINGS_BOX_WIDTH
        self.image_button.height = SETTINGS_BOX_HEIGHT
        self.image_button.opacity = 230
        self.image_label.x = self.image_button.x + 6
        self.image_label.y = self.image_button.y + 4
        y -= SETTINGS_BOX_HEIGHT + 10
        self.image_info_label.x = x
        self.image_info_label.y = y
        y -= 16
        self.settings_labels["mode"].x = x
        self.settings_labels["mode"].y = y + 4
        self.settings_labels["mode"].color = (180, 180, 180, 220)
        mode_x = x + 70
        mode_y = y
        for idx, btn in enumerate(self.mode_buttons):
            rect = btn["rect"]
            rect.x = mode_x + idx * (SETTINGS_BUTTON_WIDTH + 6)
            rect.y = mode_y
            rect.width = SETTINGS_BUTTON_WIDTH
            rect.height = SETTINGS_BUTTON_HEIGHT
            rect.opacity = 220
            btn["label"].x = rect.x + rect.width / 2
            btn["label"].y = rect.y + rect.height / 2
            btn["label"].color = (220, 220, 220, 220)

        face = self.faces[self.current_face_index]
        font_rgb_text = self._settings_input_text if self._settings_input_target == "font_rgb" else ",".join(
            str(c) for c in face.font_color
        )
        if face.background_def.get("type") == "color":
            face_color = parse_color(face.background_def.get("value"))
        else:
            face_color = parse_color(DEFAULT_BACKGROUNDS[face.name])
        face_rgb_text = self._settings_input_text if self._settings_input_target == "face_rgb" else ",".join(
            str(c) for c in face_color
        )
        self.font_rgb_label.text = font_rgb_text
        self.face_rgb_label.text = face_rgb_text
        if self._settings_input_target == "font_rgb":
            self.font_rgb_box.color = (55, 55, 70)
        else:
            self.font_rgb_box.color = (35, 35, 45)
        if self._settings_input_target == "face_rgb":
            self.face_rgb_box.color = (55, 55, 70)
        else:
            self.face_rgb_box.color = (35, 35, 45)
        image_name = "Select..."
        image_info = ""
        if face.background_def.get("type") == "image":
            image_path = Path(face.background_def.get("value", ""))
            if image_path.name:
                image_name = image_path.name
            if face.background_image():
                img = face.background_image()
                if isinstance(img, sprite.Sprite):
                    width = img.image.width
                    height = img.image.height
                else:
                    width = img.width
                    height = img.height
                multiples = ", ".join(
                    f"{width * mult}x{height * mult}" for mult in range(1, 4)
                )
                image_info = f"{width}x{height} -> {multiples}"
        if len(image_name) > 18:
            image_name = image_name[:15] + "..."
        self.image_label.text = image_name
        self.image_info_label.text = image_info
        if face.background_def.get("type") != "image":
            self.settings_labels["mode"].color = (180, 180, 180, 0)
            for btn in self.mode_buttons:
                btn["rect"].opacity = 0
                btn["label"].color = (*btn["label"].color[:3], 0)
        else:
            selected_mode = face.background_image_mode()
            for btn in self.mode_buttons:
                is_selected = btn["name"] == selected_mode
                btn["rect"].opacity = 220
                btn["rect"].color = (70, 90, 140) if is_selected else (40, 40, 55)
                btn["label"].color = (255, 255, 255, 230) if is_selected else (220, 220, 220, 220)

        self._settings_font_rgb_bounds = (
            self.font_rgb_box.x,
            self.font_rgb_box.y,
            self.font_rgb_box.x + self.font_rgb_box.width,
            self.font_rgb_box.y + self.font_rgb_box.height,
        )
        self._settings_face_rgb_bounds = (
            self.face_rgb_box.x,
            self.face_rgb_box.y,
            self.face_rgb_box.x + self.face_rgb_box.width,
            self.face_rgb_box.y + self.face_rgb_box.height,
        )
        self._settings_image_bounds = (
            self.image_button.x,
            self.image_button.y,
            self.image_button.x + self.image_button.width,
            self.image_button.y + self.image_button.height,
        )

    def _parse_rgb(self, text):
        cleaned = text.replace(" ", ",")
        parts = [part for part in cleaned.split(",") if part]
        if len(parts) < 3:
            return None
        try:
            values = [max(0, min(255, int(part))) for part in parts[:3]]
        except ValueError:
            return None
        return tuple(values)

    def _apply_font_color(self, rgb):
        face = self.faces[self.current_face_index]
        face.set_font_color(rgb)
        self.config_data["font_colors"][face.name] = list(rgb)
        save_config(self.config_data)

    def _apply_face_color(self, rgb):
        face = self.faces[self.current_face_index]
        hex_value = "#{:02x}{:02x}{:02x}".format(*rgb)
        face.set_background_def({"type": "color", "value": hex_value})
        self.config_data["backgrounds"][face.name] = {"type": "color", "value": hex_value}
        save_config(self.config_data)

    def _apply_rgb_input(self):
        if not self._settings_input_target:
            return
        rgb = self._parse_rgb(self._settings_input_text)
        if not rgb:
            self._settings_input_target = None
            self._settings_input_text = ""
            return
        if self._settings_input_target == "font_rgb":
            self._apply_font_color(rgb)
        elif self._settings_input_target == "face_rgb":
            self._apply_face_color(rgb)
        self._settings_input_target = None
        self._settings_input_text = ""

    def _select_image_for_face(self):
        try:
            from tkinter import Tk, filedialog
        except Exception:
            self._toast("Image picker unavailable (tkinter).")
            return
        root = Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select background image",
            filetypes=[("Media files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.mp4;*.mov;*.m4v;*.webm;*.avi"), ("All files", "*.*")],
        )
        root.destroy()
        if not path:
            return
        face = self.faces[self.current_face_index]
        bg_def = {"type": "image", "value": path, "mode": face.background_def.get("mode", "scale")}
        face.set_background_def(bg_def)
        self.config_data["backgrounds"][face.name] = bg_def
        save_config(self.config_data)

    def _handle_settings_click(self, x, y):
        if not self.settings_open or not self._settings_panel_bounds:
            return False
        if not self._settings_panel_hit(x, y):
            return False
        for swatch in self.font_color_swatches:
            rect = swatch["rect"]
            if rect.x <= x <= rect.x + rect.width and rect.y <= y <= rect.y + rect.height:
                self._apply_font_color(swatch["color"])
                return True
        for swatch in self.face_color_swatches:
            rect = swatch["rect"]
            if rect.x <= x <= rect.x + rect.width and rect.y <= y <= rect.y + rect.height:
                self._apply_face_color(swatch["color"])
                return True
        if self._settings_font_rgb_bounds:
            x0, y0, x1, y1 = self._settings_font_rgb_bounds
            if x0 <= x <= x1 and y0 <= y <= y1:
                self._settings_input_target = "font_rgb"
                self._settings_input_text = ""
                return True
        if self._settings_face_rgb_bounds:
            x0, y0, x1, y1 = self._settings_face_rgb_bounds
            if x0 <= x <= x1 and y0 <= y <= y1:
                self._settings_input_target = "face_rgb"
                self._settings_input_text = ""
                return True
        if self._settings_image_bounds:
            x0, y0, x1, y1 = self._settings_image_bounds
            if x0 <= x <= x1 and y0 <= y <= y1:
                self._select_image_for_face()
                return True
        face = self.faces[self.current_face_index]
        if face.background_def.get("type") == "image":
            for btn in self.mode_buttons:
                rect = btn["rect"]
                if rect.x <= x <= rect.x + rect.width and rect.y <= y <= rect.y + rect.height:
                    face.background_def["mode"] = btn["name"]
                    self.config_data["backgrounds"][face.name] = dict(face.background_def)
                    save_config(self.config_data)
                    return True
        return True

    def _draw_face_background_image(self, face, image):
        if isinstance(image, sprite.Sprite):
            self._draw_sprite_background(face, image)
            return
        mode = face.background_image_mode()
        if mode == "repeat":
            self._draw_image_tiled(image)
        elif mode == "crop":
            self._draw_image_cropped(image)
        else:
            self._draw_image_scaled(image)

    def _draw_sprite_background(self, face, image):
        mode = face.background_image_mode()
        if mode == "repeat":
            self._draw_sprite_tiled(image)
        elif mode == "crop":
            self._draw_sprite_cropped(image)
        else:
            self._draw_sprite_scaled(image)

    def _draw_sprite_scaled(self, image):
        if image.image.width <= 0 or image.image.height <= 0:
            return
        image.scale_x = self.editor_rect.width / image.image.width
        image.scale_y = self.editor_rect.height / image.image.height
        image.x = self.editor_rect.x
        image.y = self.editor_rect.y
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(
            int(self.editor_rect.x),
            int(self.editor_rect.y),
            int(self.editor_rect.width),
            int(self.editor_rect.height),
        )
        image.draw()
        gl.glDisable(gl.GL_SCISSOR_TEST)

    def _draw_sprite_cropped(self, image):
        if image.image.width <= 0 or image.image.height <= 0:
            return
        scale = max(self.editor_rect.width / image.image.width, self.editor_rect.height / image.image.height)
        draw_w = int(image.image.width * scale)
        draw_h = int(image.image.height * scale)
        image.scale_x = scale
        image.scale_y = scale
        image.x = int(self.editor_rect.x + (self.editor_rect.width - draw_w) / 2)
        image.y = int(self.editor_rect.y + (self.editor_rect.height - draw_h) / 2)
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(
            int(self.editor_rect.x),
            int(self.editor_rect.y),
            int(self.editor_rect.width),
            int(self.editor_rect.height),
        )
        image.draw()
        gl.glDisable(gl.GL_SCISSOR_TEST)

    def _draw_sprite_tiled(self, image):
        if image.image.width <= 0 or image.image.height <= 0:
            return
        original_x = image.x
        original_y = image.y
        original_scale_x = image.scale_x
        original_scale_y = image.scale_y
        image.scale_x = 1.0
        image.scale_y = 1.0
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(
            int(self.editor_rect.x),
            int(self.editor_rect.y),
            int(self.editor_rect.width),
            int(self.editor_rect.height),
        )
        start_x = int(self.editor_rect.x)
        start_y = int(self.editor_rect.y)
        end_x = int(self.editor_rect.x + self.editor_rect.width)
        end_y = int(self.editor_rect.y + self.editor_rect.height)
        step_x = max(1, image.image.width)
        step_y = max(1, image.image.height)
        for tx in range(start_x, end_x, step_x):
            for ty in range(start_y, end_y, step_y):
                image.x = tx
                image.y = ty
                image.draw()
        gl.glDisable(gl.GL_SCISSOR_TEST)
        image.x = original_x
        image.y = original_y
        image.scale_x = original_scale_x
        image.scale_y = original_scale_y

    def _draw_image_scaled(self, image):
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(
            int(self.editor_rect.x),
            int(self.editor_rect.y),
            int(self.editor_rect.width),
            int(self.editor_rect.height),
        )
        image.blit(
            self.editor_rect.x,
            self.editor_rect.y,
            width=self.editor_rect.width,
            height=self.editor_rect.height,
        )
        gl.glDisable(gl.GL_SCISSOR_TEST)

    def _draw_image_cropped(self, image):
        scale = max(self.editor_rect.width / image.width, self.editor_rect.height / image.height)
        draw_w = int(image.width * scale)
        draw_h = int(image.height * scale)
        draw_x = int(self.editor_rect.x + (self.editor_rect.width - draw_w) / 2)
        draw_y = int(self.editor_rect.y + (self.editor_rect.height - draw_h) / 2)
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(
            int(self.editor_rect.x),
            int(self.editor_rect.y),
            int(self.editor_rect.width),
            int(self.editor_rect.height),
        )
        image.blit(draw_x, draw_y, width=draw_w, height=draw_h)
        gl.glDisable(gl.GL_SCISSOR_TEST)

    def _draw_image_tiled(self, image):
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(
            int(self.editor_rect.x),
            int(self.editor_rect.y),
            int(self.editor_rect.width),
            int(self.editor_rect.height),
        )
        start_x = int(self.editor_rect.x)
        start_y = int(self.editor_rect.y)
        end_x = int(self.editor_rect.x + self.editor_rect.width)
        end_y = int(self.editor_rect.y + self.editor_rect.height)
        step_x = max(1, image.width)
        step_y = max(1, image.height)
        for tx in range(start_x, end_x, step_x):
            for ty in range(start_y, end_y, step_y):
                image.blit(tx, ty)
        gl.glDisable(gl.GL_SCISSOR_TEST)

    def _make_topmost(self):
        try:
            hwnd = self._hwnd  # type: ignore[attr-defined]
            HWND_TOPMOST = -1
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_SHOWWINDOW = 0x0040
            ctypes.windll.user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
        except Exception:
            pass

    def on_draw(self):
        self.clear()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self._setup_3d()
        self._update_face_previews()
        self._draw_cube()
        self._setup_2d()
        self._draw_editor()
        self._capture_screenshot_if_pending()

    def _setup_3d(self):
        self.cube.setup_3d(self, self.rotation)
        self._mvp_3d = self.projection @ self.view

    def _draw_cube(self):
        textures = {face.name: face.preview_texture() for face in self.faces if face.preview_texture()}
        if self.dragging:
            depths = {
                name: rotation_matrix_apply(self.cube.face_centers[name], self.rotation)[2]
                for name in self.cube.face_order
            }
            front_face = max(depths, key=depths.get)
            face_alpha = {name: self._cube_alpha_drag_other for name in self.cube.face_order}
            face_alpha[front_face] = self._cube_alpha_drag_front
        else:
            face_alpha = self._cube_alpha_normal
        if self.dragging:
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(770, 771)  # SRC_ALPHA, ONE_MINUS_SRC_ALPHA
            gl.glDepthMask(False)
        self.cube.draw_textured_with_alpha(
            textures,
            self.config_data.get("backgrounds", {}),
            self.rotation,
            face_alpha,
            sort_faces=self.dragging,
        )
        if self.dragging:
            gl.glDepthMask(True)

    def _setup_2d(self):
        gl.glDisable(gl.GL_DEPTH_TEST)
        self.projection = pyglet.math.Mat4.orthogonal_projection(0, self.width, 0, self.height, -1, 1)
        self.view = pyglet.math.Mat4()
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(770, 771)  # SRC_ALPHA, ONE_MINUS_SRC_ALPHA

    def _draw_editor(self):
        active_face = self.faces[self.current_face_index]
        color = active_face.background_color()
        if self.edit_mode and self._mvp_3d is not None:
            bbox = self._face_bbox_on_screen(active_face.name, self._mvp_3d)
            if bbox:
                x0, y0, x1, y1 = bbox
                width = max(1, int(x1 - x0))
                height = max(1, int(y1 - y0))
                side = max(1, min(width, height))
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                self.editor_rect.width = side
                self.editor_rect.height = side
                self.editor_rect.x = int(max(0, min(self.width - side, cx - side / 2.0)))
                self.editor_rect.y = int(max(0, min(self.height - side, cy - side / 2.0)))
            else:
                width = int(self.width * EDITOR_SCALE)
                height = int(self.height * EDITOR_SCALE)
                self.editor_rect.width = width
                self.editor_rect.height = height
                self.editor_rect.x = int((self.width - width) / 2)
                self.editor_rect.y = int((self.height - height) / 2)
        else:
            width = int(self.width * EDITOR_SCALE)
            height = int(self.height * EDITOR_SCALE)
            self.editor_rect.width = width
            self.editor_rect.height = height
            self.editor_rect.x = int((self.width - width) / 2)
            self.editor_rect.y = int((self.height - height) / 2)
        self.editor_rect.color = color
        self.editor_rect.opacity = 255 if self.edit_mode else 0
        self._update_labels()
        if self.toast_label:
            if time.time() <= self._toast_until:
                self.toast_label.color = (240, 240, 240, 220)
            else:
                self.toast_label.color = (240, 240, 240, 0)
        if self.edit_mode:
            if self.editor_rect.opacity > 0:
                self.editor_rect.draw()
            bg_image = active_face.background_image()
            if bg_image:
                self._draw_face_background_image(active_face, bg_image)
            self.ui_batch.draw()
            pad = 12
            active_face.layout.x = self.editor_rect.x + pad
            active_face.layout.y = self.editor_rect.y + pad
            active_face.layout.width = max(1, self.editor_rect.width - pad * 2)
            active_face.layout.height = max(1, self.editor_rect.height - pad * 2)
            self._apply_pending_caret()
            active_face.layout.draw()
            self._update_scrollbar(active_face, pad)
            self._draw_scrollbar()
            self._update_settings_ui()
            self.settings_batch.draw()
        else:
            self.settings_open = False
            self._settings_input_target = None
            # Draw labels (and any active toast); editor rect is fully transparent in rotate mode.
            self.ui_batch.draw()

    def on_resize(self, width, height):
        super().on_resize(width, height)
        width = max(EDITOR_MARGIN * 2 + 50, width)
        height = max(EDITOR_MARGIN * 2 + 50, height)
        for face in self.faces:
            face.resize_layout(width - 2 * EDITOR_MARGIN, height - 2 * EDITOR_MARGIN)
        self._build_editor_overlay()
        self._build_labels()
        self._build_face_keys()

    def on_mouse_press(self, x, y, button, modifiers):
        if button != mouse.LEFT:
            return
        if self.settings_open and self._handle_settings_click(x, y):
            return
        if self.edit_mode and self._cog_hit(x, y):
            self.settings_open = not self.settings_open
            self._settings_input_target = None
            self._settings_input_text = ""
            return
        if self.settings_open and not self._settings_panel_hit(x, y) and not self._cog_hit(x, y):
            self.settings_open = False
            self._settings_input_target = None
        if self._spin_key_hit(x, y):
            self.auto_spin = not self.auto_spin
            if self.auto_spin:
                _yaw, pitch = self._rotation_angles()
                if abs(pitch) < AUTO_SPIN_TILT_DEGREES * 0.5:
                    tilt = pyglet.math.Mat4.from_rotation(math.radians(AUTO_SPIN_TILT_DEGREES), Vec3(1, 0, 0))
                    self.rotation = tilt @ self.rotation
                self.edit_mode = False
            else:
                self.edit_mode = True
            return
        face_hit = self._face_key_hit(x, y)
        if face_hit:
            self._snap_to_face(face_hit)
            self.edit_mode = True
            self._return_to_edit_when_aligned = False
            self.dragging = False
            return
        if self.edit_mode and self._point_in_editor(x, y):
            self.edit_mode = True
            self._return_to_edit_when_aligned = False
            self._activate_caret(x, y, button, modifiers)
            return
        if not self._point_in_cube(x, y):
            self.edit_mode = False
            self._return_to_edit_when_aligned = False
            self.dragging = True
            self._drag_started_outside_cube = True
            self.last_mouse = (x, y)
            self._drag_vector = self._arcball_vector(x, y)
            return
        if not self.edit_mode:
            self.edit_mode = True
            self._return_to_edit_when_aligned = False
            self.dragging = False
            self._drag_started_outside_cube = False
            self._pending_edit_click = (x, y)
            return
        self.edit_mode = False
        self._return_to_edit_when_aligned = False
        self.dragging = True
        self._drag_started_outside_cube = False
        self.last_mouse = (x, y)
        self._drag_vector = self._arcball_vector(x, y)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & mouse.LEFT and self.dragging:
            if not self._drag_vector:
                self._drag_vector = self._arcball_vector(x, y)
            current_vec = self._arcball_vector(x, y)
            axis = self._drag_vector.cross(current_vec)
            axis_len = axis.length()
            if axis_len > 0:
                dot = max(-1.0, min(1.0, self._drag_vector.dot(current_vec)))
                angle = math.acos(dot) * ROTATE_SENSITIVITY
                rot = pyglet.math.Mat4.from_rotation(angle, axis.normalize())
                self.rotation = rot @ self.rotation
            self._drag_vector = current_vec
            self.last_mouse = (x, y)
            return
        if self.edit_mode and buttons & mouse.MIDDLE:
            active_face = self.faces[self.current_face_index]
            active_face.scroll(-dy * 2)
            return
        if self.edit_mode and self._point_in_editor(x, y):
            active_face = self.faces[self.current_face_index]
            if active_face.caret:
                active_face.caret.on_mouse_drag(x, y, dx, dy, buttons, modifiers)

    def on_mouse_release(self, x, y, button, modifiers):
        if button != mouse.LEFT:
            return
        if self.dragging:
            self.dragging = False
            self._drag_vector = None
            face_name = self._nearest_face()
            self._snap_to_face(face_name)
            if not self._drag_started_outside_cube:
                self.edit_mode = True
                self._return_to_edit_when_aligned = False
            else:
                self.edit_mode = False
            self._drag_started_outside_cube = False

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if not self.edit_mode:
            return
        active_face = self.faces[self.current_face_index]
        if active_face.caret:
            active_face.caret.on_mouse_scroll(x, y, scroll_x, scroll_y)
            return
        active_face.scroll(-scroll_y * 20)

    def on_text(self, text):
        if self.settings_open and self._settings_input_target:
            if text.isdigit() or text in {",", " "}:
                self._settings_input_text += text
            return
        if self.edit_mode:
            active_face = self.faces[self.current_face_index]
            if active_face.caret:
                active_face.record_undo()
                active_face.caret.on_text(text)
                active_face.mark_preview_dirty()

    def on_text_motion(self, motion):
        if self.settings_open and self._settings_input_target:
            return
        if self.edit_mode:
            active_face = self.faces[self.current_face_index]
            if active_face.caret:
                if motion in (key.MOTION_BACKSPACE, key.MOTION_DELETE, key.MOTION_PASTE):
                    active_face.record_undo()
                active_face.caret.on_text_motion(motion)
                if motion in (key.MOTION_BACKSPACE, key.MOTION_DELETE, key.MOTION_PASTE):
                    active_face.mark_preview_dirty()

    def on_text_motion_select(self, motion):
        if self.settings_open and self._settings_input_target:
            return
        if self.edit_mode:
            active_face = self.faces[self.current_face_index]
            if active_face.caret:
                active_face.caret.on_text_motion_select(motion)
                active_face.mark_preview_dirty()

    def on_key_press(self, symbol, modifiers):
        if self.settings_open and self._settings_input_target:
            if symbol in (key.ENTER, key.NUM_ENTER):
                self._apply_rgb_input()
            elif symbol == key.BACKSPACE:
                self._settings_input_text = self._settings_input_text[:-1]
            elif symbol == key.ESCAPE:
                self._settings_input_target = None
                self._settings_input_text = ""
            return
        if not self.edit_mode and symbol in (key.LEFT, key.RIGHT, key.UP, key.DOWN):
            self._rotate_via_keys(symbol, return_to_edit=False, play_sound=True)
            return
        if modifiers & key.MOD_ALT:
            if symbol in (key.LEFT, key.RIGHT, key.UP, key.DOWN):
                self._rotate_via_keys(symbol, return_to_edit=True, play_sound=True)
                return
        if self.edit_mode and (modifiers & key.MOD_CTRL):
            active_face = self.faces[self.current_face_index]
            if symbol == key.A:
                if active_face.caret:
                    active_face.caret.select_all()
                return
            if symbol == key.Z:
                if modifiers & key.MOD_SHIFT:
                    active_face.redo()
                else:
                    active_face.undo()
                return
            if symbol == key.Y:
                active_face.redo()
                return
        if symbol == key.F12:
            self._queue_screenshot()
            return
        if modifiers & key.MOD_CTRL and symbol == key.S:
            self.save_all("manual save")
            return
        if symbol == key.ESCAPE:
            if self.settings_open:
                self.settings_open = False
                self._settings_input_target = None
                return
            self.edit_mode = not self.edit_mode
            if self.edit_mode:
                self._return_to_edit_when_aligned = False
            return
        # Text editing is handled via on_text / on_text_motion events; pyglet 2.x Caret has no on_key_press.

    def _point_in_editor(self, x, y):
        return (
            self.editor_rect.x <= x <= self.editor_rect.x + self.editor_rect.width
            and self.editor_rect.y <= y <= self.editor_rect.y + self.editor_rect.height
        )

    def _cube_bbox_on_screen(self, mvp):
        pts = []
        positions = self.cube.positions
        for i in range(0, len(positions), 3):
            x, y, z = positions[i], positions[i + 1], positions[i + 2]
            clip = mvp @ Vec4(x, y, z, 1.0)
            if clip.w == 0:
                continue
            ndc_x = clip.x / clip.w
            ndc_y = clip.y / clip.w
            sx = (ndc_x * 0.5 + 0.5) * self.width
            sy = (ndc_y * 0.5 + 0.5) * self.height
            pts.append((sx, sy))
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0 = max(0, min(xs))
        y0 = max(0, min(ys))
        x1 = min(self.width, max(xs))
        y1 = min(self.height, max(ys))
        if x1 - x0 < 10 or y1 - y0 < 10:
            return None
        return x0, y0, x1, y1

    def _point_in_cube(self, x, y):
        if self._mvp_3d is None:
            return self._point_in_editor(x, y)
        bbox = self._cube_bbox_on_screen(self._mvp_3d)
        if not bbox:
            return self._point_in_editor(x, y)
        x0, y0, x1, y1 = bbox
        return x0 <= x <= x1 and y0 <= y <= y1

    def _activate_caret(self, x, y, button, modifiers):
        active_face = self.faces[self.current_face_index]
        if active_face.caret:
            active_face.caret.on_mouse_press(x, y, button, modifiers)

    def _face_key_hit(self, x, y):
        for item in self.face_key_items:
            rect = item["rect"]
            if rect.x <= x <= rect.x + rect.width and rect.y <= y <= rect.y + rect.height:
                return item["name"]
        return None

    def _spin_key_hit(self, x, y):
        if not self.spin_key_item:
            return False
        rect = self.spin_key_item["rect"]
        return rect.x <= x <= rect.x + rect.width and rect.y <= y <= rect.y + rect.height

    def _arcball_vector(self, x, y):
        if self.width == 0 or self.height == 0:
            return Vec3(0, 0, 1)
        nx = (2.0 * x - self.width) / self.width
        ny = (2.0 * y - self.height) / self.height
        length_sq = nx * nx + ny * ny
        if length_sq > 1.0:
            length = math.sqrt(length_sq)
            return Vec3(nx / length, ny / length, 0.0)
        return Vec3(nx, ny, math.sqrt(1.0 - length_sq))

    def _nearest_face(self):
        best_face = "front"
        best_score = -999
        for name, normal in CubeRenderer.NORMALS.items():
            rotated = rotation_matrix_apply(normal, self.rotation)
            score = rotated[2]  # alignment with viewer's Z+
            if score > best_score:
                best_score = score
                best_face = name
        return best_face

    def _rotation_for_face(self, face_name):
        yaw, pitch = CubeRenderer.ORIENTATIONS.get(face_name, (0.0, 0.0))
        return rotation_from_yaw_pitch(yaw, pitch)

    def _face_bbox_on_screen(self, face_name, mvp):
        verts = self.cube.face_verts.get(face_name)
        if not verts:
            return None
        pts = []
        for x, y, z in verts:
            clip = mvp @ Vec4(x, y, z, 1.0)
            if clip.w == 0:
                continue
            ndc_x = clip.x / clip.w
            ndc_y = clip.y / clip.w
            sx = (ndc_x * 0.5 + 0.5) * self.width
            sy = (ndc_y * 0.5 + 0.5) * self.height
            pts.append((sx, sy))
        if len(pts) != 4:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0 = max(0, min(xs))
        y0 = max(0, min(ys))
        x1 = min(self.width, max(xs))
        y1 = min(self.height, max(ys))
        if x1 - x0 < 10 or y1 - y0 < 10:
            return None
        return x0, y0, x1, y1

    def _update_face_previews(self):
        # Only render previews when needed (requires a valid GL context, so keep this in on_draw).
        preview_size = self._preview_render_size()
        for face in self.faces:
            if getattr(face, "_preview_size", None) != preview_size:
                face.preview_dirty = True
        any_dirty = any(face.preview_dirty for face in self.faces)
        if not any_dirty:
            return
        # Update at least the active face immediately; update the rest opportunistically.
        active = self.faces[self.current_face_index]
        if active.preview_dirty:
            active.render_preview_to_texture(self, preview_size)
        for face in self.faces:
            if face is active:
                continue
            if face.preview_dirty:
                face.render_preview_to_texture(self, preview_size)

    def _preview_render_size(self):
        size = int(min(self.width, self.height) * EDITOR_SCALE)
        if self._mvp_3d is not None:
            active_face = self.faces[self.current_face_index]
            bbox = self._face_bbox_on_screen(active_face.name, self._mvp_3d)
            if bbox:
                x0, y0, x1, y1 = bbox
                side = int(max(1, min(x1 - x0, y1 - y0)))
                size = side
        return max(128, size)

    def _rotation_angles(self):
        forward = rotation_matrix_apply((0, 0, 1), self.rotation)
        yaw = math.degrees(math.atan2(forward[0], forward[2]))
        horiz = math.hypot(forward[0], forward[2])
        pitch = math.degrees(math.atan2(forward[1], horiz))
        return yaw, pitch

    def _queue_screenshot(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        face = self.faces[self.current_face_index].name
        shots_dir = DATA_DIR / "screenshots"
        shots_dir.mkdir(exist_ok=True)
        yaw, pitch = self._rotation_angles()
        path = shots_dir / f"{timestamp}-{face}-yaw{int(round(yaw))}-pitch{int(round(pitch))}.png"
        self._pending_screenshot_path = path
        self._toast(f"Screenshot queued: {path.name}")

    def _capture_screenshot_if_pending(self):
        if not self._pending_screenshot_path:
            return
        path = self._pending_screenshot_path
        self._pending_screenshot_path = None
        try:
            gl.glFinish()
            pyglet.image.get_buffer_manager().get_color_buffer().save(path.as_posix())
            self._toast(f"Saved screenshot: {path.name}")
            print(f"Saved screenshot: {path}")
        except Exception as exc:
            self._toast("Screenshot failed (see console).")
            print(f"Screenshot failed: {exc}")

    def _toast(self, message, seconds=2.0):
        if self.toast_label is None:
            self.toast_label = pyglet.text.Label(
                message,
                font_size=11,
                x=20,
                y=20,
                color=(240, 240, 240, 220),
                batch=self.ui_batch,
            )
        else:
            self.toast_label.text = message
        self.toast_label.x = 20
        self.toast_label.y = 20
        self._toast_until = time.time() + float(seconds)

    def _snap_to_face(self, face_name):
        if face_name not in FACE_NAMES:
            face_name = "front"
        self.current_face_index = FACE_NAMES.index(face_name)
        # Snap immediately so face + text are never out of alignment.
        self.rotation = self._rotation_for_face(face_name)
        self.config_data["last_face"] = self.current_face_index
        save_config(self.config_data)
        self._build_editor_overlay()
        self._build_labels()

    def _rotate_via_keys(self, symbol, return_to_edit=True, play_sound=False):
        name = self.faces[self.current_face_index].name
        ring = ["front", "right", "back", "left"]
        vertical_ring = ["front", "top", "back", "bottom"]
        if symbol in (key.LEFT, key.RIGHT):
            if name not in ring:
                name = "front"
            idx = ring.index(name)
            if symbol == key.RIGHT:
                idx = (idx + 1) % len(ring)
            else:
                idx = (idx - 1) % len(ring)
            self._snap_to_face(ring[idx])
        elif symbol == key.UP:
            if name not in vertical_ring:
                name = "front"
            idx = vertical_ring.index(name)
            idx = (idx + 1) % len(vertical_ring)
            self._snap_to_face(vertical_ring[idx])
        elif symbol == key.DOWN:
            if name not in vertical_ring:
                name = "front"
            idx = vertical_ring.index(name)
            idx = (idx - 1) % len(vertical_ring)
            self._snap_to_face(vertical_ring[idx])
        if play_sound:
            self._play_woosh()
        if return_to_edit:
            self.edit_mode = True
            self._return_to_edit_when_aligned = False

    def _play_woosh(self):
        try:
            envelope = synthesis.LinearDecayEnvelope(peak=0.35)
            synthesis.WhiteNoise(0.18, envelope=envelope).play()
        except Exception:
            pass

    def update(self, dt):
        if time.time() - self.last_save > AUTOSAVE_SECONDS:
            self.save_all("autosave")
            self.last_save = time.time()
        if self.auto_spin and not self.dragging:
            rot = pyglet.math.Mat4.from_rotation(self.auto_spin_speed * dt, Vec3(0, 1, 0))
            self.rotation = rot @ self.rotation
            self.rotation = orthonormalize_rotation(self.rotation)

    def save_all(self, reason):
        for face in self.faces:
            face.save()
        save_config(self.config_data)
        git_sync(f"{reason} - {self.faces[self.current_face_index].name}", self.config_data.get("remote"))
        self.last_save = time.time()

    def on_close(self):
        self.save_all("app close")
        return super().on_close()


def headless_test():
    config = load_config()
    ensure_data_folder()
    faces = []
    for name in FACE_NAMES:
        face = FaceState(
            name,
            DATA_DIR / f"{name}.txt",
            config["backgrounds"].get(name),
            config["font_colors"].get(name),
        )
        faces.append(face)
    for face in faces:
        face.save()
    git_sync("headless test", config.get("remote"))
    print("Headless test completed (files + git).")


def main():
    parser = argparse.ArgumentParser(description="Notes Cubed - floating cube notes")
    parser.add_argument("--headless-test", action="store_true", help="run file + git checks without UI")
    args = parser.parse_args()
    if args.headless_test:
        headless_test()
        return
    config = load_config()
    app = NotesCubedApp(config)
    pyglet.app.run()


if __name__ == "__main__":
    main()
