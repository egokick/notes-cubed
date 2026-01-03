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
}
EDITOR_MARGIN = 80
EDITOR_SCALE = 0.72  # fraction of window size used by editor overlay
AUTOSAVE_SECONDS = 30
ROTATE_SENSITIVITY = 2.0
FACE_PREVIEW_SIZE = 512
FACE_PREVIEW_PADDING = 18
FACE_KEY_WIDTH = 120
FACE_KEY_HEIGHT = 22
FACE_KEY_SPACING = 6
FACE_KEY_MARGIN = 18


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
    if "remote" not in data:
        data["remote"] = ""
    if "last_face" not in data:
        data["last_face"] = 0
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
    def __init__(self, name, path, background_def):
        self.name = name
        self.path = path
        self.background_def = background_def or {"type": "color", "value": DEFAULT_BACKGROUNDS[name]}
        self.document = pyglet.text.document.UnformattedDocument(self._load_text())
        base_style = {"font_name": "Consolas", "font_size": 14, "color": (255, 255, 255, 255)}
        self.document.set_style(0, len(self.document.text), base_style)
        self.layout = None
        self.caret = None
        self._bg_image = None
        self.preview_dirty = True
        self._preview_last_text = None
        self._preview_texture = None
        self._preview_fbo = None
        self._preview_batch = None
        self._preview_doc = None
        self._preview_layout = None
        self._preview_bg = None
        self._load_background_asset()

    def _load_text(self):
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def save(self):
        self.path.write_text(self.document.text, encoding="utf-8")

    def bind_layout(self, width, height):
        self.layout = pyglet.text.layout.IncrementalTextLayout(
            self.document, width=width, height=height, multiline=True, wrap_lines=False
        )
        self.layout.x = EDITOR_MARGIN
        self.layout.y = EDITOR_MARGIN
        self.caret = pyglet.text.caret.Caret(self.layout, color=(255, 255, 255))

    def resize_layout(self, width, height):
        if not self.layout:
            return
        self.layout.width = width
        self.layout.height = height
        self.layout.x = EDITOR_MARGIN
        self.layout.y = EDITOR_MARGIN

    def mark_preview_dirty(self):
        self.preview_dirty = True

    def preview_texture(self):
        return self._preview_texture

    def _ensure_preview_resources(self):
        if self._preview_texture is not None:
            return
        self._preview_texture = pyglet.image.Texture.create(FACE_PREVIEW_SIZE, FACE_PREVIEW_SIZE)
        self._preview_fbo = Framebuffer()
        self._preview_fbo.attach_texture(self._preview_texture)
        self._preview_batch = pyglet.graphics.Batch()
        self._preview_bg = shapes.Rectangle(
            0,
            0,
            FACE_PREVIEW_SIZE,
            FACE_PREVIEW_SIZE,
            color=(0, 0, 0),
            batch=self._preview_batch,
        )
        self._preview_bg.opacity = 0
        self._preview_doc = pyglet.text.document.UnformattedDocument("")
        self._preview_doc.set_style(
            0,
            0,
            {"font_name": "Consolas", "font_size": 14, "color": (255, 255, 255, 255)},
        )
        self._preview_layout = pyglet.text.layout.TextLayout(
            self._preview_doc,
            x=FACE_PREVIEW_PADDING,
            y=FACE_PREVIEW_PADDING,
            width=FACE_PREVIEW_SIZE - FACE_PREVIEW_PADDING * 2,
            height=FACE_PREVIEW_SIZE - FACE_PREVIEW_PADDING * 2,
            multiline=True,
            wrap_lines=True,
            batch=self._preview_batch,
        )

    def render_preview_to_texture(self, window):
        self._ensure_preview_resources()
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
                {"font_name": "Consolas", "font_size": 14, "color": (255, 255, 255, 255)},
            )

        prev_view = window.view
        prev_proj = window.projection
        prev_viewport = window.viewport

        try:
            self._preview_fbo.bind()
            gl.glViewport(0, 0, FACE_PREVIEW_SIZE, FACE_PREVIEW_SIZE)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(770, 771)  # SRC_ALPHA, ONE_MINUS_SRC_ALPHA
            gl.glClearColor(0.0, 0.0, 0.0, 0.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            window.projection = pyglet.math.Mat4.orthogonal_projection(0, FACE_PREVIEW_SIZE, 0, FACE_PREVIEW_SIZE, -1, 1)
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
        return self._bg_image

    def _load_background_asset(self):
        if self.background_def.get("type") != "image":
            self._bg_image = None
            return
        path = Path(self.background_def.get("value", ""))
        if not path.exists():
            self._bg_image = None
            return
        try:
            self._bg_image = pyglet.image.load(path.as_posix())
        except Exception:
            self._bg_image = None


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
            face = FaceState(name, DATA_DIR / f"{name}.txt", config["backgrounds"].get(name))
            face.bind_layout(self.width - 2 * EDITOR_MARGIN, self.height - 2 * EDITOR_MARGIN)
            self.faces.append(face)
        self.current_face_index = max(0, min(len(self.faces) - 1, config.get("last_face", 0)))

        self.rotation = self._rotation_for_face(self.faces[self.current_face_index].name)

        self.dragging = False
        self.last_mouse = (0, 0)
        self._drag_vector = None
        self.edit_mode = True
        self._return_to_edit_when_aligned = False
        self.last_save = time.time()

        self.ui_batch = pyglet.graphics.Batch()
        self.editor_rect = None
        self.face_label = None
        self.mode_label = None
        self.toast_label = None
        self.face_key_items = []
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
                batch=self.ui_batch,
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

    def _build_face_keys(self):
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
                    face_name.capitalize(),
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
        current_name = self.faces[self.current_face_index].name
        for index, item in enumerate(self.face_key_items):
            rect = item["rect"]
            label = item["label"]
            rect.x = right - FACE_KEY_WIDTH
            rect.y = top - (FACE_KEY_HEIGHT + FACE_KEY_SPACING) * index - FACE_KEY_HEIGHT
            label.x = rect.x + 8
            label.y = rect.y + FACE_KEY_HEIGHT / 2
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
        self._update_face_previews()
        self._setup_3d()
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
        self.editor_rect.opacity = 210 if self.edit_mode else 0
        self._update_labels()
        if self.toast_label:
            if time.time() <= self._toast_until:
                self.toast_label.color = (240, 240, 240, 220)
            else:
                self.toast_label.color = (240, 240, 240, 0)
        if self.edit_mode:
            bg_image = active_face.background_image()
            if bg_image:
                bg_image.blit(
                    self.editor_rect.x,
                    self.editor_rect.y,
                    width=self.editor_rect.width,
                    height=self.editor_rect.height,
                )
            self.ui_batch.draw()
            pad = 12
            active_face.layout.x = self.editor_rect.x + pad
            active_face.layout.y = self.editor_rect.y + pad
            active_face.layout.width = max(1, self.editor_rect.width - pad * 2)
            active_face.layout.height = max(1, self.editor_rect.height - pad * 2)
            active_face.layout.draw()
        else:
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
        face_hit = self._face_key_hit(x, y)
        if face_hit:
            self._snap_to_face(face_hit)
            self.edit_mode = True
            self._return_to_edit_when_aligned = False
            self.dragging = False
            return
        if self._point_in_editor(x, y):
            self.edit_mode = True
            self._return_to_edit_when_aligned = False
            self._activate_caret(x, y, button, modifiers)
            return
        self.edit_mode = False
        self._return_to_edit_when_aligned = False
        self.dragging = True
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
            self.edit_mode = True
            self._return_to_edit_when_aligned = False

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if not self.edit_mode:
            return
        active_face = self.faces[self.current_face_index]
        if active_face.caret:
            active_face.caret.on_mouse_scroll(x, y, scroll_x, scroll_y)
            return
        active_face.scroll(-scroll_y * 20)

    def on_text(self, text):
        if self.edit_mode:
            active_face = self.faces[self.current_face_index]
            if active_face.caret:
                active_face.caret.on_text(text)
                active_face.mark_preview_dirty()

    def on_text_motion(self, motion):
        if self.edit_mode:
            active_face = self.faces[self.current_face_index]
            if active_face.caret:
                active_face.caret.on_text_motion(motion)
                active_face.mark_preview_dirty()

    def on_text_motion_select(self, motion):
        if self.edit_mode:
            active_face = self.faces[self.current_face_index]
            if active_face.caret:
                active_face.caret.on_text_motion_select(motion)
                active_face.mark_preview_dirty()

    def on_key_press(self, symbol, modifiers):
        if modifiers & key.MOD_ALT:
            if symbol in (key.LEFT, key.RIGHT, key.UP, key.DOWN):
                self._rotate_via_keys(symbol)
                return
        if symbol == key.F12:
            self._queue_screenshot()
            return
        if modifiers & key.MOD_CTRL and symbol == key.S:
            self.save_all("manual save")
            return
        if symbol == key.ESCAPE:
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
        any_dirty = any(face.preview_dirty for face in self.faces)
        if not any_dirty:
            return
        # Update at least the active face immediately; update the rest opportunistically.
        active = self.faces[self.current_face_index]
        if active.preview_dirty:
            active.render_preview_to_texture(self)
        for face in self.faces:
            if face is active:
                continue
            if face.preview_dirty:
                face.render_preview_to_texture(self)

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

    def _rotate_via_keys(self, symbol):
        name = self.faces[self.current_face_index].name
        ring = ["front", "right", "back", "left"]
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
            self._snap_to_face("top")
        elif symbol == key.DOWN:
            self._snap_to_face("bottom")
        self.edit_mode = True
        self._return_to_edit_when_aligned = False

    def update(self, dt):
        if time.time() - self.last_save > AUTOSAVE_SECONDS:
            self.save_all("autosave")
            self.last_save = time.time()

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
        face = FaceState(name, DATA_DIR / f"{name}.txt", config["backgrounds"].get(name))
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
