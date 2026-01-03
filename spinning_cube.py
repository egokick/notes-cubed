import math

import pyglet
from pyglet import gl
from pyglet.graphics import shader
from pyglet.math import Mat4, Vec3


class SpinningCube(pyglet.window.Window):
    def __init__(self):
        config = gl.Config(double_buffer=True, depth_size=24, sample_buffers=1, samples=4)
        super().__init__(800, 600, "Spinning Cube", resizable=True, config=config)

        gl.glClearColor(0.02, 0.02, 0.03, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)

        self.program = shader.ShaderProgram(
            shader.Shader(
                """#version 150 core
                in vec3 position;
                in vec4 colors;

                uniform mat4 projection;
                uniform mat4 view;

                out vec4 v_color;

                void main()
                {
                    gl_Position = projection * view * vec4(position, 1.0);
                    v_color = colors;
                }
                """,
                "vertex",
            ),
            shader.Shader(
                """#version 150 core
                in vec4 v_color;
                out vec4 final_color;

                void main()
                {
                    final_color = v_color;
                }
                """,
                "fragment",
            ),
        )

        self.yaw = 0.0
        self.pitch = 0.0
        self._phase = 0.0

        self._build_geometry()
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)

    def _build_geometry(self):
        s = 1.2
        faces = [
            ("front", [(s, -s, s), (-s, -s, s), (-s, s, s), (s, s, s)], (220, 90, 120, 255)),
            ("back", [(-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s)], (80, 140, 220, 255)),
            ("left", [(-s, -s, s), (-s, -s, -s), (-s, s, -s), (-s, s, s)], (100, 220, 160, 255)),
            ("right", [(s, -s, -s), (s, -s, s), (s, s, s), (s, s, -s)], (240, 210, 90, 255)),
            ("top", [(-s, s, s), (-s, s, -s), (s, s, -s), (s, s, s)], (180, 140, 240, 255)),
            ("bottom", [(-s, -s, -s), (-s, -s, s), (s, -s, s), (s, -s, -s)], (220, 140, 90, 255)),
        ]

        positions = []
        indices = []
        colors = []
        for _name, verts, rgba in faces:
            start = len(positions) // 3
            for x, y, z in verts:
                positions.extend([x, y, z])
                colors.extend(rgba)
            indices.extend([start, start + 1, start + 2, start, start + 2, start + 3])

        self._triangles = self.program.vertex_list_indexed(
            len(positions) // 3,
            gl.GL_TRIANGLES,
            indices,
            position=("f", positions),
            colors=("Bn", colors),
        )

        # Edges (wireframe overlay)
        edge_pairs = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        # First 8 verts correspond to front+back? We duplicated vertices per face, so rebuild edges from unique cube verts.
        unique = [
            (s, -s, s),
            (-s, -s, s),
            (-s, s, s),
            (s, s, s),
            (s, -s, -s),
            (-s, -s, -s),
            (-s, s, -s),
            (s, s, -s),
        ]
        edge_positions = [c for v in unique for c in v]
        edge_indices = [i for a, b in edge_pairs for i in (a, b)]
        self._edges = self.program.vertex_list_indexed(
            len(edge_positions) // 3,
            gl.GL_LINES,
            edge_indices,
            position=("f", edge_positions),
            colors=("Bn", [15, 15, 20, 255] * (len(edge_positions) // 3)),
        )

    def on_resize(self, width, height):
        super().on_resize(width, height)
        self.viewport = 0, 0, width, height

    def update(self, dt):
        self._phase += dt
        self.yaw = (self.yaw + dt * 40.0) % 360.0
        self.pitch = 20.0 + 15.0 * math.sin(self._phase * 0.8)

    def on_draw(self):
        self.clear()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        aspect = self.width / float(self.height or 1)
        projection = Mat4.perspective_projection(aspect, 0.1, 100.0, 60.0)

        rot_y = Mat4.from_rotation(math.radians(self.yaw), Vec3(0, 1, 0))
        rot_x = Mat4.from_rotation(math.radians(self.pitch), Vec3(1, 0, 0))
        view = Mat4.look_at(Vec3(0, 0, 4.0), Vec3(0, 0, 0), Vec3(0, 1, 0)) @ rot_y @ rot_x

        self.program.use()
        self.program["projection"] = projection
        self.program["view"] = view
        self._triangles.draw(gl.GL_TRIANGLES)
        self._edges.draw(gl.GL_LINES)
        self.program.stop()


if __name__ == "__main__":
    SpinningCube()
    pyglet.app.run()
