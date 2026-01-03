from pathlib import Path

import pyglet

from spinning_cube import SpinningCube


def main():
    out_path = Path("spinning_cube_capture.png").resolve()

    window = SpinningCube()
    window.set_visible(True)

    def capture(_dt):
        pyglet.image.get_buffer_manager().get_color_buffer().save(out_path.as_posix())
        print(f"Saved: {out_path}")
        window.close()
        pyglet.app.exit()

    pyglet.clock.schedule_once(capture, 0.25)
    pyglet.app.run()


if __name__ == "__main__":
    main()
