# Specifications

User-level behavior for Notes Cubed. Update this document whenever a new requirement is added or a behavior detail is discovered.

## Core experience
- A borderless, always-on-top window fills the screen and renders a 3D cube.
- The cube has six faces: front, right, back, left, top, bottom.
- Each face is its own text document with its own scroll position, background, and font color.
- The current face is shown in the UI and persists across launches.

## Modes and rotation
- Edit mode shows the editor and allows typing and selection.
- Rotate mode hides the editor and shows the cube plus UI labels.
- Dragging rotates the cube; releasing snaps to the nearest face.
- Clicking a cube face in rotate mode switches to edit mode for that face.
- Clicking outside the cube in rotate mode does not enter edit mode.
- Auto-spin rotates the cube continuously until stopped.
- When auto-spin is active, any click stops it; clicking the spin key toggles it on or off.

## Mouse and keyboard controls
- Left-click inside the editor places the caret.
- Mouse wheel scrolls the current face.
- Middle-button drag scrolls the current face.
- `Alt`+`Left/Right/Up/Down` rotates to the target face and returns to edit mode.
- `Ctrl`+`S` saves all faces and runs Git sync.
- `Ctrl`+`A` selects all; `Ctrl`+`Z` undo; `Ctrl`+`Shift`+`Z` redo; `Ctrl`+`Y` redo.
- `F12` saves a PNG screenshot to `data/screenshots/`.
- `Esc` toggles rotate/edit mode (or closes the settings panel if open).
- Double-click outside the cube minimizes the window.

## On-screen keys and labels
- Face keys snap the cube to the selected face.
- The spin key toggles auto-spin.
- Keys are visible in both edit and rotate modes.

## Settings panel
- A cog icon in edit mode opens per-face settings.
- Font color and face color can be set via preset swatches.
- Font RGB and face RGB accept comma- or space-separated `0-255` values.
- Background images can be selected via a file picker.
- Image mode can be set to crop, repeat, or scale.
- Clearing the image restores the face's default background color.
- Background images apply to the editor; the cube face uses the configured color.

## Persistence and sync
- Text and config data live in `data/`.
- Autosave runs periodically and on exit.
- Manual save (`Ctrl`+`S`) saves all faces, commits in `data/`, and pushes to `origin main` if a remote is configured.
- If Git is unavailable or push fails, the app continues running and logs the error.
