# Specifications

User-level behavior for Notes Cubed. Update this document whenever a new requirement is added or a behavior detail is discovered.

## Core experience
- A borderless, always-on-top window fills the default screen (positioned at 0,0) and renders a 3D cube.
- The window background is transparent; only the cube, editor, and UI labels are visible.
- The cube has six faces: front, right, back, left, top, bottom.
- Each face has its own text document, scroll position, background (color or media), and font color.
- The current face is shown in the UI and persists across launches.

## Modes and rotation
- Edit mode shows the editor overlay and allows typing and selection.
- Rotate mode hides the editor (fully transparent) but keeps the cube and UI labels visible.
- In rotate mode, the cube edges are drawn with a 1px black outline.
- In edit mode, left-click drag outside the editor area rotates the cube.
- In rotate mode, left-click drag anywhere not on UI keys rotates the cube.
- Text positioning aligns between edit mode and rotate mode so face text does not shift.
- Releasing after a drag snaps to the nearest face and sets it as the current face.
- If the drag starts outside the cube, releasing keeps the app in rotate mode.
- If the drag starts on the cube, releasing returns to edit mode.
- Clicking the cube (not a UI key) in rotate mode snaps to the nearest face and returns to edit mode.
- Clicking outside the cube in rotate mode does not enter edit mode.
- Auto-spin rotates the cube continuously around the Y axis and keeps the app in rotate mode.
- When auto-spin is active, any click stops it; the click then follows normal rotate-mode rules.

## Mouse and keyboard controls
- Left-click inside the editor places the caret (edit mode).
- Mouse wheel scrolls the current face (edit mode).
- Middle-button drag scrolls the current face (edit mode).
- `Alt`+`Left/Right/Up/Down` rotates to the target face and returns to edit mode.
- `Left/Right/Up/Down` in rotate mode snaps to a face and stays in rotate mode.
- `Home` moves the caret to the start of the current line (`Shift`+`Home` selects to the start of the line).
- `Tab` inserts a tab character or indents selected lines; `Shift`+`Tab` removes one indent from the current line or selection (edit mode).
- `Ctrl`+`S` saves all faces and runs Git sync.
- `Ctrl`+`Backspace/Delete` deletes the previous/next word (edit mode).
- `Ctrl`+`C/X` copies/cuts the selection; if nothing is selected it copies/cuts the current line (edit mode).
- `Ctrl`+`V` (or `Ctrl`+`Shift`+`V`) pastes from the clipboard (edit mode).
- `Ctrl`+`D` duplicates the current line or selected lines (edit mode).
- `Alt`+`Up/Down` moves the current line or selected lines; `Shift`+`Alt`+`Up/Down` copies them (edit mode).
- `Ctrl`+`L` deletes the current line (edit mode).
- `Ctrl`+`A` selects all; `Ctrl`+`Z` undo; `Ctrl`+`Shift`+`Z` redo; `Ctrl`+`Y` redo.
- `F12` saves a PNG screenshot to `data/screenshots/` with the current face and rotation in the filename.
- `Esc` cancels RGB entry if active; otherwise it closes the settings panel, and if neither are active it toggles rotate/edit mode.
- Double-click anywhere outside the cube (two clicks within ~0.35s) minimizes the window.

## On-screen keys and labels
- Face keys snap the cube to the selected face; they keep the current mode (edit stays edit, rotate stays rotate).
- The spin key toggles auto-spin.
- Keys are visible in both edit and rotate modes.

## Settings panel
- A cog icon in edit mode opens per-face settings and sits in the bottom-right of the editor.
- Clicking outside the settings panel closes it without triggering other actions.
- Font color and face color can be set via preset swatches.
- Font and face preset swatches share the same color-sorted palette, including black, white, and the primary colors.
- Clicking Font RGB or Face RGB opens a color picker; canceling it enables manual entry.
- Font RGB and face RGB fields prefill the current value and accept `R,G,B`, `R G B`, or hex (`#RRGGBB`/`#RGB`); `Enter` applies and `Esc` cancels.
- Font size can be set per face from the settings panel (range 8-48, default 14).
- Font thickness can be set per face from the settings panel (1 = normal, 2 = bold).
- Font size and font thickness boxes include left/right arrow buttons for decrement/increment.
- Background Preset dropdown lists images from `data/presetimages` and applies the selected image to the current face.
- On first run (no `config.json` yet), each face background is assigned a random preset image when available.
- A delete icon next to the preset dropdown clears the preset selection and restores the face's default background color.
- Background media can be selected via a file picker (images and common video formats).
- Image mode can be set to crop, repeat, or scale when a background image is selected.
- Clearing the image restores the face's default background color.
- Background media applies to the editor; the cube face uses the configured color.
- Settings panel text and controls are sized up, with spacing adjusted so controls do not overlap.

## Persistence and sync
- Text and config data live in `data/` (one file per face plus `config.json`).
- Autosave runs every 30 seconds and on exit; it always saves all face text files.
- Git sync is optional and disabled by default; when enabled, manual save (`Ctrl`+`S`) and autosave commit in `data/` and push to `origin main` if a remote is configured.
- If Git is unavailable or push fails, the app continues running and prints a console message.
