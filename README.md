# NotesA3 (Notes Cubed)

A floating 3D cube where each face is a full text editor. Rotate to a face, type like in Notepad++, save, and sync the six notes with Git. The window is borderless and always-on-top so it feels like a cube hovering above everything else.

## What it is
- Six faces, six editable documents (plain text / Markdown friendly).
- Rotate the cube with the mouse or `Alt`+`Arrow`, snap to a face, and immediately type.
- Per-face backgrounds (color or image) and per-face scrolling.
- Autosave plus `Ctrl+S` to write to disk and create Git commits; optional push to a configured GitHub remote.

## Controls (MVP)
- **Launch**: `python notes_cubed.py`
- **Rotate**: Left-click and drag outside the editor area (release snaps to the nearest face and returns to edit mode).
- **Keyboard rotate**: `Alt`+`Left/Right` cycles faces; `Alt`+`Up` goes to the top face; `Alt`+`Down` goes to the bottom face. After rotation, you are in edit mode.
- **Edit**: Click inside the editor area to place the caret; type normally.
- **Scroll**: Mouse wheel inside the editor scrolls the current face's content.
- **Screenshot**: `F12` saves a PNG to `data/screenshots/` (useful for mid-rotation visual verification).
- **Save + sync**: `Ctrl`+`S` saves all six face files and commits them. Pushes if you configure a remote.
- **Exit**: Close the window (autosaves first).

## Install
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```
python notes_cubed.py
```
The cube opens centered and always-on-top with no window chrome.

## Data + Git
- Data lives in `data/` alongside the script: one file per face (`front.txt`, `right.txt`, `back.txt`, `left.txt`, `top.txt`, `bottom.txt`) plus `config.json`.
- Autosave and `Ctrl`+`S` always write the text files.
- Git sync is optional and disabled by default. Set `git_sync: true` in `data/config.json` to enable it; when enabled, saves run `git init` in `data/`, stage, and commit changes. If a `remote` URL is set, it attempts to push (`origin main`). If Git is missing or push fails, the app keeps running and prints a console message.

## Config
Edit `data/config.json` to tweak backgrounds and Git remote. Example:
```json
{
  "last_face": 0,
  "remote": "",
  "backgrounds": {
    "front":  {"type": "color", "value": "#1e1e2e"},
    "right":  {"type": "color", "value": "#10314a"},
    "back":   {"type": "color", "value": "#2d1b3f"},
    "left":   {"type": "color", "value": "#3c2f2f"},
    "top":    {"type": "color", "value": "#0f3d3e"},
    "bottom": {"type": "color", "value": "#3a2a1a"}
  }
}
```
- Backgrounds support `type: "color"` with hex `value` (e.g., `#1e1e2e`). Image paths are accepted (`type: "image"`, `value: "path/to/file.png"`) for the editor background; the cube face still uses a color.

## How it behaves (MVP scope)
- **Always-on-top, borderless**: Implemented with a transparent-hitbox window; visually only the cube and editor show.
- **Rotate vs Edit modes**: Drag outside the editor to rotate (Rotate mode). Release snaps to the nearest face; the editor returns when the snap animation finishes.
- **Keyboard rotation**: `Alt`+`Arrow` animates the cube to the target face; the editor returns when the animation finishes.
- **Scroll per face**: Each face has its own scroll position.
- **Persistence**: Autosaves on close; `Ctrl`+`S` saves + commits.
- **GitHub sync**: Works if a remote is configured; otherwise commits stay local.

## Specifications
User-level behavior is documented in `specifications.md`. Whenever you add a new requirement or discover a new behavior detail, update `specifications.md` to keep it current.

## Testing
Basic headless check (no window) to verify file IO and git plumbing:
```
python notes_cubed.py --headless-test
```
This ensures the data folder, config, and git commands work without launching the UI.

## Roadmap (next steps you might want)
- Texture text directly on cube faces.
- Tray icon toggle / hide-on-Esc.
- Conflict visualization when pulling remote changes.
- Inline background picker and per-face titles.
