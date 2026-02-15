# Surv

`Surv` is a browser-based 3D survival sandbox with procedural terrain, gathering, crafting, building, and animal interactions.

## Run

1. Start a local web server in the project folder.
2. Open the served URL in your browser.
3. Click the scene to lock the mouse and enable controls.

Example server commands:

```bash
python -m http.server 8080
```

```bash
npx serve .
```

## Controls

- `Mouse`: look around
- `W A S D`: move
- `Space`: move up (free-fly) / jump (walk mode)
- `Left Shift`: move down (free-fly)
- `Left Ctrl`: sprint
- `E`: pick up focused item
- `C`: open/close crafting panel
- `Left Click`: use active tool / place building piece
- `R`: rotate placement piece (`Triangle Wall` flips orientation)
- `1..9` (or numpad): activate actionbar slot

## Core Features

- Procedural mountainous terrain with chunk streaming.
- Weather/environment toggles in UI: rain, mist, wind.
- Volumetric cloud system with slow drifting clouds and UI toggle.
- Camera mode toggle in UI: `Free-Fly` / `Walk`.
- Inventory UI with resources, food, materials, mushrooms, and tools.
- Bottom-center 9-slot actionbar with drag-and-drop reordering.
- UI sliders for camera speed and view distance.
- Live FPS and triangle counters.

## Gathering and Resources

- Gather with `E`: branches, stones, logs, berries, red mushrooms, yellow mushrooms.
- Loot drops: `Raw Meat`, `Leather`

## Crafting

- `Stone Axe`: `1 Branch + 2 Stones`
- `Stone Club`: `1 Log + 1 Stone`
- `Firepit`: `5 Logs + 2 Stones`
- `Wooden Wall`: `2 Logs`
- `Wooden Floor`: `2 Logs`
- `Wooden Roof`: `2 Logs`
- `Triangle Wall`: `1 Log`

## Tools and Durability

- Crafted tools start at `100` durability.
- Each use costs `1` durability.
- Tools break automatically at `0` durability.
- `Stone Axe` is for chopping trees.
- Axe viewmodel uses a sharp blade facing away from the player.
- `Stone Club` cannot chop trees.

## Pig Combat and Loot

- Club can hit pigs in melee range.
- Pig stats: `10 HP`.
- Club damage: `3` per hit.
- Damaged pigs show floating health bars.
- Pig death drops `1-3 Raw Meat` and `1 Leather`.
- Drops scatter, spawn above terrain, fall with gravity, and can be picked up with `E`.
- Meat and leather drop models are enlarged (~50%) for visibility.

## Animal Grounding and Terrain Contact

- Pigs and rabbits use per-leg vertical ground probing.
- Animal body orientation aligns to local slope (not forced flat).
- Terrain height sampling uses chunk triangle interpolation for accurate leg contact.
- Drop support sampling keeps loot above terrain in uneven areas.

## Building

- Crafting build pieces enters placement mode.
- Place with `Left Click`.
- Rotate with `R`.
- Snap support exists between walls, floors, roofs, and triangle walls.
- Placeable structures: `Firepit`, `Wooden Wall`, `Wooden Floor`, `Wooden Roof`, `Triangle Wall`.
