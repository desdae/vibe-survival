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
- `E`: pick up focused item / interact with firepit
- `C`: open/close crafting panel
- `I`: open/close inventory grid
- `X`: eat the best available food from inventory
- `Left Click`: use active tool / place building piece
- `R`: rotate placement piece (`Triangle Wall` flips orientation)
- `1..9` (or numpad): activate actionbar slot

## Core Features

- Procedural mountainous terrain with chunk streaming.
- Weather/environment toggles in UI: rain, mist, wind.
- Volumetric cloud system with slow drifting clouds and UI toggle.
- Camera mode toggle in UI: `Free-Fly` / `Walk`.
- Inventory UI as an `8x5` slot grid with names, icons, and tooltips.
- Inventory supports drag/drop rearranging and drag from inventory to actionbar.
- Actionbar supports tools and food items.
- Hunger + stamina survival loop with food-driven recovery and temporary buffs.
- Firepit cooking UI with single-item and combo food recipes.
- Food spoilage system with live countdown timers and weighted freshness blending.
- Bottom-center 9-slot actionbar with drag-and-drop reordering.
- UI sliders for camera speed and view distance.
- Live FPS and triangle counters.
- Animated undead skeleton warriors spawn near player start and roam using idle/walk/attack animations.

## Gathering and Resources

- Gather with `E`: branches, stones, logs, berries, red mushrooms, yellow mushrooms.
- Loot drops: `Raw Meat`, `Leather`

## Survival and Food

- Hunger drains over time and drains faster while sprinting.
- Stamina capacity and sprint effectiveness are influenced by current hunger.
- Better cooked/combo foods restore more hunger/stamina and can grant temporary sprint/regen buffs.
- Press `X` to consume the best available edible item quickly.
- Food items spoil over time.
- Spoilage is tuned to be relatively slow and degrades stacks gradually.
- Spoilage timers in inventory update live.
- Picking up fresh perishable food updates stack spoilage using weighted averaging, with a small freshness bonus.

## Crafting

- `Stone Axe`: `1 Branch + 2 Stones`
- `Stone Club`: `1 Log + 1 Stone`
- `Firepit`: `5 Logs + 2 Stones`
- `Wooden Wall`: `2 Logs`
- `Wooden Floor`: `2 Logs`
- `Wooden Roof`: `2 Logs`
- `Triangle Wall`: `1 Log`

## Firepit Cooking

- Look at a placed `Firepit` and press `E` to open/close the cooking panel.
- Cooking requires ingredient availability in your inventory.
- Cooking recipes:
- `Cooked Meat`: `1 Raw Meat`
- `Grilled Red Mushroom`: `1 Red Mushroom`
- `Grilled Yellow Mushroom`: `1 Yellow Mushroom`
- `Berry-Glazed Cut`: `1 Raw Meat + 2 Berries`
- `Forest Skewer`: `1 Raw Meat + 1 Red Mushroom + 1 Yellow Mushroom`
- `Hearty Stew`: `2 Raw Meat + 1 Red Mushroom + 1 Yellow Mushroom + 2 Berries`

## Tools and Durability

- Crafted tools start at `100` durability.
- Each use costs `1` durability.
- Tools break automatically at `0` durability.
- `Stone Axe` is for chopping trees.
- `Stone Club` cannot chop trees.
- First-person tool viewmodels use low-poly GLTF assets loaded from `assets/models/`.

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

## Icon Credits

- Item/tool icons are from the **Game Icons** project.
- Source: `https://game-icons.net/`
- License: `CC BY 3.0`
- License text: `https://github.com/game-icons/icons/blob/master/CC-BY-3.0.txt`
- Icons are stored locally in `assets/icons/` and used by both inventory and actionbar UI.

## Tool Model Credits

- `Stone Axe` model source: `https://poly.pizza/m/lmO4Yq56e5`
- `Stone Club` model source (using low-poly sledge hammer asset): `https://poly.pizza/m/yfAhQ8PECT`
- `Skeleton Warrior` model source: `https://poly.pizza/m/wODZYCgX5Z`
- Asset file host: `https://static.poly.pizza/`
- License for all listed models: `Public Domain (CC0)`
- Model files are stored locally in `assets/models/`.
