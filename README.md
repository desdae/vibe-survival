# Surv

`Surv` is a browser-based 3D survival sandbox with procedural terrain, gathering, crafting, combat, and building.

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
- `Left Click`: use active tool / place build piece
- `E`: pick up focused item / interact with firepit or smelter
- `C`: open/close crafting panel
- `I`: open/close inventory grid
- `X`: consume the best available food
- `R`: rotate placement piece (`Triangle Wall` flips orientation)
- `1..9` (or numpad): activate actionbar slot

## Core Features

- Procedural mountainous terrain with chunk streaming.
- UI environment toggles: rain, mist, wind, volumetric clouds.
- Camera mode toggle: `Free-Fly` / `Walk`.
- Inventory UI as an `8x5` grid with drag/drop reordering.
- Drag food/tools from inventory to actionbar.
- Bottom-center `1..9` actionbar with drag/drop slot rearranging.
- Item stacks up to `999`.
- Hunger + stamina loop with buffs from better food.
- Food spoilage with live timers and weighted freshness blending.
- Animated skeleton warriors spawn near the player start area.
- Live FPS and triangle counters.

## Gathering and Resources

- Gather with `E`: branches, stones, logs, berries, red mushrooms, yellow mushrooms.
- Ore nodes spawn as copper/iron deposits.
- Ore deposits are solid meshes and hold multiple ore units.
- Ore is mined one unit at a time.
- Loot drops use physics, settle on slopes, and can be picked up with `E`.
- Creature drops:
- Pig: `1-3 Raw Meat`, `1 Leather`
- Skeleton: `2-4 Bones`

## Crafting Progression

- `Stone Axe`: `1 Branch + 2 Stones`
- `Stone Club`: `1 Log + 1 Stone`
- `Leather Strips`: `1 Leather -> 2 Leather Strips`
- `Bone Axe`: `Stone Axe + 3 Bones + 1 Leather Strips`
- `Bone Club`: `Stone Club + 3 Bones + 1 Leather Strips`
- `Bone Pickaxe`: `3 Bones + 1 Leather Strips + 1 Stone`
- `Copper Pickaxe`: `Bone Pickaxe + 2 Copper Ingots + 1 Leather Strips`
- `Metal Axe`: `Bone Axe + 2 Iron Ingots + 1 Copper Ingot + 1 Leather Strips`
- `Metal Club`: `Bone Club + 1 Iron Ingot + 1 Copper Ingot + 1 Leather Strips`
- `Firepit`: `5 Logs + 2 Stones`
- `Smelter`: `8 Stones + 4 Logs`
- `Wooden Wall`: `2 Logs`
- `Wooden Floor`: `2 Logs`
- `Wooden Roof`: `2 Logs`
- `Triangle Wall`: `1 Log`

## Tool Roles and Durability

- Crafted tools start at `100` durability.
- Each use costs `1` durability.
- Upgraded tools have higher max durability based on tool definition.
- Axes are woodcutting tools (tree harvesting).
- Clubs are melee weapons (no chopping, no ore mining).
- Pickaxes are ore-mining tools:
- `Bone Pickaxe` can mine copper (`tier 1` ore access).
- `Copper Pickaxe` can mine copper and iron (`tier 2` ore access).
- Pickaxe viewmodel head tint reflects tier:
- Bone pickaxe head is white.
- Copper pickaxe head is orange.

## Cooking and Smelting

- Look at a placed firepit or smelter and press `E` to open/close station cooking UI.
- Firepit recipes:
- `Cooked Meat`: `1 Raw Meat`
- `Grilled Red Mushroom`: `1 Red Mushroom`
- `Grilled Yellow Mushroom`: `1 Yellow Mushroom`
- `Berry-Glazed Cut`: `1 Raw Meat + 2 Berries`
- `Forest Skewer`: `1 Raw Meat + 1 Red Mushroom + 1 Yellow Mushroom`
- `Hearty Stew`: `2 Raw Meat + 1 Red Mushroom + 1 Yellow Mushroom + 2 Berries`
- `Charcoal`: `1 Log`
- Smelter recipes:
- `Copper Ingot`: `2 Copper Ore + 1 Charcoal`
- `Iron Ingot`: `2 Iron Ore + 1 Charcoal`

## Combat and Creatures

- Club can hit pigs and skeletons in melee range.
- Pig stats: `10 HP`, club base hit: `3 damage`.
- Pigs show floating healthbars when damaged.
- Skeletons show floating healthbars when damaged.
- Pigs and rabbits use per-leg ground probing for better terrain contact.

## Building

- Crafting build pieces enters placement mode.
- Place with `Left Click` and rotate with `R`.
- Snap support exists between walls, floors, roofs, and triangle walls.

## Debug Helpers

- Left-panel debug button: `+100 All Resources`.
- Adds `100` of each inventory resource item for rapid testing.

## Icon Credits

- Item/tool icons are from the **Game Icons** project.
- Source: `https://game-icons.net/`
- License: `CC BY 3.0`
- License text: `https://github.com/game-icons/icons/blob/master/CC-BY-3.0.txt`
- Icons are stored locally in `assets/icons/`.

## Tool Model Credits

- `Stone Axe` model source: `https://poly.pizza/m/lmO4Yq56e5`
- `Stone Club` model source (low-poly sledge asset): `https://poly.pizza/m/yfAhQ8PECT`
- `Pickaxe` model source (used for Bone/Copper Pickaxe): `https://poly.pizza/m/cJp88qPPLc`
- `Skeleton Warrior` model source: `https://poly.pizza/m/wODZYCgX5Z`
- Asset file host: `https://static.poly.pizza/`
- License for listed models: `Public Domain (CC0)`
- Model files are stored locally in `assets/models/`.
