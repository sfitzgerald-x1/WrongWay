# WrongWay

## 9×9×9 cube concept

The interactive concept keeps the original client intact and adds a standalone
volumetric duel at [`/concept-3d.html`](./concept-3d.html). Players begin as
spheres at the centers of opposite faces and move through a 9×9×9 coordinate
lattice. The cube can be rotated throughout play.

Each player receives 10 parametric barricades. A barricade controls exactly
2/9 of a construction plane: its 3×6 footprint blocks 18 of the plane's 81
crossings and can rotate in-place to 6×3. Barricades can be built in XY, XZ, or
YZ planes, provided both players retain a path to their goal face.

Run it from the repository root with:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open <http://localhost:8000/concept-3d.html>.

Run the focused rule tests with:

```sh
node --test tests/cube-rules.test.mjs
```
