import * as THREE from './vendor/three.module.min.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import {
  SIZE,
  blockedEdgesForWalls,
  createInitialState,
  legalMoves,
  positionKey,
  samePosition,
  shortestPath,
  validateWallCandidate,
  wallDimensions
} from './cube-rules.mjs';

const STEP = 1.12;
const HALF = ((SIZE - 1) * STEP) / 2;
const BOUNDARY = (SIZE * STEP) / 2;
const WALL_THICKNESS = 0.13;
const PLAYER_COLORS = { A: 0xff4b3e, B: 0x5a7dff };
const PLAYER_NAMES = { A: 'Red', B: 'Blue' };
const PLAYER_GOALS = { A: 0, B: SIZE - 1 };
const AXES = {
  XY: { u: 'x', v: 'y', normal: 'z' },
  XZ: { u: 'x', v: 'z', normal: 'y' },
  YZ: { u: 'y', v: 'z', normal: 'x' }
};

const viewport = document.getElementById('viewport');
const toastElement = document.getElementById('toast');
const builderElement = document.getElementById('builder');
const modeButtons = [...document.querySelectorAll('[data-mode]')];
const planeButtons = [...document.querySelectorAll('[data-plane]')];

let state = createInitialState();
let mode = 'move';
let build = { plane: 'XY', layer: 4, rotated: false, u: 3, v: 1 };
let toastTimer = 0;
let hoveredObject = null;
let pointerStart = null;
let interactiveMeshes = [];

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x07090f, 0.025);

const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
camera.position.set(11.8, 9.6, 13.8);

let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
} catch (error) {
  viewport.innerHTML = '<p class="webgl-error">This concept needs WebGL to render the cube.</p>';
  throw error;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(viewport.clientWidth, viewport.clientHeight, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
viewport.appendChild(renderer.domElement);
window.dispatchEvent(new CustomEvent('wrongway:renderer-ready'));

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.065;
controls.enablePan = false;
controls.minDistance = 10;
controls.maxDistance = 27;
controls.target.set(0, 0, 0);

scene.add(new THREE.HemisphereLight(0x9eb8ff, 0x12131a, 1.5));

const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
keyLight.position.set(7, 11, 8);
scene.add(keyLight);

const rimLight = new THREE.DirectionalLight(0x587dff, 1.6);
rimLight.position.set(-8, -4, -9);
scene.add(rimLight);

const cubeGroup = new THREE.Group();
const dynamicGroup = new THREE.Group();
scene.add(cubeGroup, dynamicGroup);

buildStaticCube();
render();

function buildStaticCube() {
  const shellGeometry = new THREE.EdgesGeometry(
    new THREE.BoxGeometry(BOUNDARY * 2, BOUNDARY * 2, BOUNDARY * 2)
  );
  const shell = new THREE.LineSegments(
    shellGeometry,
    new THREE.LineBasicMaterial({ color: 0xaeb9d5, transparent: true, opacity: 0.6 })
  );
  cubeGroup.add(shell);

  const latticePoints = [];
  for (let a = 0; a < SIZE; a += 1) {
    for (let b = 0; b < SIZE; b += 1) {
      const av = coordinateToWorld(a);
      const bv = coordinateToWorld(b);
      latticePoints.push(
        new THREE.Vector3(-HALF, av, bv), new THREE.Vector3(HALF, av, bv),
        new THREE.Vector3(av, -HALF, bv), new THREE.Vector3(av, HALF, bv),
        new THREE.Vector3(av, bv, -HALF), new THREE.Vector3(av, bv, HALF)
      );
    }
  }

  const latticeGeometry = new THREE.BufferGeometry().setFromPoints(latticePoints);
  const lattice = new THREE.LineSegments(
    latticeGeometry,
    new THREE.LineBasicMaterial({
      color: 0x76829e,
      transparent: true,
      opacity: 0.105,
      depthWrite: false
    })
  );
  cubeGroup.add(lattice);

  const centerGeometry = new THREE.EdgesGeometry(
    new THREE.BoxGeometry(STEP * 0.92, STEP * 0.92, STEP * 0.92)
  );
  const centerCell = new THREE.LineSegments(
    centerGeometry,
    new THREE.LineBasicMaterial({ color: 0x62e5ff, transparent: true, opacity: 0.38 })
  );
  cubeGroup.add(centerCell);

  const faceGeometry = new THREE.PlaneGeometry(BOUNDARY * 2, BOUNDARY * 2);
  const redFace = new THREE.Mesh(
    faceGeometry,
    new THREE.MeshBasicMaterial({
      color: PLAYER_COLORS.A,
      transparent: true,
      opacity: 0.025,
      side: THREE.DoubleSide,
      depthWrite: false
    })
  );
  redFace.position.z = BOUNDARY;

  const blueFace = new THREE.Mesh(
    faceGeometry,
    new THREE.MeshBasicMaterial({
      color: PLAYER_COLORS.B,
      transparent: true,
      opacity: 0.03,
      side: THREE.DoubleSide,
      depthWrite: false
    })
  );
  blueFace.position.z = -BOUNDARY;
  cubeGroup.add(redFace, blueFace);

  const axisMaterialX = new THREE.LineBasicMaterial({ color: 0xff786d, transparent: true, opacity: 0.7 });
  const axisMaterialY = new THREE.LineBasicMaterial({ color: 0x6ef0a7, transparent: true, opacity: 0.7 });
  const axisMaterialZ = new THREE.LineBasicMaterial({ color: 0x79a0ff, transparent: true, opacity: 0.7 });
  cubeGroup.add(
    makeLine([new THREE.Vector3(-BOUNDARY, -BOUNDARY, -BOUNDARY), new THREE.Vector3(BOUNDARY, -BOUNDARY, -BOUNDARY)], axisMaterialX),
    makeLine([new THREE.Vector3(-BOUNDARY, -BOUNDARY, -BOUNDARY), new THREE.Vector3(-BOUNDARY, BOUNDARY, -BOUNDARY)], axisMaterialY),
    makeLine([new THREE.Vector3(-BOUNDARY, -BOUNDARY, -BOUNDARY), new THREE.Vector3(-BOUNDARY, -BOUNDARY, BOUNDARY)], axisMaterialZ)
  );
}

function makeLine(points, material) {
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), material);
}

function coordinateToWorld(value) {
  return (value - (SIZE - 1) / 2) * STEP;
}

function cellToWorld(cell) {
  return new THREE.Vector3(
    coordinateToWorld(cell.x),
    coordinateToWorld(cell.y),
    coordinateToWorld(cell.z)
  );
}

function gapToWorld(layer) {
  return (layer + 0.5 - (SIZE - 1) / 2) * STEP;
}

function playerPosition(player) {
  return state.players[player];
}

function otherPlayer(player = state.turn) {
  return player === 'A' ? 'B' : 'A';
}

function snapshot() {
  return {
    players: structuredClone(state.players),
    walls: structuredClone(state.walls),
    remaining: { ...state.remaining },
    turn: state.turn,
    turnCount: state.turnCount,
    winner: state.winner,
    feed: structuredClone(state.feed)
  };
}

function restore(saved) {
  const history = state.history;
  state = { ...structuredClone(saved), history };
}

function pushHistory() {
  state.history.push(snapshot());
}

function addFeed(player, text) {
  state.feed.unshift({
    player,
    text,
    index: String(state.turnCount).padStart(2, '0')
  });
  state.feed = state.feed.slice(0, 5);
}

function notify(message) {
  window.clearTimeout(toastTimer);
  toastElement.textContent = message;
  toastElement.classList.add('show');
  toastTimer = window.setTimeout(() => toastElement.classList.remove('show'), 2100);
}

function formatCoordinate(position, compact = false) {
  const separator = compact ? ' · ' : ' ';
  return `X${position.x + 1}${separator}Y${position.y + 1}${separator}Z${position.z + 1}`;
}

function attemptMove(destination) {
  if (state.winner) {
    notify('Start a new match to keep playing.');
    return;
  }

  const blocked = blockedEdgesForWalls(state.walls);
  const legal = legalMoves(playerPosition(state.turn), playerPosition(otherPlayer()), blocked);
  if (!legal.some((move) => samePosition(move, destination))) {
    notify('That position is not reachable this turn.');
    return;
  }

  pushHistory();
  const player = state.turn;
  state.players[player] = { ...destination };
  addFeed(player, `Moved to ${formatCoordinate(destination)}`);

  if (destination.z === PLAYER_GOALS[player]) {
    state.winner = player;
    notify(`${PLAYER_NAMES[player]} reached the opposite face.`);
  } else {
    state.turn = otherPlayer(player);
    state.turnCount += 1;
  }
  render();
}

function attemptWall(candidate) {
  if (state.winner) {
    notify('Start a new match to keep playing.');
    return;
  }
  if (state.remaining[state.turn] <= 0) {
    notify('No barricades remain for this player.');
    return;
  }

  const result = validateWallCandidate(candidate, state.walls, state.players.A, state.players.B);
  if (!result.valid) {
    notify(result.reason);
    return;
  }

  pushHistory();
  const player = state.turn;
  state.walls.push({ ...candidate, player });
  state.remaining[player] -= 1;
  addFeed(
    player,
    `${candidate.plane} plate · gap ${candidate.layer + 1} · ${wallDimensions(candidate).width}×${wallDimensions(candidate).height}`
  );
  state.turn = otherPlayer(player);
  state.turnCount += 1;
  mode = 'move';
  render();
}

function render() {
  renderDynamicScene();
  renderInterface();
}

function clearDynamicScene() {
  interactiveMeshes = [];
  hoveredObject = null;
  renderer.domElement.style.cursor = 'grab';
  for (const child of [...dynamicGroup.children]) {
    dynamicGroup.remove(child);
    child.traverse((object) => {
      if (object.geometry) object.geometry.dispose();
      if (object.material) {
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        materials.forEach((material) => material.dispose());
      }
    });
  }
}

function renderDynamicScene() {
  clearDynamicScene();
  const blocked = blockedEdgesForWalls(state.walls);
  addActiveSlice();
  addRoutes(blocked);
  state.walls.forEach(addWallMesh);
  addPawn('A');
  addPawn('B');

  if (state.winner) return;
  if (mode === 'move') addMoveMarkers(blocked);
  if (mode === 'wall' && state.remaining[state.turn] > 0) addBuildLayer();
}

function addActiveSlice() {
  const position = playerPosition(state.turn);
  const geometry = new THREE.BoxGeometry(BOUNDARY * 2, BOUNDARY * 2, 0.018);
  const material = new THREE.MeshBasicMaterial({
    color: 0x62e5ff,
    transparent: true,
    opacity: 0.035,
    side: THREE.DoubleSide,
    depthWrite: false
  });
  const slice = new THREE.Mesh(geometry, material);
  slice.position.z = coordinateToWorld(position.z);
  dynamicGroup.add(slice);
}

function addRoutes(blocked) {
  for (const player of ['A', 'B']) {
    const path = shortestPath(state.players[player], PLAYER_GOALS[player], blocked);
    if (!path) continue;
    const geometry = new THREE.BufferGeometry().setFromPoints(path.map(cellToWorld));
    const material = new THREE.LineBasicMaterial({
      color: PLAYER_COLORS[player],
      transparent: true,
      opacity: player === state.turn ? 0.62 : 0.25,
      depthTest: false
    });
    const line = new THREE.Line(geometry, material);
    line.renderOrder = 4;
    dynamicGroup.add(line);
  }
}

function addPawn(player) {
  const color = PLAYER_COLORS[player];
  const group = new THREE.Group();
  group.position.copy(cellToWorld(state.players[player]));

  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.31, 32, 22),
    new THREE.MeshPhysicalMaterial({
      color,
      emissive: color,
      emissiveIntensity: player === state.turn ? 1.15 : 0.72,
      roughness: 0.24,
      metalness: 0.18,
      clearcoat: 1,
      clearcoatRoughness: 0.1
    })
  );
  group.add(sphere);

  const halo = new THREE.Mesh(
    new THREE.SphereGeometry(0.43, 20, 14),
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: player === state.turn ? 0.11 : 0.045,
      side: THREE.BackSide,
      depthWrite: false
    })
  );
  group.add(halo);

  if (player === state.turn) {
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.47, 0.018, 8, 48),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.72, depthTest: false })
    );
    ring.rotation.x = Math.PI / 2;
    ring.userData.spin = true;
    group.add(ring);
  }
  dynamicGroup.add(group);
}

function wallVisual(wall) {
  const { width, height } = wallDimensions(wall);
  const axes = AXES[wall.plane];
  const center = { x: 0, y: 0, z: 0 };
  center[axes.u] = coordinateToWorld(wall.u + (width - 1) / 2);
  center[axes.v] = coordinateToWorld(wall.v + (height - 1) / 2);
  center[axes.normal] = gapToWorld(wall.layer);

  const longU = width * STEP - 0.08;
  const longV = height * STEP - 0.08;
  let dimensions;
  if (wall.plane === 'XY') dimensions = [longU, longV, WALL_THICKNESS];
  if (wall.plane === 'XZ') dimensions = [longU, WALL_THICKNESS, longV];
  if (wall.plane === 'YZ') dimensions = [WALL_THICKNESS, longU, longV];
  return { center, dimensions };
}

function createWallObject(wall, style = 'placed', valid = true) {
  const { center, dimensions } = wallVisual(wall);
  let material;
  if (style === 'ghost') {
    material = new THREE.MeshPhysicalMaterial({
      color: valid ? 0x62e5ff : 0xff4b3e,
      emissive: valid ? 0x226779 : 0x6d1510,
      emissiveIntensity: 0.6,
      transparent: true,
      opacity: 0.38,
      roughness: 0.3,
      metalness: 0.18,
      side: THREE.DoubleSide,
      depthWrite: false
    });
  } else {
    const color = wall.player === 'B' ? 0x5a7dff : 0xff4b3e;
    material = new THREE.MeshPhysicalMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.26,
      transparent: true,
      opacity: 0.75,
      roughness: 0.28,
      metalness: 0.34,
      clearcoat: 0.7,
      side: THREE.DoubleSide
    });
  }

  const mesh = new THREE.Mesh(new THREE.BoxGeometry(...dimensions), material);
  mesh.position.set(center.x, center.y, center.z);
  mesh.renderOrder = style === 'ghost' ? 5 : 2;
  return mesh;
}

function addWallMesh(wall) {
  const object = createWallObject(wall);
  dynamicGroup.add(object);

  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(object.geometry),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 })
  );
  edges.position.copy(object.position);
  dynamicGroup.add(edges);
}

function addMoveMarkers(blocked) {
  const legal = legalMoves(playerPosition(state.turn), playerPosition(otherPlayer()), blocked);
  for (const destination of legal) {
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.16, 16, 12),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.7,
        depthTest: false
      })
    );
    marker.position.copy(cellToWorld(destination));
    marker.userData = { action: 'move', destination, pulse: true };
    marker.renderOrder = 8;
    dynamicGroup.add(marker);
    interactiveMeshes.push(marker);

    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.24, 0.29, 24),
      new THREE.MeshBasicMaterial({
        color: PLAYER_COLORS[state.turn],
        transparent: true,
        opacity: 0.65,
        side: THREE.DoubleSide,
        depthTest: false
      })
    );
    ring.position.copy(marker.position);
    ring.lookAt(camera.position);
    ring.userData.billboard = true;
    ring.renderOrder = 7;
    dynamicGroup.add(ring);
  }
}

function clampBuildAnchor() {
  const { width, height } = wallDimensions(build);
  build.u = Math.max(0, Math.min(SIZE - width, build.u));
  build.v = Math.max(0, Math.min(SIZE - height, build.v));
}

function addBuildLayer() {
  clampBuildAnchor();
  addConstructionPlane();

  const { width, height } = wallDimensions(build);
  for (let u = 0; u <= SIZE - width; u += 1) {
    for (let v = 0; v <= SIZE - height; v += 1) {
      const candidate = { ...build, u, v };
      const validation = validateWallCandidate(candidate, state.walls, state.players.A, state.players.B);
      const marker = makeAnchorMarker(candidate, validation.valid);
      marker.userData = {
        action: 'wall',
        candidate,
        valid: validation.valid,
        reason: validation.reason,
        pulse: validation.valid
      };
      dynamicGroup.add(marker);
      interactiveMeshes.push(marker);
    }
  }

  const selected = { ...build };
  const selectedValidation = validateWallCandidate(selected, state.walls, state.players.A, state.players.B);
  const ghost = createWallObject(selected, 'ghost', selectedValidation.valid);
  ghost.userData.ghost = true;
  dynamicGroup.add(ghost);
}

function addConstructionPlane() {
  const span = BOUNDARY * 2;
  let dimensions;
  if (build.plane === 'XY') dimensions = [span, span, 0.025];
  if (build.plane === 'XZ') dimensions = [span, 0.025, span];
  if (build.plane === 'YZ') dimensions = [0.025, span, span];

  const plane = new THREE.Mesh(
    new THREE.BoxGeometry(...dimensions),
    new THREE.MeshBasicMaterial({
      color: 0x62e5ff,
      transparent: true,
      opacity: 0.045,
      side: THREE.DoubleSide,
      depthWrite: false
    })
  );
  plane.position[AXES[build.plane].normal] = gapToWorld(build.layer);
  dynamicGroup.add(plane);
}

function makeAnchorMarker(candidate, valid) {
  const axes = AXES[candidate.plane];
  const point = { x: 0, y: 0, z: 0 };
  point[axes.u] = coordinateToWorld(candidate.u);
  point[axes.v] = coordinateToWorld(candidate.v);
  point[axes.normal] = gapToWorld(candidate.layer);

  const selected = candidate.u === build.u && candidate.v === build.v;
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(selected ? 0.13 : 0.095, 12, 8),
    new THREE.MeshBasicMaterial({
      color: valid ? 0x62e5ff : 0xff4b3e,
      transparent: true,
      opacity: selected ? 0.95 : valid ? 0.46 : 0.18,
      depthTest: false
    })
  );
  marker.position.set(point.x, point.y, point.z);
  marker.renderOrder = 9;
  return marker;
}

function renderInterface() {
  const active = state.turn;
  const activePosition = state.players[active];
  const blocked = blockedEdgesForWalls(state.walls);
  const redPath = shortestPath(state.players.A, PLAYER_GOALS.A, blocked);
  const bluePath = shortestPath(state.players.B, PLAYER_GOALS.B, blocked);

  document.getElementById('wallsA').textContent = state.remaining.A;
  document.getElementById('wallsB').textContent = state.remaining.B;
  document.getElementById('turnCount').textContent = String(state.turnCount).padStart(2, '0');
  document.getElementById('cardA').classList.toggle('active', active === 'A');
  document.getElementById('cardB').classList.toggle('active', active === 'B');
  document.getElementById('coordHud').textContent = formatCoordinate(activePosition, true);
  document.getElementById('coordReadout').textContent = formatCoordinate(activePosition);
  document.getElementById('activePlayerLabel').textContent = `${PLAYER_NAMES[active].toUpperCase()} / PLAYER ${active === 'A' ? '01' : '02'}`;
  document.getElementById('activeOrb').classList.toggle('blue', active === 'B');
  document.getElementById('pathA').textContent = redPath ? redPath.length - 1 : '—';
  document.getElementById('pathB').textContent = bluePath ? bluePath.length - 1 : '—';
  document.getElementById('routeInsight').textContent = routeMessage(redPath, bluePath);

  const turnDot = document.getElementById('turnDot');
  turnDot.style.background = active === 'A' ? '#ff4b3e' : '#5a7dff';
  turnDot.style.boxShadow = `0 0 13px ${active === 'A' ? '#ff4b3e' : '#5a7dff'}`;
  document.getElementById('turnLabel').textContent = state.winner
    ? `${PLAYER_NAMES[state.winner].toUpperCase()} WINS`
    : `${PLAYER_NAMES[active].toUpperCase()} TO ${mode === 'move' ? 'MOVE' : 'BUILD'}`;
  document.getElementById('modeHint').textContent = state.winner
    ? 'The opposite face has been reached.'
    : mode === 'move'
      ? 'Choose a glowing destination.'
      : 'Choose an anchor point in the active plane.';

  modeButtons.forEach((button) => button.classList.toggle('active', button.dataset.mode === mode));
  builderElement.classList.toggle('enabled', mode === 'wall');
  planeButtons.forEach((button) => button.classList.toggle('active', button.dataset.plane === build.plane));
  document.getElementById('footprintLabel').textContent = build.rotated ? '6 × 3' : '3 × 6';
  document.getElementById('layerAxis').textContent = `${AXES[build.plane].normal.toUpperCase()} GAP`;
  document.getElementById('layerLabel').textContent = `${build.layer + 1} / 8`;
  document.getElementById('layerDown').disabled = build.layer === 0;
  document.getElementById('layerUp').disabled = build.layer === SIZE - 2;
  document.getElementById('undoButton').disabled = state.history.length === 0;

  renderDepthStack();
  renderFeed();
}

function routeMessage(redPath, bluePath) {
  if (state.winner) return `${PLAYER_NAMES[state.winner]} completed the crossing.`;
  const redDistance = redPath ? redPath.length - 1 : Infinity;
  const blueDistance = bluePath ? bluePath.length - 1 : Infinity;
  if (redDistance === blueDistance) return 'The routes are balanced. The next plate can change the geometry.';
  const leader = redDistance < blueDistance ? 'Red' : 'Blue';
  return `${leader} currently owns the shorter route through the volume.`;
}

function renderDepthStack() {
  const activeZ = state.players[state.turn].z;
  const rows = [];
  for (let z = SIZE - 1; z >= 0; z -= 1) {
    const pieces = [];
    if (state.players.A.z === z) pieces.push('<i class="depth-piece red"></i>');
    if (state.players.B.z === z) pieces.push('<i class="depth-piece blue"></i>');
    rows.push(
      `<div class="depth-layer${z === activeZ ? ' active' : ''}">`
      + `<span class="depth-number">Z${z + 1}</span>`
      + `<span class="depth-pieces">${pieces.join('')}</span>`
      + '</div>'
    );
  }
  document.getElementById('depthStack').innerHTML = rows.join('');
}

function renderFeed() {
  document.getElementById('actionLog').innerHTML = state.feed.map((item) => (
    `<li class="${item.player === 'B' ? 'blue' : ''}">`
    + `<b>${item.index}</b><i></i><span>${item.text}</span></li>`
  )).join('');
}

function setMode(nextMode) {
  mode = nextMode;
  render();
}

function setCameraView(view) {
  const activeTarget = cellToWorld(state.players[state.turn]);
  let target = new THREE.Vector3(0, 0, 0);
  let position;

  if (view === 'front') position = new THREE.Vector3(0.1, 1.2, 18);
  if (view === 'side') position = new THREE.Vector3(18, 1.2, 0.1);
  if (view === 'top') position = new THREE.Vector3(0.1, 18, 0.1);
  if (view === 'focus') {
    target = activeTarget;
    position = activeTarget.clone().add(new THREE.Vector3(7, 6, 8));
  }

  if (!position) return;
  camera.position.copy(position);
  controls.target.copy(target);
  controls.update();
}

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function raycastAt(clientX, clientY) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  return raycaster.intersectObjects(interactiveMeshes, false)[0]?.object || null;
}

renderer.domElement.addEventListener('pointerdown', (event) => {
  pointerStart = { x: event.clientX, y: event.clientY };
});

renderer.domElement.addEventListener('pointermove', (event) => {
  if (pointerStart && Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 5) return;
  const hit = raycastAt(event.clientX, event.clientY);
  renderer.domElement.style.cursor = hit ? 'pointer' : 'grab';

  if (hit === hoveredObject) return;
  hoveredObject = hit;
  if (hit?.userData.action === 'wall') {
    build.u = hit.userData.candidate.u;
    build.v = hit.userData.candidate.v;
    render();
  }
});

renderer.domElement.addEventListener('pointerup', (event) => {
  if (!pointerStart) return;
  const distance = Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y);
  pointerStart = null;
  if (distance > 5) return;

  const hit = raycastAt(event.clientX, event.clientY);
  if (!hit) return;
  if (hit.userData.action === 'move') attemptMove(hit.userData.destination);
  if (hit.userData.action === 'wall') {
    if (!hit.userData.valid) notify(hit.userData.reason);
    else attemptWall(hit.userData.candidate);
  }
});

renderer.domElement.addEventListener('pointerleave', () => {
  pointerStart = null;
  renderer.domElement.style.cursor = 'grab';
});

modeButtons.forEach((button) => {
  button.addEventListener('click', () => setMode(button.dataset.mode));
});

planeButtons.forEach((button) => {
  button.addEventListener('click', () => {
    build.plane = button.dataset.plane;
    clampBuildAnchor();
    render();
  });
});

document.getElementById('rotateWall').addEventListener('click', () => {
  build.rotated = !build.rotated;
  clampBuildAnchor();
  render();
});

document.getElementById('layerDown').addEventListener('click', () => {
  build.layer = Math.max(0, build.layer - 1);
  render();
});

document.getElementById('layerUp').addEventListener('click', () => {
  build.layer = Math.min(SIZE - 2, build.layer + 1);
  render();
});

document.getElementById('undoButton').addEventListener('click', () => {
  const previous = state.history.pop();
  if (!previous) return;
  restore(previous);
  mode = 'move';
  render();
  notify('Last turn restored.');
});

document.getElementById('resetButton').addEventListener('click', () => {
  state = createInitialState();
  mode = 'move';
  build = { plane: 'XY', layer: 4, rotated: false, u: 3, v: 1 };
  controls.target.set(0, 0, 0);
  setCameraView('front');
  render();
  notify('New cube initialized.');
});

document.querySelectorAll('[data-view]').forEach((button) => {
  button.addEventListener('click', () => setCameraView(button.dataset.view));
});

window.addEventListener('keydown', (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
  if (event.key.toLowerCase() === 'm') setMode('move');
  if (event.key.toLowerCase() === 'b') setMode('wall');
  if (event.key.toLowerCase() === 'r' && mode === 'wall') {
    build.rotated = !build.rotated;
    clampBuildAnchor();
    render();
  }
  if (event.key === '[' && mode === 'wall') {
    build.layer = Math.max(0, build.layer - 1);
    render();
  }
  if (event.key === ']' && mode === 'wall') {
    build.layer = Math.min(SIZE - 2, build.layer + 1);
    render();
  }
});

const resizeObserver = new ResizeObserver(() => {
  const width = Math.max(1, viewport.clientWidth);
  const height = Math.max(1, viewport.clientHeight);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
});
resizeObserver.observe(viewport);

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const elapsed = clock.getElapsedTime();
  controls.update();

  dynamicGroup.traverse((object) => {
    if (object.userData.pulse) {
      const scale = 1 + Math.sin(elapsed * 3.4 + object.position.x) * 0.13;
      object.scale.setScalar(scale);
    }
    if (object.userData.billboard) object.lookAt(camera.position);
    if (object.userData.spin) object.rotation.z = elapsed * 0.7;
  });

  renderer.render(scene, camera);
});

window.addEventListener('beforeunload', () => {
  resizeObserver.disconnect();
  renderer.setAnimationLoop(null);
  renderer.dispose();
});
