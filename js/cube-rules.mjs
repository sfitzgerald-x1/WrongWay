export const SIZE = 9;
export const BARRICADES_PER_PLAYER = 10;
export const BARRICADE_AREA = 18;

export const DIRECTIONS = Object.freeze([
  Object.freeze({ x: 1, y: 0, z: 0 }),
  Object.freeze({ x: -1, y: 0, z: 0 }),
  Object.freeze({ x: 0, y: 1, z: 0 }),
  Object.freeze({ x: 0, y: -1, z: 0 }),
  Object.freeze({ x: 0, y: 0, z: 1 }),
  Object.freeze({ x: 0, y: 0, z: -1 })
]);

const PLANE_AXES = Object.freeze({
  XY: Object.freeze({ u: 'x', v: 'y', normal: 'z' }),
  XZ: Object.freeze({ u: 'x', v: 'z', normal: 'y' }),
  YZ: Object.freeze({ u: 'y', v: 'z', normal: 'x' })
});

export function samePosition(a, b) {
  return a.x === b.x && a.y === b.y && a.z === b.z;
}

export function positionKey(position) {
  return `${position.x},${position.y},${position.z}`;
}

export function inBounds(position) {
  return Number.isInteger(position.x)
    && Number.isInteger(position.y)
    && Number.isInteger(position.z)
    && position.x >= 0 && position.x < SIZE
    && position.y >= 0 && position.y < SIZE
    && position.z >= 0 && position.z < SIZE;
}

export function edgeKey(a, b) {
  const aKey = positionKey(a);
  const bKey = positionKey(b);
  return aKey < bKey ? `${aKey}|${bKey}` : `${bKey}|${aKey}`;
}

export function wallDimensions(wall) {
  return wall.rotated ? { width: 6, height: 3 } : { width: 3, height: 6 };
}

export function wallEdgeKeys(wall) {
  const axes = PLANE_AXES[wall.plane];
  if (!axes) return [];

  const { width, height } = wallDimensions(wall);
  const edges = [];

  for (let uOffset = 0; uOffset < width; uOffset += 1) {
    for (let vOffset = 0; vOffset < height; vOffset += 1) {
      const from = { x: 0, y: 0, z: 0 };
      from[axes.u] = wall.u + uOffset;
      from[axes.v] = wall.v + vOffset;
      from[axes.normal] = wall.layer;
      const to = { ...from, [axes.normal]: wall.layer + 1 };
      edges.push(edgeKey(from, to));
    }
  }

  return edges;
}

export function blockedEdgesForWalls(walls) {
  const blocked = new Set();
  for (const wall of walls) {
    for (const key of wallEdgeKeys(wall)) blocked.add(key);
  }
  return blocked;
}

export function wallGeometryIsValid(wall) {
  if (!wall || !PLANE_AXES[wall.plane]) return false;
  if (!Number.isInteger(wall.layer) || wall.layer < 0 || wall.layer >= SIZE - 1) return false;
  if (!Number.isInteger(wall.u) || !Number.isInteger(wall.v)) return false;

  const { width, height } = wallDimensions(wall);
  return wall.u >= 0
    && wall.v >= 0
    && wall.u + width <= SIZE
    && wall.v + height <= SIZE
    && width * height === BARRICADE_AREA;
}

export function shortestPath(start, goalZ, blockedEdges) {
  if (!inBounds(start) || (goalZ !== 0 && goalZ !== SIZE - 1)) return null;

  const startKey = positionKey(start);
  const queue = [{ ...start }];
  const parent = new Map([[startKey, null]]);
  let head = 0;
  let goal = null;

  while (head < queue.length) {
    const current = queue[head];
    head += 1;

    if (current.z === goalZ) {
      goal = current;
      break;
    }

    for (const direction of DIRECTIONS) {
      const next = {
        x: current.x + direction.x,
        y: current.y + direction.y,
        z: current.z + direction.z
      };
      const nextKey = positionKey(next);
      if (!inBounds(next) || parent.has(nextKey) || blockedEdges.has(edgeKey(current, next))) continue;
      parent.set(nextKey, positionKey(current));
      queue.push(next);
    }
  }

  if (!goal) return null;

  const path = [];
  let cursor = positionKey(goal);
  while (cursor !== null) {
    const [x, y, z] = cursor.split(',').map(Number);
    path.push({ x, y, z });
    cursor = parent.get(cursor);
  }
  return path.reverse();
}

export function legalMoves(position, opponent, blockedEdges) {
  const moves = [];

  for (const direction of DIRECTIONS) {
    const adjacent = {
      x: position.x + direction.x,
      y: position.y + direction.y,
      z: position.z + direction.z
    };
    if (!inBounds(adjacent) || blockedEdges.has(edgeKey(position, adjacent))) continue;

    if (samePosition(adjacent, opponent)) {
      for (const exitDirection of DIRECTIONS) {
        const landing = {
          x: opponent.x + exitDirection.x,
          y: opponent.y + exitDirection.y,
          z: opponent.z + exitDirection.z
        };
        if (samePosition(landing, position)
          || !inBounds(landing)
          || blockedEdges.has(edgeKey(opponent, landing))) continue;
        moves.push(landing);
      }
    } else {
      moves.push(adjacent);
    }
  }

  const unique = new Map();
  for (const move of moves) unique.set(positionKey(move), move);
  return [...unique.values()];
}

export function validateWallCandidate(candidate, walls, playerA, playerB) {
  if (!wallGeometryIsValid(candidate)) {
    return { valid: false, reason: 'The plate falls outside the cube.' };
  }

  const occupied = blockedEdgesForWalls(walls);
  const candidateEdges = wallEdgeKeys(candidate);
  if (candidateEdges.some((key) => occupied.has(key))) {
    return { valid: false, reason: 'That plate overlaps a blocked crossing.' };
  }

  const nextBlocked = new Set(occupied);
  for (const key of candidateEdges) nextBlocked.add(key);

  if (!shortestPath(playerA, 0, nextBlocked) || !shortestPath(playerB, SIZE - 1, nextBlocked)) {
    return { valid: false, reason: 'Every player must retain a route to the opposite face.' };
  }

  return { valid: true, reason: '', blockedEdges: nextBlocked };
}

export function createInitialState() {
  const center = Math.floor(SIZE / 2);
  return {
    players: {
      A: { x: center, y: center, z: SIZE - 1 },
      B: { x: center, y: center, z: 0 }
    },
    walls: [],
    remaining: { A: BARRICADES_PER_PLAYER, B: BARRICADES_PER_PLAYER },
    turn: 'A',
    turnCount: 1,
    winner: null,
    history: [],
    feed: [{ player: 'A', text: 'Cube initialized', index: '00' }]
  };
}
