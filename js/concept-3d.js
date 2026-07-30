(function () {
  'use strict';

  var N = 9;
  var MAX_ANCHOR = N - 2;
  var DIRECTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  var mode = 'move';
  var toastTimer = null;
  var camera = { pitch: 57, yaw: -3 };
  var drag = null;

  var boardEl = document.getElementById('board');
  var wallLayerEl = document.getElementById('wallLayer');
  var pieceLayerEl = document.getElementById('pieceLayer');
  var wallTargetLayerEl = document.getElementById('wallTargetLayer');
  var worldEl = document.getElementById('world');
  var sceneEl = document.getElementById('scene');
  var toastEl = document.getElementById('toast');

  var state = freshState();

  function freshState() {
    return {
      pA: { r: N - 1, c: Math.floor(N / 2) },
      pB: { r: 0, c: Math.floor(N / 2) },
      walls: new Set(),
      barr: { A: BARR, B: BARR },
      turn: 'A',
      winner: null,
      history: [],
      feed: [{ player: 'A', text: 'Arena ready', index: '00' }]
    };
  }

  function cloneSnapshot() {
    return {
      pA: { r: state.pA.r, c: state.pA.c },
      pB: { r: state.pB.r, c: state.pB.c },
      walls: Array.from(state.walls),
      barr: { A: state.barr.A, B: state.barr.B },
      turn: state.turn,
      winner: state.winner,
      feed: state.feed.map(function (item) {
        return { player: item.player, text: item.text, index: item.index };
      })
    };
  }

  function restoreSnapshot(snapshot) {
    state.pA = { r: snapshot.pA.r, c: snapshot.pA.c };
    state.pB = { r: snapshot.pB.r, c: snapshot.pB.c };
    state.walls = new Set(snapshot.walls);
    state.barr = { A: snapshot.barr.A, B: snapshot.barr.B };
    state.turn = snapshot.turn;
    state.winner = snapshot.winner;
    state.feed = snapshot.feed.map(function (item) {
      return { player: item.player, text: item.text, index: item.index };
    });
  }

  function pushHistory() {
    state.history.push(cloneSnapshot());
  }

  function samePos(a, b) {
    return a.r === b.r && a.c === b.c;
  }

  // Mirrors the live 1v1 client: when the opponent is adjacent, every
  // unblocked non-return edge from the opponent is a valid landing square.
  function validMoves(pos, opponent, walls) {
    var moves = [];

    DIRECTIONS.forEach(function (direction) {
      var nr = pos.r + direction[0];
      var nc = pos.c + direction[1];

      if (nr < 0 || nr >= N || nc < 0 || nc >= N) return;
      if (edgeBlocked(pos.r, pos.c, nr, nc, walls)) return;

      if (nr === opponent.r && nc === opponent.c) {
        DIRECTIONS.forEach(function (exitDirection) {
          var jr = nr + exitDirection[0];
          var jc = nc + exitDirection[1];

          if (jr === pos.r && jc === pos.c) return;
          if (jr < 0 || jr >= N || jc < 0 || jc >= N) return;
          if (edgeBlocked(nr, nc, jr, jc, walls)) return;
          moves.push({ r: jr, c: jc });
        });
      } else {
        moves.push({ r: nr, c: nc });
      }
    });

    return moves.filter(function (candidate, index, all) {
      return all.findIndex(function (item) {
        return samePos(item, candidate);
      }) === index;
    });
  }

  function currentPawn() {
    return state.turn === 'A' ? state.pA : state.pB;
  }

  function otherPawn() {
    return state.turn === 'A' ? state.pB : state.pA;
  }

  function goalFor(player) {
    return player === 'A' ? 0 : N - 1;
  }

  function addFeed(player, text) {
    var number = String(state.feed.length).padStart(2, '0');
    state.feed.unshift({ player: player, text: text, index: number });
    state.feed = state.feed.slice(0, 5);
  }

  function notify(message) {
    window.clearTimeout(toastTimer);
    toastEl.textContent = message;
    toastEl.classList.add('show');
    toastTimer = window.setTimeout(function () {
      toastEl.classList.remove('show');
    }, 1800);
  }

  function cellName(r, c) {
    return String.fromCharCode(65 + c) + String(N - r);
  }

  function attemptMove(r, c) {
    if (state.winner) {
      notify('Start a new match to keep playing.');
      return;
    }

    var legal = validMoves(currentPawn(), otherPawn(), state.walls);
    var destination = { r: r, c: c };
    if (!legal.some(function (candidate) { return samePos(candidate, destination); })) {
      notify('That square is not reachable this turn.');
      return;
    }

    pushHistory();
    var player = state.turn;
    if (player === 'A') state.pA = destination;
    else state.pB = destination;
    addFeed(player, 'Moved to ' + cellName(r, c));

    if (r === goalFor(player)) {
      state.winner = player;
      notify((player === 'A' ? 'Red' : 'Blue') + ' reached the far edge.');
    } else {
      state.turn = player === 'A' ? 'B' : 'A';
    }

    render();
  }

  function wallKey(orientation, r, c) {
    return orientation + '-' + r + '-' + c;
  }

  function wallIsLegal(key) {
    if (state.winner || state.barr[state.turn] <= 0) return false;
    return !!tryWall(key, state.walls, state.pA, state.pB, 0, N - 1);
  }

  function attemptWall(orientation, r, c) {
    if (state.winner) {
      notify('Start a new match to keep playing.');
      return;
    }

    if (state.barr[state.turn] <= 0) {
      notify('No barricades remain for this player.');
      return;
    }

    var key = wallKey(orientation, r, c);
    var next = tryWall(key, state.walls, state.pA, state.pB, 0, N - 1);
    if (!next) {
      notify('That wall would overlap or close the last route.');
      return;
    }

    pushHistory();
    var player = state.turn;
    state.walls = next;
    state.barr[player] -= 1;
    addFeed(player, (orientation === 'H' ? 'Horizontal' : 'Vertical') + ' wall at ' + cellName(r, c));
    state.turn = player === 'A' ? 'B' : 'A';
    render();
  }

  function createBoard() {
    var fragment = document.createDocumentFragment();

    for (var r = 0; r < N; r += 1) {
      for (var c = 0; c < N; c += 1) {
        var cell = document.createElement('button');
        cell.type = 'button';
        cell.className = 'cell';
        cell.dataset.row = String(r);
        cell.dataset.col = String(c);
        cell.setAttribute('role', 'gridcell');
        cell.setAttribute('aria-label', 'Square ' + cellName(r, c));
        cell.addEventListener('click', onCellClick);
        fragment.appendChild(cell);
      }
    }

    boardEl.appendChild(fragment);
  }

  function onCellClick(event) {
    if (mode !== 'move') return;
    attemptMove(Number(event.currentTarget.dataset.row), Number(event.currentTarget.dataset.col));
  }

  function createWallTargets() {
    var fragment = document.createDocumentFragment();

    ['H', 'V'].forEach(function (orientation) {
      for (var r = 0; r <= MAX_ANCHOR; r += 1) {
        for (var c = 0; c <= MAX_ANCHOR; c += 1) {
          var target = document.createElement('button');
          target.type = 'button';
          target.className = 'wall-target';
          target.dataset.orientation = orientation;
          target.dataset.row = String(r);
          target.dataset.col = String(c);
          target.setAttribute('aria-label', (orientation === 'H' ? 'Horizontal' : 'Vertical') + ' wall at ' + cellName(r, c));

          if (orientation === 'H') {
            target.style.left = (c * 100 / N) + '%';
            target.style.top = (((r + 1) * 100 / N) - 3) + '%';
            target.style.width = (200 / N) + '%';
            target.style.height = '6%';
          } else {
            target.style.left = (((c + 1) * 100 / N) - 3) + '%';
            target.style.top = (r * 100 / N) + '%';
            target.style.width = '6%';
            target.style.height = (200 / N) + '%';
          }

          target.addEventListener('click', function (event) {
            var button = event.currentTarget;
            attemptWall(
              button.dataset.orientation,
              Number(button.dataset.row),
              Number(button.dataset.col)
            );
          });
          fragment.appendChild(target);
        }
      }
    });

    wallTargetLayerEl.appendChild(fragment);
  }

  function renderCells() {
    var legal = state.winner ? [] : validMoves(currentPawn(), otherPawn(), state.walls);
    var cells = boardEl.querySelectorAll('.cell');

    cells.forEach(function (cell) {
      var position = {
        r: Number(cell.dataset.row),
        c: Number(cell.dataset.col)
      };
      var isLegal = mode === 'move' && legal.some(function (candidate) {
        return samePos(candidate, position);
      });
      cell.classList.toggle('is-valid', isLegal);
      cell.disabled = !isLegal;
    });
  }

  function renderPieces() {
    pieceLayerEl.replaceChildren();
    [
      { id: 'A', position: state.pA, className: 'pawn-red' },
      { id: 'B', position: state.pB, className: 'pawn-blue' }
    ].forEach(function (item) {
      var pawn = document.createElement('div');
      pawn.className = 'pawn ' + item.className;
      pawn.dataset.player = item.id;
      pawn.style.left = ((item.position.c + 0.176) * 100 / N) + '%';
      pawn.style.top = ((item.position.r + 0.176) * 100 / N) + '%';
      pieceLayerEl.appendChild(pawn);
    });
  }

  function renderWalls() {
    wallLayerEl.replaceChildren();
    state.walls.forEach(function (key) {
      var parts = key.split('-');
      var orientation = parts[0];
      var r = Number(parts[1]);
      var c = Number(parts[2]);
      var wall = document.createElement('div');
      wall.className = 'wall ' + (orientation === 'H' ? 'wall-h' : 'wall-v');

      if (orientation === 'H') {
        wall.style.left = (c * 100 / N) + '%';
        wall.style.top = (((r + 1) * 100 / N) - 1.2) + '%';
        wall.style.width = (200 / N) + '%';
      } else {
        wall.style.left = (((c + 1) * 100 / N) - 1.2) + '%';
        wall.style.top = (r * 100 / N) + '%';
        wall.style.height = (200 / N) + '%';
      }

      wallLayerEl.appendChild(wall);
    });
  }

  function renderWallTargets() {
    var orientation = mode === 'wall-h' ? 'H' : mode === 'wall-v' ? 'V' : null;
    wallTargetLayerEl.classList.toggle('active', !!orientation);
    var targets = wallTargetLayerEl.querySelectorAll('.wall-target');

    targets.forEach(function (target) {
      var visible = target.dataset.orientation === orientation;
      var key = wallKey(
        target.dataset.orientation,
        Number(target.dataset.row),
        Number(target.dataset.col)
      );
      target.hidden = !visible;
      target.classList.toggle('is-legal', visible && wallIsLegal(key));
    });
  }

  function renderHud() {
    document.getElementById('wallsA').textContent = String(state.barr.A);
    document.getElementById('wallsB').textContent = String(state.barr.B);
    document.getElementById('cardA').classList.toggle('active', state.turn === 'A' && !state.winner);
    document.getElementById('cardB').classList.toggle('active', state.turn === 'B' && !state.winner);
    document.getElementById('undoButton').disabled = state.history.length === 0;

    var turnName = state.turn === 'A' ? 'RED' : 'BLUE';
    var turnColor = state.turn === 'A' ? 'var(--red)' : 'var(--blue)';
    var turnLabel = document.getElementById('turnLabel');
    var modeHint = document.getElementById('modeHint');
    var turnDot = document.getElementById('turnDot');

    turnDot.style.background = turnColor;
    turnDot.style.color = turnColor;

    if (state.winner) {
      var winnerName = state.winner === 'A' ? 'RED' : 'BLUE';
      turnLabel.textContent = winnerName + ' WINS';
      modeHint.textContent = 'The far edge belongs to ' + winnerName.toLowerCase() + '.';
    } else {
      turnLabel.textContent = turnName + ' TO MOVE';
      modeHint.textContent = mode === 'move'
        ? 'Choose a glowing destination.'
        : 'Choose a glowing ' + (mode === 'wall-h' ? 'horizontal' : 'vertical') + ' anchor.';
    }

    document.body.classList.toggle('winner-red', state.winner === 'A');
    document.body.classList.toggle('winner-blue', state.winner === 'B');
  }

  function renderPaths() {
    var pathA = bfsPath(state.pA, state.walls, 0);
    var pathB = bfsPath(state.pB, state.walls, N - 1);
    var distanceA = pathA ? pathA.length - 1 : 0;
    var distanceB = pathB ? pathB.length - 1 : 0;
    var maxDistance = Math.max(16, distanceA, distanceB);

    document.getElementById('pathA').textContent = String(distanceA).padStart(2, '0');
    document.getElementById('pathB').textContent = String(distanceB).padStart(2, '0');
    document.getElementById('pathBarA').style.width = Math.max(8, 100 - distanceA / maxDistance * 82) + '%';
    document.getElementById('pathBarB').style.width = Math.max(8, 100 - distanceB / maxDistance * 82) + '%';

    var delta = distanceA - distanceB;
    var insight = 'The arena is balanced. One wall can change the tempo.';
    if (delta <= -2) insight = 'Red owns the shorter route. Blue needs to reshape the board.';
    if (delta >= 2) insight = 'Blue owns the shorter route. Red needs to disrupt the line.';
    document.getElementById('routeInsight').textContent = insight;
  }

  function renderFeed() {
    var list = document.getElementById('actionLog');
    list.replaceChildren();
    state.feed.forEach(function (item) {
      var entry = document.createElement('li');
      entry.className = item.player === 'A' ? 'red' : 'blue';

      var dot = document.createElement('i');
      var text = document.createElement('span');
      var index = document.createElement('b');
      text.textContent = item.text;
      index.textContent = item.index;

      entry.append(dot, text, index);
      list.appendChild(entry);
    });
  }

  function renderModes() {
    document.querySelectorAll('.mode-button').forEach(function (button) {
      button.classList.toggle('active', button.dataset.mode === mode);
      button.setAttribute('aria-pressed', button.dataset.mode === mode ? 'true' : 'false');
    });
  }

  function render() {
    renderCells();
    renderPieces();
    renderWalls();
    renderWallTargets();
    renderHud();
    renderPaths();
    renderFeed();
    renderModes();
  }

  function setMode(nextMode) {
    mode = nextMode;
    render();
  }

  function undo() {
    if (!state.history.length) return;
    var snapshot = state.history.pop();
    restoreSnapshot(snapshot);
    notify('Last action rewound.');
    render();
  }

  function resetMatch() {
    state = freshState();
    mode = 'move';
    notify('A new arena is ready.');
    render();
  }

  function applyCamera() {
    worldEl.style.setProperty('--pitch', camera.pitch + 'deg');
    worldEl.style.setProperty('--yaw', camera.yaw + 'deg');
  }

  function rotateCamera(delta) {
    camera.yaw += delta;
    applyCamera();
  }

  function resetCamera() {
    camera.pitch = 57;
    camera.yaw = -3;
    applyCamera();
  }

  function onPointerDown(event) {
    if (event.target.closest('button')) return;
    drag = {
      id: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      pitch: camera.pitch,
      yaw: camera.yaw
    };
    sceneEl.setPointerCapture(event.pointerId);
    worldEl.classList.add('dragging');
  }

  function onPointerMove(event) {
    if (!drag || drag.id !== event.pointerId) return;
    camera.yaw = drag.yaw + (event.clientX - drag.x) * 0.16;
    camera.pitch = Math.max(42, Math.min(68, drag.pitch - (event.clientY - drag.y) * 0.12));
    applyCamera();
  }

  function onPointerUp(event) {
    if (!drag || drag.id !== event.pointerId) return;
    drag = null;
    worldEl.classList.remove('dragging');
    if (sceneEl.hasPointerCapture(event.pointerId)) sceneEl.releasePointerCapture(event.pointerId);
  }

  document.querySelectorAll('.mode-button').forEach(function (button) {
    button.addEventListener('click', function () {
      setMode(button.dataset.mode);
    });
  });

  document.getElementById('undoButton').addEventListener('click', undo);
  document.getElementById('resetButton').addEventListener('click', resetMatch);
  document.getElementById('viewLeft').addEventListener('click', function () { rotateCamera(-12); });
  document.getElementById('viewRight').addEventListener('click', function () { rotateCamera(12); });
  document.getElementById('cameraHome').addEventListener('click', resetCamera);

  sceneEl.addEventListener('pointerdown', onPointerDown);
  sceneEl.addEventListener('pointermove', onPointerMove);
  sceneEl.addEventListener('pointerup', onPointerUp);
  sceneEl.addEventListener('pointercancel', onPointerUp);

  document.addEventListener('keydown', function (event) {
    if (event.target.matches('input, textarea, select')) return;
    var key = event.key.toLowerCase();
    if (key === 'm') setMode('move');
    if (key === 'h') setMode('wall-h');
    if (key === 'v') setMode('wall-v');
    if (key === 'u') undo();
    if (key === 'r') resetMatch();
    if (event.key === 'ArrowLeft') rotateCamera(-6);
    if (event.key === 'ArrowRight') rotateCamera(6);
    if (event.key === 'ArrowUp') {
      camera.pitch = Math.min(68, camera.pitch + 3);
      applyCamera();
    }
    if (event.key === 'ArrowDown') {
      camera.pitch = Math.max(42, camera.pitch - 3);
      applyCamera();
    }
  });

  createBoard();
  createWallTargets();
  applyCamera();
  render();
}());
