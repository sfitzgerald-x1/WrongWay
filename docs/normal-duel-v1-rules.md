# normal-duel-v1 rules contract

- **Ruleset id:** `normal-duel-v1`
- **Contract version:** `1.0.0`
- **Status:** normative Stage 0 contract

This document defines the pure, deterministic board rules used by the future
reference engine, fixtures, search, and self-play corpora. An implementation
conforming to this contract MUST follow every MUST below.

## Scope

Only normal, local, one-versus-one Duel is in scope: Standard Duel is 9x9 and
Blitz Duel is 7x7. Classic, Chaos, Hammer, stochastic/random wall drops, 2v2,
clocks, timeouts, disconnects, forfeits, online/server authority, items, stock
grants, and wall destruction are excluded. A state containing any excluded
mechanic MUST be rejected rather than implicitly reinterpreted.

The shipped client does not yet implement threefold repetition or the ply-cap
draw. This contract deliberately specifies future behavior; it is not a claim
that current online games already use these outcomes. The two complete 1v1 move
generators are `getValidMoves` in `index.html` and `getMovesFrom` in `js/ai.js`;
`moveTowardGoal` is a third partial jump path that must migrate with them.
`validMoves2` is intentionally excluded because it uses the different 2v2 jump
rule.

## Configuration

The immutable configuration contains `ruleset`, `rows`, `columns`, `start`,
`goalRows`, `initialStock`, `jumpRule`, `repetitionThreshold`, `plyCap`, and
`firstPlayer`.
For 9x9, starts are A `(8,4)`, B `(0,4)` and goals A row 0, B row 8. For 7x7,
they are A `(6,3)`, B `(0,3)`, and the same opposite goal edges. The current
product default is 10 barricades each; stock is configurable but non-negative.
`jumpRule` is `permissive-adjacent-exit-v1`; the threshold is 3. The 9x9 cap is
200. Every configuration, including 7x7, MUST explicitly provide a positive
cap; the 7x7 fixtures use 200 without declaring it a product default.

`firstPlayer` is either A or B. The side to move is `firstPlayer` after an even
number of completed plies and the other player after an odd number.

Coordinates are zero-based `{ "r": row, "c": column }`, with row zero at A's
goal edge. `square(r,c) = r * columns + c`.

## Position and GameState

A **Position** (the transposition and repetition identity) consists of config
identity, A and B squares, sorted horizontal anchors, sorted vertical anchors,
A/B remaining stock, and side to move. Wall ownership is absent: stock cannot
be inferred from the board.

The canonical Position key is the UTF-8 JSON serialization with no whitespace
of this exact array:

```text
[
  ruleset, rows, columns,
  square(start.A), square(start.B), goalRows.A, goalRows.B,
  initialStock.A, initialStock.B, jumpRule, repetitionThreshold, plyCap,
  firstPlayer,
  square(pawn.A), square(pawn.B),
  [H anchor indexes in numeric order],
  [V anchor indexes in numeric order],
  stock.A, stock.B, turn
]
```

An anchor index is `r*(columns-1)+c`. JSON numbers use their shortest decimal
integer spelling. Implementations MUST compare and sort Position keys by their
UTF-8 byte sequence.

A **GameState** is Position plus `ply`, repetition history/counts, and outcome.
`ply` counts completed player actions. History is excluded from a transposition
key. `historyStartPly` is 0 initially and becomes the resulting ply after every
wall. Canonical repetition counts include every Position from
`historyStartPly` through the current Position, are keyed by the canonical
Position key, and are sorted by key bytes. Their counts sum to
`ply-historyStartPly+1`. Immediately after a wall, the resulting Position is the
sole entry with count 1. Implementations MAY retain older entries internally,
but MUST omit them from canonical GameState serialization.

## Walls, paths, and pawn moves

Canonical walls are `H-r-c` or `V-r-c`, with `0 <= r < rows-1` and
`0 <= c < columns-1`. H blocks vertical crossings between rows `r`/`r+1` at
columns `c` and `c+1`; V blocks horizontal crossings between columns `c`/`c+1`
at rows `r` and `r+1`.

A wall is legal only if the mover has stock, the anchor is in bounds, and it:

- is not already present;
- for H, has no `H-r-(c-1)`, `H-r-(c+1)`, or `V-r-c`; for V, has no
  `V-(r-1)-c`, `V-(r+1)-c`, or `H-r-c`;
- leaves each pawn a wall-only orthogonal path to its own goal row.

The path test ignores pawn occupancy and considers only bounds and walls.

A pawn moves one open orthogonal edge to an unoccupied square. If an adjacent
open square is the opponent, it may instead exit from the opponent's square by
any open, in-bounds orthogonal edge except the edge it entered. This permissive
rule means straight and side exits may coexist; the opponent square is never a
destination.

## Actions and encoding

Actions are `{ "kind":"pawn", "to":coord }` or
`{ "kind":"wall", "wall":"H-r-c|V-r-c" }`. The canonical policy code is:

```text
pawn(r,c) = r*columns + c
H(r,c)    = rows*columns + r*(columns-1) + c
V(r,c)    = rows*columns + (rows-1)*(columns-1) + r*(columns-1) + c
total     = rows*columns + 2*(rows-1)*(columns-1)
```

9x9 therefore has 81 pawn, 64 H, 64 V actions (209 total): `H-0-0` is 81 and
`V-0-0` is 145. A legal mask spans the full policy space. Every ongoing state
MUST have at least one legal pawn action.

Whenever legal actions or destinations are serialized as arrays, they MUST be
unique and sorted by ascending policy code. Internal move-generation order is
not normative.

## Transitions and outcomes

Applying a legal action changes exactly the mover's pawn or adds one wall,
decrements only the mover's stock for a wall, increments `ply`, switches turn,
and records the resulting Position in repetition counts. The outcome is then
adjudicated in this order:

1. reaching a goal wins with
   `{ "kind":"win", "winner":"A|B", "reason":"goal" }`;
2. a third occurrence draws with
   `{ "kind":"draw", "reason":"threefold_repetition" }`;
3. completing at least `plyCap` draws with
   `{ "kind":"draw", "reason":"ply_cap" }`;
4. otherwise play remains ongoing.

Goal wins over both draw conditions, and repetition wins the tie at the cap.
No action is legal from a terminal state.

Timeout, disconnect, forfeit, and server outcomes are typed external
adjudications. The pure engine MUST NOT infer them; online server authority
requires a separate versioned parity protocol.

## Fixtures

Fixture query cases are synthetic Position-level probes and do not claim a
complete move history. Transition cases contain complete canonical GameStates
and MUST satisfy every history, stock, turn, and wall invariant. Adjudication
cases isolate winner/repetition/cap precedence from move generation. Rejection
cases pin failures for terminal actions and excluded mechanics.

`legacyComparable` is fixture metadata: `true` means the expected wall result
must match the current `tryWall`; `false` marks a deliberate contract addition
that legacy `tryWall` leaves to its caller, currently stock and bounds checks.

## Invariants and examples

Pawns are distinct/in-bounds; stocks and ply are non-negative integers; walls
are unique, in-bounds, and geometry-compatible; both pawns retain a wall-only
path; and terminal outcome agrees with history/cap rules. For a canonical
GameState, `walls.length` equals
`(initialStock.A-stock.A)+(initialStock.B-stock.B)`, neither stock exceeds its
initial value, turn agrees with `firstPlayer` and ply parity, repetition keys
are unique/sorted, and repetition counts satisfy the history sum above.

On empty 9x9 with A `(4,4)` and B `(3,4)`, A can go `(5,4)`, `(4,3)`, `(4,5)`,
`(2,4)`, `(3,3)`, `(3,5)`. With `H-2-4`, only `(2,4)` is removed. At the normal
start A's pawn codes are 67 (`(7,4)`), 75 (`(8,3)`), 77 (`(8,5)`); `H-0-0` is
81 and makes adjacent `H-0-1` and crossing `V-0-0` illegal.
