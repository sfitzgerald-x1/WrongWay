//! Exact solution of positions in which both players have spent every barricade.
//!
//! With no stock left the wall layout is frozen for the rest of the game, so what
//! remains is a two-pawn race on a fixed board. That is a finite game small enough
//! to solve outright: `cells * cells * 2` states, about 13k on the canonical 9x9,
//! which retrograde analysis settles in one pass.
//!
//! **It is not two independent shortest paths.** The pawns interact: when they are
//! adjacent the mover may hop, which can shorten a route or open one that the
//! opponent's body was blocking. So the answer is a real solve, and the moves it
//! prefers are not always the ones a distance field would pick. Move generation is
//! therefore taken from [`Board::legal_codes_into`] -- the engine's own -- rather
//! than reimplemented here, because a jump rule reimplemented slightly differently
//! would produce a solver that is confidently wrong.
//!
//! Measured on played games, every game exhausts both stocks and 18.2% of all plies
//! are played after that point. Those plies currently consume a full simulation
//! budget to approximate an answer this module computes exactly.
use crate::{Board, Config, Coord, NormalDuelError, Player, Players, Result};

/// Who wins under optimal play, and in how many plies.
///
/// `plies` is the distance to the win, which the caller needs: a forced win longer
/// than the remaining ply cap is a draw in the game actually being played.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endgame {
    Wins { player: Player, plies: u32 },
    /// Neither side can force a crossing. Under threefold repetition and the ply
    /// cap this is the drawn result.
    Draw,
}

/// A solved wall layout: every (pawn A, pawn B, mover) triple on one frozen board.
#[derive(Debug, Clone)]
pub struct EndgameTable {
    cells: usize,
    /// Indexed by [`Self::index`]. `None` is a draw.
    entries: Vec<Option<(Player, u32)>>,
}

impl EndgameTable {
    #[must_use]
    fn index(&self, a: usize, b: usize, turn: Player) -> usize {
        ((a * self.cells) + b) * 2 + usize::from(turn == Player::B)
    }

    /// The exact result from this position, or `None` if the pawns are not both on
    /// the board or share a square.
    #[must_use]
    pub fn lookup(&self, config: &Config, pawns: Players<Coord>, turn: Player) -> Option<Endgame> {
        let a = coord_index(config, pawns.a)?;
        let b = coord_index(config, pawns.b)?;
        if a == b {
            return None;
        }
        Some(match self.entries[self.index(a, b, turn)] {
            Some((player, plies)) => Endgame::Wins { player, plies },
            None => Endgame::Draw,
        })
    }

    /// The best pawn destination from this position, as an action code.
    ///
    /// Wins are taken as fast as possible and losses delayed as long as possible,
    /// which is what makes a rollout of this line terminate in exactly the number
    /// of plies [`Endgame::Wins`] reports -- the property the tests check, and the
    /// one that would break first if move generation here drifted from the engine's.
    #[must_use]
    pub fn best_move(
        &self,
        config: &Config,
        board_walls: &[String],
        pawns: Players<Coord>,
        turn: Player,
    ) -> Option<u16> {
        let board = Board::from_layout(config, board_walls).ok()?;
        let zero = Players { a: 0_u64, b: 0_u64 };
        let mut codes = vec![0_u16; config.policy_size()];
        let mut stats = Default::default();
        let count = board.legal_codes_into(config, pawns, zero, turn, &mut codes, &mut stats);
        let a = coord_index(config, pawns.a)?;
        let b = coord_index(config, pawns.b)?;
        let mut best: Option<(u16, i64)> = None;
        for code in &codes[..count] {
            let destination = usize::from(*code);
            if destination >= self.cells {
                continue;
            }
            let (na, nb) = match turn {
                Player::A => (destination, b),
                Player::B => (a, destination),
            };
            if na == nb {
                continue;
            }
            // Rank from the MOVER's view: winning soon is best, losing late is least
            // bad, and a draw sits between the two.
            let score = match self.entries[self.index(na, nb, turn.other())] {
                Some((winner, plies)) if winner == turn => 1_000_000 - i64::from(plies),
                Some((_, plies)) => -1_000_000 + i64::from(plies),
                None => 0,
            };
            if best.is_none_or(|(_, current)| score > current) {
                best = Some((*code, score));
            }
        }
        best.map(|(code, _)| code)
    }

    /// States solved, for tests and instrumentation.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn coord_index(config: &Config, coord: Coord) -> Option<usize> {
    if coord.r >= config.rows || coord.c >= config.columns {
        return None;
    }
    Some(coord.r as usize * config.columns as usize + coord.c as usize)
}

fn coord_of(config: &Config, index: usize) -> Coord {
    let columns = config.columns as usize;
    Coord { r: (index / columns) as u8, c: (index % columns) as u8 }
}

/// Solve every zero-stock position on the board described by `walls`.
///
/// `walls` is the frozen layout, in the same textual form a [`crate::Position`]
/// carries. The result depends only on that layout, so a caller may cache it by
/// wall-layout hash and reuse it for every position sharing those walls.
pub fn solve_layout(config: &Config, walls: &[String]) -> Result<EndgameTable> {
    config.validate()?;
    let cells = config.cells();
    let board = Board::from_layout(config, walls)?;
    let zero = Players { a: 0_u64, b: 0_u64 };

    // 1. Successors, from the engine's own move generation. At zero stock every
    //    legal action is a pawn move, and pawn codes are exactly the destination
    //    square, so a code IS the successor's mover square.
    let states = cells * cells * 2;
    let mut successors: Vec<Vec<u32>> = vec![Vec::new(); states];
    let mut codes = vec![0_u16; config.policy_size()];
    let index = |a: usize, b: usize, turn: Player| ((a * cells) + b) * 2 + usize::from(turn == Player::B);

    for a in 0..cells {
        for b in 0..cells {
            if a == b {
                continue;
            }
            let pawns = Players { a: coord_of(config, a), b: coord_of(config, b) };
            if decided(config, pawns).is_some() {
                continue; // terminal: the game is already over here
            }
            for turn in [Player::A, Player::B] {
                let mut stats = Default::default();
                let count = board.legal_codes_into(config, pawns, zero, turn, &mut codes, &mut stats);
                let here = index(a, b, turn);
                for code in &codes[..count] {
                    let destination = usize::from(*code);
                    debug_assert!(destination < cells, "zero stock must leave only pawn moves");
                    let (na, nb) = match turn {
                        Player::A => (destination, b),
                        Player::B => (a, destination),
                    };
                    successors[here].push(index(na, nb, turn.other()) as u32);
                }
            }
        }
    }

    // 2. Retrograde analysis. Predecessors first, then a queue from the terminals.
    let mut predecessors: Vec<Vec<u32>> = vec![Vec::new(); states];
    let mut remaining = vec![0_u32; states];
    for (from, tos) in successors.iter().enumerate() {
        remaining[from] = tos.len() as u32;
        for to in tos {
            predecessors[*to as usize].push(from as u32);
        }
    }

    let mut label: Vec<Option<(Player, u32)>> = vec![None; states];
    let mut queue: Vec<u32> = Vec::new();
    for a in 0..cells {
        for b in 0..cells {
            if a == b {
                continue;
            }
            let pawns = Players { a: coord_of(config, a), b: coord_of(config, b) };
            if let Some(winner) = decided(config, pawns) {
                for turn in [Player::A, Player::B] {
                    let here = index(a, b, turn);
                    label[here] = Some((winner, 0));
                    queue.push(here as u32);
                }
            }
        }
    }

    let mut head = 0;
    while head < queue.len() {
        let state = queue[head] as usize;
        head += 1;
        let (winner, plies) = label[state].expect("queued states are labelled");
        for previous in &predecessors[state] {
            let previous = *previous as usize;
            if label[previous].is_some() {
                continue;
            }
            let mover = if previous % 2 == 0 { Player::A } else { Player::B };
            if winner == mover {
                // The mover can step into a position it already wins: take it, and
                // the first one found is the shortest because the queue is in
                // non-decreasing distance order.
                label[previous] = Some((mover, plies + 1));
                queue.push(previous as u32);
            } else {
                remaining[previous] -= 1;
                if remaining[previous] == 0 {
                    // Every move loses; the mover is lost and delays as long as it can.
                    label[previous] = Some((winner, plies + 1));
                    queue.push(previous as u32);
                }
            }
        }
    }

    Ok(EndgameTable { cells, entries: label })
}

/// The winner if this pawn placement has already ended the game.
fn decided(config: &Config, pawns: Players<Coord>) -> Option<Player> {
    if pawns.a.r == config.goal_rows.a {
        return Some(Player::A);
    }
    if pawns.b.r == config.goal_rows.b {
        return Some(Player::B);
    }
    None
}

impl Board {
    /// A board from a bare wall list, with no pawns involved.
    fn from_layout(config: &Config, walls: &[String]) -> Result<Self> {
        let position = crate::Position {
            pawns: Players { a: Coord { r: 0, c: 0 }, b: Coord { r: config.rows - 1, c: 0 } },
            walls: walls.to_vec(),
            stock: Players { a: 0, b: 0 },
            turn: Player::A,
        };
        Board::from_position(config, &position).map_err(|_| NormalDuelError::InvalidPosition)
    }
}
