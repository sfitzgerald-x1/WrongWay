//! Native, deterministic implementation of the `normal-duel-v1` contract.
//!
//! The crate is intentionally platform-neutral: its public data structures use
//! the fixture JSON field names, while [`SearchPosition`] supplies a compact
//! mutable apply/undo representation for a future native evaluator.

#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;

pub mod js_math;
pub mod mock_evaluator;
pub mod puct;
mod search;
pub mod selfplay;

pub use search::{search_for, search_nodes, SearchDiagnostics, SearchOptions, SearchReport};

pub const RULESET: &str = "normal-duel-v1";
pub const JUMP_RULE: &str = "permissive-adjacent-exit-v1";
pub const REPETITION_THRESHOLD: u8 = 3;
pub const MAX_PERFT_DEPTH: u8 = 4;
pub const DEFAULT_MAX_PERFT_NODES: u64 = 400;
pub const MAX_PERFT_NODES_HARD_CAP: u64 = 5_000;
/// JavaScript accepts only safe integers. All wire-format counters and stock
/// values are therefore bounded here even though Rust's `u64` can represent a
/// wider range.
pub const MAX_JS_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

const DIRECTIONS: [(i8, i8); 4] = [(-1, 0), (0, -1), (0, 1), (1, 0)];

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NormalDuelError {
    #[error("invalid_config")]
    InvalidConfig,
    #[error("invalid_position")]
    InvalidPosition,
    #[error("invalid_state")]
    InvalidState,
    #[error("invalid_action")]
    InvalidAction,
    #[error("invalid_action_code")]
    InvalidActionCode,
    #[error("invalid_wall")]
    InvalidWall,
    #[error("invalid_wall_geometry")]
    InvalidWallGeometry,
    #[error("path_blocked")]
    PathBlocked,
    #[error("invalid_edge")]
    InvalidEdge,
    #[error("invalid_adjudication")]
    InvalidAdjudication,
    #[error("terminal_state")]
    TerminalState,
    #[error("illegal_action")]
    IllegalAction,
    #[error("unsupported_feature")]
    UnsupportedFeature,
    #[error("perft depth exceeds {MAX_PERFT_DEPTH}")]
    PerftDepth,
    #[error("perft count exceeds u64")]
    PerftOverflow,
    #[error("invalid perft node budget")]
    InvalidPerftBudget,
    #[error("perft node budget exceeded")]
    PerftNodeBudget,
    #[error("invalid search options")]
    InvalidSearchOptions,
    #[error("invalid search budget")]
    InvalidSearchBudget,
}

impl NormalDuelError {
    /// Stable, JSON-facing failure code shared with the reference contract.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidConfig => "invalid_config",
            Self::InvalidPosition => "invalid_position",
            Self::InvalidState => "invalid_state",
            Self::InvalidAction => "invalid_action",
            Self::InvalidActionCode => "invalid_action_code",
            Self::InvalidWall => "invalid_wall",
            Self::InvalidWallGeometry => "invalid_wall_geometry",
            Self::PathBlocked => "path_blocked",
            Self::InvalidEdge => "invalid_edge",
            Self::InvalidAdjudication => "invalid_adjudication",
            Self::TerminalState => "terminal_state",
            Self::IllegalAction => "illegal_action",
            Self::UnsupportedFeature => "unsupported_feature",
            Self::PerftDepth => "perft_depth",
            Self::PerftOverflow => "perft_overflow",
            Self::InvalidPerftBudget => "invalid_perft_budget",
            Self::PerftNodeBudget => "perft_node_budget",
            Self::InvalidSearchOptions => "invalid_search_options",
            Self::InvalidSearchBudget => "invalid_search_budget",
        }
    }
}

pub type Result<T> = std::result::Result<T, NormalDuelError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Player {
    #[serde(rename = "A")]
    A,
    #[serde(rename = "B")]
    B,
}

impl Player {
    #[must_use]
    pub const fn other(self) -> Self {
        match self {
            Self::A => Self::B,
            Self::B => Self::A,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Players<T> {
    #[serde(rename = "A")]
    pub a: T,
    #[serde(rename = "B")]
    pub b: T,
}

impl<T> Players<T> {
    #[must_use]
    pub fn get(&self, player: Player) -> &T {
        match player {
            Player::A => &self.a,
            Player::B => &self.b,
        }
    }

    pub fn get_mut(&mut self, player: Player) -> &mut T {
        match player {
            Player::A => &mut self.a,
            Player::B => &mut self.b,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Coord {
    pub r: u8,
    pub c: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub ruleset: String,
    pub rows: u8,
    pub columns: u8,
    pub start: Players<Coord>,
    #[serde(rename = "goalRows")]
    pub goal_rows: Players<u8>,
    #[serde(rename = "initialStock")]
    pub initial_stock: Players<u64>,
    #[serde(rename = "jumpRule")]
    pub jump_rule: String,
    #[serde(rename = "repetitionThreshold")]
    pub repetition_threshold: u8,
    #[serde(rename = "plyCap")]
    pub ply_cap: u64,
    #[serde(rename = "firstPlayer")]
    pub first_player: Player,
}

impl Config {
    /// Validate the complete normal-duel-v1 configuration. The only accepted
    /// boards are the product's 7x7 and 9x9 center-start variants.
    pub fn validate(&self) -> Result<()> {
        if self.ruleset != RULESET
            || self.jump_rule != JUMP_RULE
            || self.repetition_threshold != REPETITION_THRESHOLD
            || !matches!((self.rows, self.columns), (7, 7) | (9, 9))
            || self.ply_cap == 0
            || self.ply_cap > MAX_JS_SAFE_INTEGER
            || self.initial_stock.a > MAX_JS_SAFE_INTEGER
            || self.initial_stock.b > MAX_JS_SAFE_INTEGER
        {
            return Err(NormalDuelError::InvalidConfig);
        }
        let middle = self.columns / 2;
        if self.start.a
            != (Coord {
                r: self.rows - 1,
                c: middle,
            })
            || self.start.b != (Coord { r: 0, c: middle })
            || self.goal_rows.a != 0
            || self.goal_rows.b != self.rows - 1
        {
            return Err(NormalDuelError::InvalidConfig);
        }
        Ok(())
    }

    #[must_use]
    pub const fn cells(&self) -> usize {
        self.rows as usize * self.columns as usize
    }

    #[must_use]
    pub const fn anchors_per_axis(&self) -> usize {
        self.rows.saturating_sub(1) as usize * self.columns.saturating_sub(1) as usize
    }

    #[must_use]
    pub const fn policy_size(&self) -> usize {
        self.cells() + 2 * self.anchors_per_axis()
    }

    #[must_use]
    pub fn expected_turn(&self, ply: u64) -> Player {
        if ply % 2 == 0 {
            self.first_player
        } else {
            self.first_player.other()
        }
    }

    #[must_use]
    pub fn completed_actions(&self, ply: u64) -> Players<u64> {
        let first = ply.div_ceil(2);
        let second = ply / 2;
        match self.first_player {
            Player::A => Players {
                a: first,
                b: second,
            },
            Player::B => Players {
                a: second,
                b: first,
            },
        }
    }
}

/// Strict JSON boundary for native/FFI callers. It preserves the reference
/// engine's useful distinction between excluded mechanics (`features`) and
/// every other malformed configuration. Typed Rust callers can use
/// [`Config::validate`] directly after deserialization.
pub fn validate_config_json(input: &Value) -> Result<Config> {
    if input
        .as_object()
        .is_some_and(|object| object.contains_key("features"))
    {
        return Err(NormalDuelError::UnsupportedFeature);
    }
    let normalized = normalize_js_safe_integers(input).ok_or(NormalDuelError::InvalidConfig)?;
    let config =
        serde_json::from_value::<Config>(normalized).map_err(|_| NormalDuelError::InvalidConfig)?;
    config.validate()?;
    Ok(config)
}

fn normalize_js_safe_integers(input: &Value) -> Option<Value> {
    match input {
        Value::Number(number) => {
            if let Some(value) = number.as_u64() {
                return (value <= MAX_JS_SAFE_INTEGER).then(|| Value::Number(value.into()));
            }
            let value = number.as_f64()?;
            if value.is_finite()
                && value >= 0.0
                && value <= MAX_JS_SAFE_INTEGER as f64
                && value.fract() == 0.0
            {
                Some(Value::Number((value as u64).into()))
            } else {
                None
            }
        }
        Value::Array(values) => values
            .iter()
            .map(normalize_js_safe_integers)
            .collect::<Option<Vec<_>>>()
            .map(Value::Array),
        Value::Object(object) => object
            .iter()
            .map(|(key, value)| {
                normalize_js_safe_integers(value).map(|normalized| (key.clone(), normalized))
            })
            .collect::<Option<serde_json::Map<_, _>>>()
            .map(Value::Object),
        _ => Some(input.clone()),
    }
}

/// Strict state DTO decode with JavaScript-number semantics. Exponent-form
/// safe integers such as `1e1` are accepted as integer 10, while fractional,
/// negative, or unsafe numeric values fail closed.
pub fn game_state_from_json(input: &Value) -> Result<GameState> {
    serde_json::from_value(normalize_js_safe_integers(input).ok_or(NormalDuelError::InvalidState)?)
        .map_err(|_| NormalDuelError::InvalidState)
}

/// Strict Position DTO decode with JavaScript safe-integer normalization.
pub fn position_from_json(input: &Value) -> Result<Position> {
    serde_json::from_value(
        normalize_js_safe_integers(input).ok_or(NormalDuelError::InvalidPosition)?,
    )
    .map_err(|_| NormalDuelError::InvalidPosition)
}

/// Strict Action DTO decode with the same JavaScript safe-integer boundary.
pub fn action_from_json(input: &Value) -> Result<Action> {
    serde_json::from_value(normalize_js_safe_integers(input).ok_or(NormalDuelError::InvalidAction)?)
        .map_err(|_| NormalDuelError::InvalidAction)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Orientation {
    Horizontal,
    Vertical,
}

impl Orientation {
    #[must_use]
    pub const fn as_char(self) -> char {
        match self {
            Self::Horizontal => 'H',
            Self::Vertical => 'V',
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Wall {
    pub orientation: Orientation,
    pub r: u8,
    pub c: u8,
}

impl Wall {
    pub fn parse(config: &Config, text: &str) -> Result<Self> {
        config.validate()?;
        let mut pieces = text.split('-');
        let Some(orientation) = pieces.next() else {
            return Err(NormalDuelError::InvalidWall);
        };
        let Some(r_text) = pieces.next() else {
            return Err(NormalDuelError::InvalidWall);
        };
        let Some(c_text) = pieces.next() else {
            return Err(NormalDuelError::InvalidWall);
        };
        if pieces.next().is_some()
            || !(orientation == "H" || orientation == "V")
            || !canonical_unsigned(r_text)
            || !canonical_unsigned(c_text)
        {
            return Err(NormalDuelError::InvalidWall);
        }
        let r = r_text
            .parse::<u8>()
            .map_err(|_| NormalDuelError::InvalidWall)?;
        let c = c_text
            .parse::<u8>()
            .map_err(|_| NormalDuelError::InvalidWall)?;
        if r >= config.rows - 1 || c >= config.columns - 1 {
            return Err(NormalDuelError::InvalidWall);
        }
        Ok(Self {
            orientation: if orientation == "H" {
                Orientation::Horizontal
            } else {
                Orientation::Vertical
            },
            r,
            c,
        })
    }

    #[must_use]
    pub fn text(self) -> String {
        format!("{}-{}-{}", self.orientation.as_char(), self.r, self.c)
    }

    #[must_use]
    /// The anchor square, i.e. the upper-left of the two cells the wall's
    /// first blocked edge separates.
    const fn coord(self) -> Coord {
        Coord {
            r: self.r,
            c: self.c,
        }
    }

    fn anchor_index(self, config: &Config) -> usize {
        self.r as usize * (config.columns as usize - 1) + self.c as usize
    }

    #[must_use]
    fn policy_code(self, config: &Config) -> usize {
        config.cells()
            + if self.orientation == Orientation::Vertical {
                config.anchors_per_axis()
            } else {
                0
            }
            + self.anchor_index(config)
    }
}

fn canonical_unsigned(text: &str) -> bool {
    !text.is_empty()
        && text.bytes().all(|byte| byte.is_ascii_digit())
        && (text == "0" || !text.starts_with('0'))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Position {
    pub pawns: Players<Coord>,
    pub walls: Vec<String>,
    pub stock: Players<u64>,
    pub turn: Player,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepetitionCount {
    #[serde(rename = "positionKey")]
    pub position_key: String,
    pub count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum Outcome {
    Ongoing,
    Win { winner: Player, reason: GoalReason },
    Draw { reason: DrawReason },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalReason {
    #[serde(rename = "goal")]
    Goal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DrawReason {
    #[serde(rename = "threefold_repetition")]
    ThreefoldRepetition,
    #[serde(rename = "ply_cap")]
    PlyCap,
}

impl Outcome {
    #[must_use]
    pub const fn is_ongoing(&self) -> bool {
        matches!(self, Self::Ongoing)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameState {
    pub position: Position,
    #[serde(rename = "positionKey")]
    pub position_key: String,
    pub ply: u64,
    #[serde(rename = "historyStartPly")]
    pub history_start_ply: u64,
    #[serde(rename = "repetitionCounts")]
    pub repetition_counts: Vec<RepetitionCount>,
    pub outcome: Outcome,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum Action {
    Pawn { to: Coord },
    Wall { wall: String },
}

impl Action {
    pub fn policy_code(&self, config: &Config) -> Result<usize> {
        config.validate()?;
        self.policy_code_validated(config)
    }

    fn policy_code_validated(&self, config: &Config) -> Result<usize> {
        match self {
            Self::Pawn { to } if in_bounds(config, *to) => Ok(square(config, *to)),
            Self::Pawn { .. } => Err(NormalDuelError::InvalidAction),
            // Wall-query APIs expose `invalid_wall`, but an invalid wall
            // carried by an Action has the reference engine's
            // `invalid_action` code.
            Self::Wall { wall } => Ok(Wall::parse(config, wall)
                .map_err(|_| NormalDuelError::InvalidAction)?
                .policy_code(config)),
        }
    }
}

/// Decodes all policy codes, including currently-illegal actions, exactly as
/// the JavaScript reference engine does.
pub fn decode_action(config: &Config, code: usize) -> Result<Action> {
    config.validate()?;
    if code >= config.policy_size() {
        return Err(NormalDuelError::InvalidActionCode);
    }
    if code < config.cells() {
        return Ok(Action::Pawn {
            to: Coord {
                r: (code / config.columns as usize) as u8,
                c: (code % config.columns as usize) as u8,
            },
        });
    }
    let offset = code - config.cells();
    let anchors = config.anchors_per_axis();
    let orientation = if offset < anchors {
        Orientation::Horizontal
    } else {
        Orientation::Vertical
    };
    let anchor = offset % anchors;
    Ok(Action::Wall {
        wall: Wall {
            orientation,
            r: (anchor / (config.columns as usize - 1)) as u8,
            c: (anchor % (config.columns as usize - 1)) as u8,
        }
        .text(),
    })
}

pub fn encode_action(config: &Config, action: &Action) -> Result<usize> {
    config.validate()?;
    action.policy_code_validated(config)
}

pub fn policy_size(config: &Config) -> Result<usize> {
    config.validate()?;
    Ok(config.policy_size())
}

fn square(config: &Config, coord: Coord) -> usize {
    coord.r as usize * config.columns as usize + coord.c as usize
}

fn in_bounds(config: &Config, coord: Coord) -> bool {
    coord.r < config.rows && coord.c < config.columns
}

/// Compact wall occupancy. The 9x9 board has exactly 64 anchors per
/// orientation, so a `u64` fully represents either anchor grid (7x7 simply
/// uses its low 36 bits).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct WallBits {
    pub horizontal: u64,
    pub vertical: u64,
}

impl WallBits {
    fn anchor_bit(wall: Wall, config: &Config) -> Option<u64> {
        let anchor_rows = config.rows.checked_sub(1)?;
        let anchor_columns = config.columns.checked_sub(1)?;
        if wall.r >= anchor_rows || wall.c >= anchor_columns {
            return None;
        }
        let anchor = usize::from(wall.r)
            .checked_mul(usize::from(anchor_columns))?
            .checked_add(usize::from(wall.c))?;
        1_u64.checked_shl(u32::try_from(anchor).ok()?)
    }

    fn contains_valid(self, wall: Wall, config: &Config) -> bool {
        let Some(bit) = Self::anchor_bit(wall, config) else {
            return false;
        };
        match wall.orientation {
            Orientation::Horizontal => self.horizontal & bit != 0,
            Orientation::Vertical => self.vertical & bit != 0,
        }
    }

    #[must_use]
    pub fn contains(self, wall: Wall, config: &Config) -> bool {
        if config.validate().is_err() {
            return false;
        }
        self.contains_valid(wall, config)
    }

    fn insert_valid(&mut self, wall: Wall, config: &Config) -> bool {
        let Some(bit) = Self::anchor_bit(wall, config) else {
            return false;
        };
        match wall.orientation {
            Orientation::Horizontal => self.horizontal |= bit,
            Orientation::Vertical => self.vertical |= bit,
        }
        true
    }

    /// Insert a valid anchor, returning false without mutation for a foreign or
    /// out-of-bounds wall. This remains total even on the 64-anchor 9x9 edge.
    pub fn insert(&mut self, wall: Wall, config: &Config) -> bool {
        if config.validate().is_err() {
            return false;
        }
        self.insert_valid(wall, config)
    }

    fn remove_valid(&mut self, wall: Wall, config: &Config) -> bool {
        let Some(bit) = Self::anchor_bit(wall, config) else {
            return false;
        };
        let occupied = match wall.orientation {
            Orientation::Horizontal => self.horizontal & bit != 0,
            Orientation::Vertical => self.vertical & bit != 0,
        };
        if !occupied {
            return false;
        }
        match wall.orientation {
            Orientation::Horizontal => self.horizontal &= !bit,
            Orientation::Vertical => self.vertical &= !bit,
        }
        true
    }

    /// Remove a valid anchor, returning false without mutation when the anchor
    /// is out of bounds or was not present.
    pub fn remove(&mut self, wall: Wall, config: &Config) -> bool {
        if config.validate().is_err() {
            return false;
        }
        self.remove_valid(wall, config)
    }

    #[must_use]
    pub fn count(self) -> u32 {
        self.horizontal.count_ones() + self.vertical.count_ones()
    }

    fn walls(self, config: &Config) -> Vec<Wall> {
        let mut result = Vec::with_capacity(self.count() as usize);
        for orientation in [Orientation::Horizontal, Orientation::Vertical] {
            let bits = match orientation {
                Orientation::Horizontal => self.horizontal,
                Orientation::Vertical => self.vertical,
            };
            for anchor in 0..config.anchors_per_axis() {
                if bits & (1_u64 << anchor) != 0 {
                    result.push(Wall {
                        orientation,
                        r: (anchor / (config.columns as usize - 1)) as u8,
                        c: (anchor % (config.columns as usize - 1)) as u8,
                    });
                }
            }
        }
        result
    }
}

#[derive(Debug, Clone, Copy)]
struct Board {
    walls: WallBits,
    /// Bit `square(r,c)` means the downward edge from `(r,c)` is blocked.
    blocked_down: u128,
    /// Bit `square(r,c)` means the rightward edge from `(r,c)` is blocked.
    blocked_right: u128,
}

impl Board {
    fn from_position(config: &Config, position: &Position) -> Result<Self> {
        let mut walls = WallBits::default();
        for text in &position.walls {
            let wall = Wall::parse(config, text)?;
            if walls.contains_valid(wall, config) {
                return Err(NormalDuelError::InvalidWall);
            }
            let inserted = walls.insert_valid(wall, config);
            debug_assert!(inserted, "parsed wall is in bounds");
        }
        Ok(Self::from_walls(config, walls))
    }

    fn from_walls(config: &Config, walls: WallBits) -> Self {
        let mut board = Self {
            walls,
            blocked_down: 0,
            blocked_right: 0,
        };
        board.rebuild_blocked_edges(config);
        board
    }

    /// Rebuild the edge bitboards straight off the anchor bits. This is on the
    /// hot path via [`Board::from_position`], so it iterates set bits rather
    /// than materializing the wall list.
    fn rebuild_blocked_edges(&mut self, config: &Config) {
        self.blocked_down = 0;
        self.blocked_right = 0;
        let anchor_columns = config.columns.saturating_sub(1) as usize;
        if anchor_columns == 0 {
            return;
        }
        // `walls()` only ever looked at anchors below `anchors_per_axis()`;
        // keep ignoring any stray high bits so behaviour is unchanged.
        let anchors = config.anchors_per_axis();
        let valid = if anchors >= 64 {
            u64::MAX
        } else {
            (1_u64 << anchors) - 1
        };
        for orientation in [Orientation::Horizontal, Orientation::Vertical] {
            let mut bits = valid
                & match orientation {
                    Orientation::Horizontal => self.walls.horizontal,
                    Orientation::Vertical => self.walls.vertical,
                };
            while bits != 0 {
                let anchor = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                self.add_wall_edges(
                    config,
                    Wall {
                        orientation,
                        r: (anchor / anchor_columns) as u8,
                        c: (anchor % anchor_columns) as u8,
                    },
                );
            }
        }
    }

    fn add_wall_edges(&mut self, config: &Config, wall: Wall) {
        match wall.orientation {
            Orientation::Horizontal => {
                self.blocked_down |= 1_u128
                    << square(
                        config,
                        Coord {
                            r: wall.r,
                            c: wall.c,
                        },
                    );
                self.blocked_down |= 1_u128
                    << square(
                        config,
                        Coord {
                            r: wall.r,
                            c: wall.c + 1,
                        },
                    );
            }
            Orientation::Vertical => {
                self.blocked_right |= 1_u128
                    << square(
                        config,
                        Coord {
                            r: wall.r,
                            c: wall.c,
                        },
                    );
                self.blocked_right |= 1_u128
                    << square(
                        config,
                        Coord {
                            r: wall.r + 1,
                            c: wall.c,
                        },
                    );
            }
        }
    }

    #[must_use]
    fn edge_blocked(self, config: &Config, from: Coord, to: Coord) -> bool {
        debug_assert!(in_bounds(config, from) && in_bounds(config, to));
        debug_assert_eq!(manhattan(from, to), 1);
        if from.r != to.r {
            let upper = Coord {
                r: from.r.min(to.r),
                c: from.c,
            };
            self.blocked_down & (1_u128 << square(config, upper)) != 0
        } else {
            let left = Coord {
                r: from.r,
                c: from.c.min(to.c),
            };
            self.blocked_right & (1_u128 << square(config, left)) != 0
        }
    }

    fn has_path(self, config: &Config, from: Coord, goal_row: u8) -> bool {
        let columns = config.columns as usize;
        let valid = (1_u128 << config.cells()) - 1;
        let mut right_column = 0_u128;
        for row in 0..config.rows {
            right_column |= 1_u128
                << square(
                    config,
                    Coord {
                        r: row,
                        c: config.columns - 1,
                    },
                );
        }
        let goal = ((1_u128 << columns) - 1) << (usize::from(goal_row) * columns);
        let mut seen = 1_u128 << square(config, from);
        let mut frontier = seen;
        loop {
            if frontier & goal != 0 {
                return true;
            }
            let upward = (frontier >> columns) & !self.blocked_down;
            let downward = ((frontier & !self.blocked_down) << columns) & valid;
            let leftward = (frontier >> 1) & !right_column & !self.blocked_right;
            let rightward = ((frontier & !right_column & !self.blocked_right) << 1) & valid;
            frontier = (upward | downward | leftward | rightward) & !seen;
            if frontier == 0 {
                return false;
            }
            seen |= frontier;
        }
    }

    fn shortest_distance(self, config: &Config, from: Coord, goal_row: u8) -> Option<u8> {
        let columns = config.columns as usize;
        let valid = (1_u128 << config.cells()) - 1;
        let mut right_column = 0_u128;
        for row in 0..config.rows {
            right_column |= 1_u128
                << square(
                    config,
                    Coord {
                        r: row,
                        c: config.columns - 1,
                    },
                );
        }
        let goal = ((1_u128 << columns) - 1) << (usize::from(goal_row) * columns);
        let mut seen = 1_u128 << square(config, from);
        let mut frontier = seen;
        let mut distance = 0_u8;
        loop {
            if frontier & goal != 0 {
                return Some(distance);
            }
            let upward = (frontier >> columns) & !self.blocked_down;
            let downward = ((frontier & !self.blocked_down) << columns) & valid;
            let leftward = (frontier >> 1) & !right_column & !self.blocked_right;
            let rightward = ((frontier & !right_column & !self.blocked_right) << 1) & valid;
            frontier = (upward | downward | leftward | rightward) & !seen;
            if frontier == 0 {
                return None;
            }
            seen |= frontier;
            distance += 1;
        }
    }

    fn candidate_geometry_ok(self, config: &Config, candidate: Wall) -> bool {
        if self.walls.contains_valid(candidate, config) {
            return false;
        }
        let adjacent_before = match candidate.orientation {
            Orientation::Horizontal => candidate.c.checked_sub(1).map(|c| Wall { c, ..candidate }),
            Orientation::Vertical => candidate.r.checked_sub(1).map(|r| Wall { r, ..candidate }),
        };
        let adjacent_after = match candidate.orientation {
            Orientation::Horizontal if candidate.c + 1 < config.columns - 1 => Some(Wall {
                c: candidate.c + 1,
                ..candidate
            }),
            Orientation::Vertical if candidate.r + 1 < config.rows - 1 => Some(Wall {
                r: candidate.r + 1,
                ..candidate
            }),
            _ => None,
        };
        let cross = Wall {
            orientation: match candidate.orientation {
                Orientation::Horizontal => Orientation::Vertical,
                Orientation::Vertical => Orientation::Horizontal,
            },
            ..candidate
        };
        !adjacent_before.is_some_and(|wall| self.walls.contains_valid(wall, config))
            && !adjacent_after.is_some_and(|wall| self.walls.contains_valid(wall, config))
            && !self.walls.contains_valid(cross, config)
    }

    fn candidate_is_legal(self, config: &Config, position: &Position, candidate: Wall) -> bool {
        self.candidate_is_legal_parts(
            config,
            position.pawns,
            position.stock,
            position.turn,
            candidate,
        )
    }

    fn candidate_is_legal_parts(
        self,
        config: &Config,
        pawns: Players<Coord>,
        stock: Players<u64>,
        turn: Player,
        candidate: Wall,
    ) -> bool {
        if stock.get(turn) == &0 || !self.candidate_geometry_ok(config, candidate) {
            return false;
        }
        let mut next = self;
        let inserted = next.walls.insert_valid(candidate, config);
        debug_assert!(inserted, "legal candidate is in bounds");
        next.add_wall_edges(config, candidate);
        next.has_path(config, pawns.a, config.goal_rows.a)
            && next.has_path(config, pawns.b, config.goal_rows.b)
    }
}

fn offset(coord: Coord, dr: i8, dc: i8) -> Option<Coord> {
    let r = i16::from(coord.r) + i16::from(dr);
    let c = i16::from(coord.c) + i16::from(dc);
    if r < 0 || c < 0 || r > i16::from(u8::MAX) || c > i16::from(u8::MAX) {
        None
    } else {
        Some(Coord {
            r: r as u8,
            c: c as u8,
        })
    }
}

fn manhattan(left: Coord, right: Coord) -> u8 {
    left.r.abs_diff(right.r) + left.c.abs_diff(right.c)
}

/// Validate and canonicalize a Position. Wall strings are returned in policy
/// order (all H anchors, then V anchors), exactly matching the JS engine.
pub fn normalize_position(config: &Config, input: &Position) -> Result<Position> {
    config.validate()?;
    if !in_bounds(config, input.pawns.a)
        || !in_bounds(config, input.pawns.b)
        || input.pawns.a == input.pawns.b
        || input.stock.a > config.initial_stock.a
        || input.stock.b > config.initial_stock.b
    {
        return Err(NormalDuelError::InvalidPosition);
    }
    let board = Board::from_position(config, input)?;
    let mut walls = board.walls.walls(config);
    walls.sort_by_key(|wall| wall.policy_code(config));
    for wall in &walls {
        if !board.candidate_geometry_ok_except_self(config, *wall) {
            return Err(NormalDuelError::InvalidWallGeometry);
        }
    }
    if !board.has_path(config, input.pawns.a, config.goal_rows.a)
        || !board.has_path(config, input.pawns.b, config.goal_rows.b)
    {
        return Err(NormalDuelError::PathBlocked);
    }
    Ok(Position {
        pawns: input.pawns,
        walls: walls.into_iter().map(Wall::text).collect(),
        stock: input.stock,
        turn: input.turn,
    })
}

/// Policy codes on the canonical 9x9 board: 81 pawn squares plus 2 x 64 wall
/// anchors. A caller buffer of this length can never be overrun by
/// [`legal_action_codes_fast`].
pub const MAX_POLICY_CODES: usize = 209;

/// Instrumentation for the shortest-path prefilter, so callers (and the
/// benchmark) can report how much search the filter actually skipped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FastLegalStats {
    /// Wall anchors examined at all (0 when the mover has no stock left).
    pub candidates: u32,
    /// Candidates that survived the geometry test and so reached the filter.
    pub geometry_ok: u32,
    /// Candidates that cut at least one player's stored shortest path.
    pub path_touching: u32,
    /// Real `has_path` calls run, excluding the two per-node path searches.
    pub searches: u32,
}

impl Board {
    /// One shortest path from `from` to `goal_row`, returned as the edges it
    /// uses in the `(down, right)` mask form [`Board::edge_blocked`] reads.
    ///
    /// Returns `None` when no path exists. Fixed-size stack scratch only: the
    /// largest supported board has 81 squares.
    fn shortest_path_edges(
        self,
        config: &Config,
        from: Coord,
        goal_row: u8,
    ) -> Option<(u128, u128)> {
        let columns = config.columns as usize;
        let start = square(config, from);
        let mut previous = [u8::MAX; 81];
        let mut queue = [0_u8; 81];
        let mut tail = 1_usize;
        let mut head = 0_usize;
        queue[0] = start as u8;
        let mut seen = 1_u128 << start;
        let mut reached = None;
        while head < tail {
            let current = usize::from(queue[head]);
            head += 1;
            let coord = Coord {
                r: (current / columns) as u8,
                c: (current % columns) as u8,
            };
            if coord.r == goal_row {
                reached = Some(current);
                break;
            }
            for (dr, dc) in DIRECTIONS {
                let Some(next) = offset(coord, dr, dc) else {
                    continue;
                };
                if !in_bounds(config, next) {
                    continue;
                }
                let index = square(config, next);
                if seen & (1_u128 << index) != 0 || self.edge_blocked(config, coord, next) {
                    continue;
                }
                seen |= 1_u128 << index;
                previous[index] = current as u8;
                queue[tail] = index as u8;
                tail += 1;
            }
        }
        let mut node = reached?;
        let (mut down, mut right) = (0_u128, 0_u128);
        while node != start {
            let parent = usize::from(previous[node]);
            if node / columns == parent / columns {
                right |= 1_u128 << node.min(parent);
            } else {
                down |= 1_u128 << node.min(parent);
            }
            node = parent;
        }
        Some((down, right))
    }

    /// The edges a candidate wall would block, in the same mask form
    /// [`Board::shortest_path_edges`] returns. Mirrors [`Board::add_wall_edges`].
    fn candidate_edge_mask(config: &Config, candidate: Wall) -> (u128, u128) {
        match candidate.orientation {
            Orientation::Horizontal => (
                (1_u128 << square(config, candidate.coord()))
                    | (1_u128
                        << square(
                            config,
                            Coord {
                                r: candidate.r,
                                c: candidate.c + 1,
                            },
                        )),
                0,
            ),
            Orientation::Vertical => (
                0,
                (1_u128 << square(config, candidate.coord()))
                    | (1_u128
                        << square(
                            config,
                            Coord {
                                r: candidate.r + 1,
                                c: candidate.c,
                            },
                        )),
            ),
        }
    }

    /// Write every legal policy code for this position into `out`, in
    /// ascending code order, allocating nothing.
    ///
    /// The wall loop uses an exact prefilter rather than the two `has_path`
    /// calls per candidate that [`Board::candidate_is_legal`] runs. Each
    /// player's current shortest path is computed once; a candidate that
    /// blocks no edge of that path leaves the path intact, so a path provably
    /// still exists and no search is needed. Only a candidate that cuts a
    /// stored path is searched, and only for the player whose path it cut.
    ///
    /// Returns the number of codes written. `out` must hold at least
    /// `config.policy_size()` entries.
    fn legal_codes_into(
        self,
        config: &Config,
        pawns: Players<Coord>,
        stock: Players<u64>,
        turn: Player,
        out: &mut [u16],
        stats: &mut FastLegalStats,
    ) -> usize {
        let mut count = 0_usize;

        // Pawn codes are all below `cells` and wall codes all at or above it,
        // so emitting every pawn destination first keeps `out` ascending.
        let mover = *pawns.get(turn);
        let opponent = *pawns.get(turn.other());
        let mut squares = [0_u16; 8];
        let mut found = 0_usize;
        for (dr, dc) in DIRECTIONS {
            let Some(adjacent) = offset(mover, dr, dc) else {
                continue;
            };
            if !in_bounds(config, adjacent) || self.edge_blocked(config, mover, adjacent) {
                continue;
            }
            if adjacent != opponent {
                squares[found] = square(config, adjacent) as u16;
                found += 1;
                continue;
            }
            for (exit_r, exit_c) in DIRECTIONS {
                let Some(destination) = offset(opponent, exit_r, exit_c) else {
                    continue;
                };
                if destination == mover
                    || !in_bounds(config, destination)
                    || self.edge_blocked(config, opponent, destination)
                {
                    continue;
                }
                let code = square(config, destination) as u16;
                if !squares[..found].contains(&code) {
                    squares[found] = code;
                    found += 1;
                }
            }
        }
        squares[..found].sort_unstable();
        out[..found].copy_from_slice(&squares[..found]);
        count += found;

        if *stock.get(turn) == 0 {
            return count;
        }
        let (Some((down_a, right_a)), Some((down_b, right_b))) = (
            self.shortest_path_edges(config, pawns.a, config.goal_rows.a),
            self.shortest_path_edges(config, pawns.b, config.goal_rows.b),
        ) else {
            // A player is already walled in, so every candidate is illegal —
            // the same answer `candidate_is_legal` gives.
            return count;
        };

        // `H` anchors then `V` anchors, each row-major: already policy order.
        for orientation in [Orientation::Horizontal, Orientation::Vertical] {
            for r in 0..config.rows - 1 {
                for c in 0..config.columns - 1 {
                    let candidate = Wall { orientation, r, c };
                    stats.candidates += 1;
                    if !self.candidate_geometry_ok(config, candidate) {
                        continue;
                    }
                    stats.geometry_ok += 1;
                    let (down, right) = Self::candidate_edge_mask(config, candidate);
                    let cuts_a = (down & down_a) | (right & right_a) != 0;
                    let cuts_b = (down & down_b) | (right & right_b) != 0;
                    if cuts_a || cuts_b {
                        stats.path_touching += 1;
                        let mut next = self;
                        let inserted = next.walls.insert_valid(candidate, config);
                        debug_assert!(inserted, "geometry-checked candidate is in bounds");
                        next.add_wall_edges(config, candidate);
                        if cuts_a {
                            stats.searches += 1;
                            if !next.has_path(config, pawns.a, config.goal_rows.a) {
                                continue;
                            }
                        }
                        if cuts_b {
                            stats.searches += 1;
                            if !next.has_path(config, pawns.b, config.goal_rows.b) {
                                continue;
                            }
                        }
                    }
                    out[count] = candidate.policy_code(config) as u16;
                    count += 1;
                }
            }
        }
        count
    }

    fn candidate_geometry_ok_except_self(self, config: &Config, wall: Wall) -> bool {
        let mut without = self;
        let removed = without.walls.remove_valid(wall, config);
        debug_assert!(removed, "normalized wall is present");
        without.candidate_geometry_ok(config, wall)
    }
}

/// True when one specified in-bounds orthogonal edge is blocked by walls.
pub fn edge_blocked(config: &Config, from: Coord, to: Coord, walls: &[String]) -> Result<bool> {
    config.validate()?;
    if !in_bounds(config, from) || !in_bounds(config, to) || manhattan(from, to) != 1 {
        return Err(NormalDuelError::InvalidEdge);
    }
    let position = Position {
        pawns: Players {
            a: config.start.a,
            b: config.start.b,
        },
        walls: walls.to_vec(),
        stock: Players {
            a: config.initial_stock.a,
            b: config.initial_stock.b,
        },
        turn: config.first_player,
    };
    let board = Board::from_position(config, &position)?;
    Ok(board.edge_blocked(config, from, to))
}

pub fn has_path(config: &Config, from: Coord, goal_row: u8, walls: &[String]) -> Result<bool> {
    config.validate()?;
    if !in_bounds(config, from) || goal_row >= config.rows {
        return Err(NormalDuelError::InvalidPosition);
    }
    let position = Position {
        pawns: Players {
            a: config.start.a,
            b: config.start.b,
        },
        walls: walls.to_vec(),
        stock: Players {
            a: config.initial_stock.a,
            b: config.initial_stock.b,
        },
        turn: config.first_player,
    };
    let board = Board::from_position(config, &position)?;
    Ok(board.has_path(config, from, goal_row))
}

/// Wall-only shortest distance to a goal edge. Pawn occupancy is deliberately
/// ignored, matching the legality-path rule.
pub fn shortest_distance(
    config: &Config,
    from: Coord,
    goal_row: u8,
    walls: &[String],
) -> Result<Option<u8>> {
    config.validate()?;
    if !in_bounds(config, from) || goal_row >= config.rows {
        return Err(NormalDuelError::InvalidPosition);
    }
    let position = Position {
        pawns: Players {
            a: config.start.a,
            b: config.start.b,
        },
        walls: walls.to_vec(),
        stock: Players {
            a: config.initial_stock.a,
            b: config.initial_stock.b,
        },
        turn: config.first_player,
    };
    let board = Board::from_position(config, &position)?;
    Ok(board.shortest_distance(config, from, goal_row))
}

fn position_key_normalized(config: &Config, position: &Position) -> String {
    let board = Board::from_position(config, position).expect("normalized walls remain parseable");
    let mut horizontal = Vec::new();
    let mut vertical = Vec::new();
    for wall in board.walls.walls(config) {
        match wall.orientation {
            Orientation::Horizontal => horizontal.push(wall.anchor_index(config)),
            Orientation::Vertical => vertical.push(wall.anchor_index(config)),
        }
    }
    horizontal.sort_unstable();
    vertical.sort_unstable();
    json!([
        config.ruleset,
        config.rows,
        config.columns,
        square(config, config.start.a),
        square(config, config.start.b),
        config.goal_rows.a,
        config.goal_rows.b,
        config.initial_stock.a,
        config.initial_stock.b,
        config.jump_rule,
        config.repetition_threshold,
        config.ply_cap,
        config.first_player,
        square(config, position.pawns.a),
        square(config, position.pawns.b),
        horizontal,
        vertical,
        position.stock.a,
        position.stock.b,
        position.turn,
    ])
    .to_string()
}

pub fn position_key(config: &Config, position: &Position) -> Result<String> {
    let normalized = normalize_position(config, position)?;
    Ok(position_key_normalized(config, &normalized))
}

pub fn compare_position_keys(left: &str, right: &str) -> Ordering {
    left.as_bytes().cmp(right.as_bytes())
}

pub fn legal_pawn_destinations(config: &Config, position: &Position) -> Result<Vec<Coord>> {
    let normalized = normalize_position(config, position)?;
    let board = Board::from_position(config, &normalized)?;
    Ok(legal_pawn_destinations_board(config, &normalized, board))
}

fn legal_pawn_destinations_board(config: &Config, position: &Position, board: Board) -> Vec<Coord> {
    let mover = *position.pawns.get(position.turn);
    let opponent = *position.pawns.get(position.turn.other());
    let mut destinations = Vec::with_capacity(6);
    for (dr, dc) in DIRECTIONS {
        let Some(adjacent) = offset(mover, dr, dc) else {
            continue;
        };
        if !in_bounds(config, adjacent) || board.edge_blocked(config, mover, adjacent) {
            continue;
        }
        if adjacent != opponent {
            destinations.push(adjacent);
            continue;
        }
        for (exit_r, exit_c) in DIRECTIONS {
            let Some(destination) = offset(opponent, exit_r, exit_c) else {
                continue;
            };
            if destination == mover
                || !in_bounds(config, destination)
                || board.edge_blocked(config, opponent, destination)
            {
                continue;
            }
            if !destinations.contains(&destination) {
                destinations.push(destination);
            }
        }
    }
    destinations.sort_by_key(|coord| square(config, *coord));
    destinations
}

pub fn is_legal_wall(config: &Config, position: &Position, wall_text: &str) -> Result<bool> {
    let normalized = normalize_position(config, position)?;
    let board = Board::from_position(config, &normalized)?;
    let candidate = match Wall::parse(config, wall_text) {
        Ok(wall) => wall,
        Err(_) => return Ok(false),
    };
    Ok(board.candidate_is_legal(config, &normalized, candidate))
}

pub fn legal_position_actions(config: &Config, position: &Position) -> Result<Vec<Action>> {
    let normalized = normalize_position(config, position)?;
    let board = Board::from_position(config, &normalized)?;
    let mut actions: Vec<Action> = legal_pawn_destinations_board(config, &normalized, board)
        .into_iter()
        .map(|to| Action::Pawn { to })
        .collect();
    if *normalized.stock.get(normalized.turn) > 0 {
        for orientation in [Orientation::Horizontal, Orientation::Vertical] {
            for r in 0..config.rows - 1 {
                for c in 0..config.columns - 1 {
                    let wall = Wall { orientation, r, c };
                    if board.candidate_is_legal(config, &normalized, wall) {
                        actions.push(Action::Wall { wall: wall.text() });
                    }
                }
            }
        }
    }
    actions.sort_by_key(|action| {
        action
            .policy_code_validated(config)
            .expect("generated action is valid")
    });
    Ok(actions)
}

pub fn legal_position_action_codes(config: &Config, position: &Position) -> Result<Vec<usize>> {
    legal_position_actions(config, position)?
        .iter()
        .map(|action| action.policy_code_validated(config))
        .collect()
}

pub fn legal_position_action_mask(config: &Config, position: &Position) -> Result<Vec<u8>> {
    let mut mask = vec![0; policy_size(config)?];
    for code in legal_position_action_codes(config, position)? {
        mask[code] = 1;
    }
    Ok(mask)
}

/// Allocation-free legal-action generation for an already normalized position.
///
/// Writes the legal policy codes into `out` in ascending code order and
/// returns how many were written; `out` must hold at least
/// `config.policy_size()` entries ([`MAX_POLICY_CODES`] always suffices).
/// Nothing is heap-allocated: no wall text is formatted and no intermediate
/// vector is built or sorted.
///
/// The result set is identical to [`legal_position_action_codes`]; only the
/// amount of pathfinding differs. See [`Board::legal_codes_into`] for why the
/// shortest-path prefilter is exact rather than a heuristic.
///
/// `position` is expected to already satisfy [`normalize_position`]. The
/// configuration and buffer are still checked, and duplicate wall text is
/// still rejected, but position invariants are not re-derived here — that is
/// the allocation this function exists to avoid.
pub fn legal_action_codes_fast(
    config: &Config,
    position: &Position,
    out: &mut [u16],
) -> Result<usize> {
    Ok(legal_action_codes_fast_stats(config, position, out)?.0)
}

/// [`legal_action_codes_fast`] plus prefilter instrumentation.
pub fn legal_action_codes_fast_stats(
    config: &Config,
    position: &Position,
    out: &mut [u16],
) -> Result<(usize, FastLegalStats)> {
    config.validate()?;
    if out.len() < config.policy_size() {
        return Err(NormalDuelError::InvalidActionCode);
    }
    if !in_bounds(config, position.pawns.a) || !in_bounds(config, position.pawns.b) {
        return Err(NormalDuelError::InvalidPosition);
    }
    let board = Board::from_position(config, position)?;
    let mut stats = FastLegalStats::default();
    let count = board.legal_codes_into(
        config,
        position.pawns,
        position.stock,
        position.turn,
        out,
        &mut stats,
    );
    Ok((count, stats))
}

/// Plane order of the Stage 4 network input, matching `NN_PLANE_LAYOUT` in
/// `js/normal-duel-nn-encoding.mjs` index for index.
pub const NN_PLANE_LAYOUT: [&str; 8] = [
    "mover_pawn",
    "opponent_pawn",
    "wall_horizontal",
    "wall_vertical",
    "mover_stock",
    "opponent_stock",
    "goal_proximity",
    "side_to_move",
];

pub const NN_INPUT_PLANES: usize = NN_PLANE_LAYOUT.len();

/// Encode a game state as `NN_INPUT_PLANES * rows * columns` floats, a port of
/// `encodeState` in `js/normal-duel-nn-encoding.mjs`.
///
/// Only `state.position` is read, so this is a thin wrapper over
/// [`encode_position_into`]. Unlike the JavaScript, the state is not
/// re-validated here: validation allocates, and this is the hot path. Callers
/// must pass a state they already trust.
pub fn encode_state_into(config: &Config, state: &GameState, out: &mut [f32]) -> Result<()> {
    encode_position_into(config, &state.position, out)
}

/// The [`encode_state_into`] body, operating on the position directly.
///
/// The frame is absolute: every plane is indexed by engine `(r, c)` with no
/// mirroring, because the policy codes it must line up with are absolute.
/// Planes 0/1 and 4/5 are mover-relative only in the sense of *which* plane a
/// value lands in — no coordinate is ever moved. See the module comment in
/// `js/normal-duel-nn-encoding.mjs`.
///
/// Writes into the caller's buffer, which must be exactly
/// `NN_INPUT_PLANES * config.cells()` long, and allocates nothing.
pub fn encode_position_into(config: &Config, position: &Position, out: &mut [f32]) -> Result<()> {
    config.validate()?;
    // Stage 4's architecture was measured on the canonical 9x9 board only,
    // matching the JS `unsupported_board` guard.
    if config.rows != 9 || config.columns != 9 {
        return Err(NormalDuelError::InvalidConfig);
    }
    let cells = config.cells();
    if out.len() != NN_INPUT_PLANES * cells {
        return Err(NormalDuelError::InvalidPosition);
    }
    if !in_bounds(config, position.pawns.a) || !in_bounds(config, position.pawns.b) {
        return Err(NormalDuelError::InvalidPosition);
    }
    let board = Board::from_position(config, position)?;
    encode_board_into(
        config,
        &board,
        position.pawns,
        position.stock,
        position.turn,
        out,
    );
    Ok(())
}

fn encode_board_into(
    config: &Config,
    board: &Board,
    pawns: Players<Coord>,
    stock: Players<u64>,
    turn: Player,
    out: &mut [f32],
) {
    let cells = config.cells();
    let columns = config.columns as usize;
    let opponent = turn.other();

    // The four sparse planes are the caller's buffer, not a fresh array.
    out[..4 * cells].fill(0.0);
    out[square(config, *pawns.get(turn))] = 1.0;
    out[cells + square(config, *pawns.get(opponent))] = 1.0;

    // `blocked_down`/`blocked_right` are exactly the engine's `edgeBlocked`
    // answers for the down and right edge of each cell, which is what the JS
    // reads back out of the engine one edge at a time.
    let (horizontal, vertical) = (2 * cells, 3 * cells);
    for r in 0..config.rows {
        for c in 0..config.columns {
            let cell = square(config, Coord { r, c });
            if r + 1 < config.rows && board.blocked_down & (1_u128 << cell) != 0 {
                out[horizontal + cell] = 1.0;
            }
            if c + 1 < config.columns && board.blocked_right & (1_u128 << cell) != 0 {
                out[vertical + cell] = 1.0;
            }
        }
    }

    // Divide in f64 and round once on store, the way a JS `Float32Array`
    // assignment does, so the bit patterns agree exactly.
    let scale = |player: Player| -> f32 {
        let initial = *config.initial_stock.get(player);
        if initial > 0 {
            (*stock.get(player) as f64 / initial as f64) as f32
        } else {
            0.0
        }
    };
    out[4 * cells..5 * cells].fill(scale(turn));
    out[5 * cells..6 * cells].fill(scale(opponent));

    let goal_row = *config.goal_rows.get(turn);
    let span = f64::from(config.rows - 1);
    let base = 6 * cells;
    for r in 0..config.rows {
        let value = ((span - f64::from(r.abs_diff(goal_row))) / span) as f32;
        let row = base + usize::from(r) * columns;
        out[row..row + columns].fill(value);
    }

    // Whose turn it is is not recoverable from an absolute frame, so state it.
    out[7 * cells..8 * cells].fill(if turn == Player::A { 1.0 } else { 0.0 });
}

pub fn adjudicate(
    config: &Config,
    goal_winner: Option<Player>,
    resulting_position_count: u64,
    resulting_ply: u64,
) -> Result<Outcome> {
    config.validate()?;
    if resulting_position_count == 0
        || resulting_position_count > MAX_JS_SAFE_INTEGER
        || resulting_ply == 0
        || resulting_ply > MAX_JS_SAFE_INTEGER
    {
        return Err(NormalDuelError::InvalidAdjudication);
    }
    if let Some(winner) = goal_winner {
        return Ok(Outcome::Win {
            winner,
            reason: GoalReason::Goal,
        });
    }
    if resulting_position_count >= u64::from(config.repetition_threshold) {
        return Ok(Outcome::Draw {
            reason: DrawReason::ThreefoldRepetition,
        });
    }
    if resulting_ply >= config.ply_cap {
        return Ok(Outcome::Draw {
            reason: DrawReason::PlyCap,
        });
    }
    Ok(Outcome::Ongoing)
}

pub fn create_initial_state(config: &Config) -> Result<GameState> {
    config.validate()?;
    let position = normalize_position(
        config,
        &Position {
            pawns: config.start,
            walls: Vec::new(),
            stock: config.initial_stock,
            turn: config.first_player,
        },
    )?;
    let position_key = position_key_normalized(config, &position);
    Ok(GameState {
        position,
        position_key: position_key.clone(),
        ply: 0,
        history_start_ply: 0,
        repetition_counts: vec![RepetitionCount {
            position_key,
            count: 1,
        }],
        outcome: Outcome::Ongoing,
    })
}

fn parse_position_key(config: &Config, key: &str) -> Result<Position> {
    let value = serde_json::from_str::<Value>(key).map_err(|_| NormalDuelError::InvalidState)?;
    let fields = value.as_array().ok_or(NormalDuelError::InvalidState)?;
    if fields.len() != 20
        || fields[0].as_str() != Some(RULESET)
        || fields[1].as_u64() != Some(u64::from(config.rows))
        || fields[2].as_u64() != Some(u64::from(config.columns))
        || fields[3].as_u64() != Some(square(config, config.start.a) as u64)
        || fields[4].as_u64() != Some(square(config, config.start.b) as u64)
        || fields[5].as_u64() != Some(u64::from(config.goal_rows.a))
        || fields[6].as_u64() != Some(u64::from(config.goal_rows.b))
        || fields[7].as_u64() != Some(config.initial_stock.a)
        || fields[8].as_u64() != Some(config.initial_stock.b)
        || fields[9].as_str() != Some(JUMP_RULE)
        || fields[10].as_u64() != Some(u64::from(config.repetition_threshold))
        || fields[11].as_u64() != Some(config.ply_cap)
        || fields[12].as_str() != Some(config.first_player.as_str())
    {
        return Err(NormalDuelError::InvalidState);
    }
    let pawn = |index: usize| -> Result<Coord> {
        let square = usize::try_from(
            fields[index]
                .as_u64()
                .ok_or(NormalDuelError::InvalidState)?,
        )
        .map_err(|_| NormalDuelError::InvalidState)?;
        if square >= config.cells() {
            return Err(NormalDuelError::InvalidState);
        }
        Ok(Coord {
            r: (square / config.columns as usize) as u8,
            c: (square % config.columns as usize) as u8,
        })
    };
    let anchors = |index: usize, orientation: Orientation| -> Result<Vec<String>> {
        let values = fields[index]
            .as_array()
            .ok_or(NormalDuelError::InvalidState)?;
        let mut prior = None;
        let mut result = Vec::with_capacity(values.len());
        for value in values {
            let anchor = usize::try_from(value.as_u64().ok_or(NormalDuelError::InvalidState)?)
                .map_err(|_| NormalDuelError::InvalidState)?;
            if anchor >= config.anchors_per_axis() || prior.is_some_and(|last| anchor <= last) {
                return Err(NormalDuelError::InvalidState);
            }
            prior = Some(anchor);
            result.push(
                Wall {
                    orientation,
                    r: (anchor / (config.columns as usize - 1)) as u8,
                    c: (anchor % (config.columns as usize - 1)) as u8,
                }
                .text(),
            );
        }
        Ok(result)
    };
    let a = pawn(13)?;
    let b = pawn(14)?;
    let mut walls = anchors(15, Orientation::Horizontal)?;
    walls.extend(anchors(16, Orientation::Vertical)?);
    let stock = Players {
        a: fields[17].as_u64().ok_or(NormalDuelError::InvalidState)?,
        b: fields[18].as_u64().ok_or(NormalDuelError::InvalidState)?,
    };
    let turn = match fields[19].as_str() {
        Some("A") => Player::A,
        Some("B") => Player::B,
        _ => return Err(NormalDuelError::InvalidState),
    };
    let position = normalize_position(
        config,
        &Position {
            pawns: Players { a, b },
            walls,
            stock,
            turn,
        },
    )?;
    if position_key_normalized(config, &position) != key {
        return Err(NormalDuelError::InvalidState);
    }
    Ok(position)
}

/// Validate an externally supplied complete state. This deliberately checks
/// enough history invariants to reject forged wall resets or incompatible
/// repetition windows even though it does not retain every historical ply.
pub fn validate_state(config: &Config, input: &GameState) -> Result<GameState> {
    config.validate()?;
    if input.ply > MAX_JS_SAFE_INTEGER || input.history_start_ply > MAX_JS_SAFE_INTEGER {
        return Err(NormalDuelError::InvalidState);
    }
    let position = normalize_position(config, &input.position)?;
    let key = position_key_normalized(config, &position);
    let initial_state = if input.ply <= 1 || input.history_start_ply == 0 {
        Some(create_initial_state(config)?)
    } else {
        None
    };
    if input.position_key != key
        || input.history_start_ply > input.ply
        || position.turn != config.expected_turn(input.ply)
        || input.repetition_counts.is_empty()
    {
        return Err(NormalDuelError::InvalidState);
    }
    let spent = Players {
        a: config.initial_stock.a - position.stock.a,
        b: config.initial_stock.b - position.stock.b,
    };
    if spent
        .a
        .checked_add(spent.b)
        .ok_or(NormalDuelError::InvalidState)?
        != position.walls.len() as u64
        || position.walls.len() as u64 > input.ply
    {
        return Err(NormalDuelError::InvalidState);
    }
    if (position.walls.is_empty()) != (input.history_start_ply == 0)
        || input.history_start_ply < position.walls.len() as u64
    {
        return Err(NormalDuelError::InvalidState);
    }
    let completed_actions = config.completed_actions(input.ply);
    let actions_at_reset = config.completed_actions(input.history_start_ply);
    if spent.a > actions_at_reset.a || spent.b > actions_at_reset.b {
        return Err(NormalDuelError::InvalidState);
    }
    if input.history_start_ply > 0 {
        let prior_mover = config.expected_turn(input.history_start_ply - 1);
        if *spent.get(prior_mover) == 0 {
            return Err(NormalDuelError::InvalidState);
        }
    }
    if input.ply == 1 {
        let initial = initial_state
            .as_ref()
            .expect("ply-one validation creates the initial state");
        let mover = config.first_player;
        let opponent = mover.other();
        let first_pawn_move = position.walls.is_empty()
            && position.pawns.get(opponent) == initial.position.pawns.get(opponent)
            && legal_pawn_destinations_board(
                config,
                &initial.position,
                Board::from_position(config, &initial.position)?,
            )
            .contains(position.pawns.get(mover));
        let expected_mover_stock = config.initial_stock.get(mover).checked_sub(1);
        let first_wall_move = position.walls.len() == 1
            && position.pawns == initial.position.pawns
            && expected_mover_stock == Some(*position.stock.get(mover))
            && position.stock.get(opponent) == config.initial_stock.get(opponent)
            && is_legal_wall(config, &initial.position, &position.walls[0])?;
        if !first_pawn_move && !first_wall_move {
            return Err(NormalDuelError::InvalidState);
        }
    }
    for player in [Player::A, Player::B] {
        let distance_from_start = u64::from(manhattan(
            *config.start.get(player),
            *position.pawns.get(player),
        ));
        if distance_from_start > 2 * *completed_actions.get(player) {
            return Err(NormalDuelError::InvalidState);
        }
    }
    let mut seen = HashSet::new();
    let mut historical = HashMap::new();
    let mut sum = 0_u64;
    for entry in &input.repetition_counts {
        if entry.count == 0
            || entry.count > MAX_JS_SAFE_INTEGER
            || !seen.insert(entry.position_key.as_str())
        {
            return Err(NormalDuelError::InvalidState);
        }
        let parsed = parse_position_key(config, &entry.position_key)?;
        if parsed.walls != position.walls || parsed.stock != position.stock {
            return Err(NormalDuelError::InvalidState);
        }
        sum = sum
            .checked_add(entry.count)
            .ok_or(NormalDuelError::InvalidState)?;
        historical.insert(entry.position_key.as_str(), parsed);
    }
    if sum != input.ply - input.history_start_ply + 1 || !seen.contains(key.as_str()) {
        return Err(NormalDuelError::InvalidState);
    }
    if input.history_start_ply == 0
        && !seen.contains(
            initial_state
                .as_ref()
                .expect("history beginning at zero creates the initial state")
                .position_key
                .as_str(),
        )
    {
        return Err(NormalDuelError::InvalidState);
    }
    if !input.repetition_counts.windows(2).all(|pair| {
        compare_position_keys(&pair[0].position_key, &pair[1].position_key) == Ordering::Less
    }) {
        return Err(NormalDuelError::InvalidState);
    }
    let history_length = input.ply - input.history_start_ply + 1;
    let max_occurrences = 1 + (history_length - 1) / 4;
    if input
        .repetition_counts
        .iter()
        .any(|entry| entry.count > max_occurrences)
    {
        return Err(NormalDuelError::InvalidState);
    }
    let history_first = config.expected_turn(input.history_start_ply);
    let expected_turn_counts = match history_first {
        Player::A => Players {
            a: history_length.div_ceil(2),
            b: history_length / 2,
        },
        Player::B => Players {
            a: history_length / 2,
            b: history_length.div_ceil(2),
        },
    };
    let mut actual_turn_counts = Players { a: 0_u64, b: 0_u64 };
    for entry in &input.repetition_counts {
        let historical_position = historical
            .get(entry.position_key.as_str())
            .expect("record was inserted");
        let player_count = actual_turn_counts.get_mut(historical_position.turn);
        *player_count = player_count
            .checked_add(entry.count)
            .ok_or(NormalDuelError::InvalidState)?;
        let has_goal = historical_position.pawns.a.r == config.goal_rows.a
            || historical_position.pawns.b.r == config.goal_rows.b;
        if has_goal && (entry.position_key != key || entry.count != 1) {
            return Err(NormalDuelError::InvalidState);
        }
    }
    if actual_turn_counts != expected_turn_counts {
        return Err(NormalDuelError::InvalidState);
    }
    let moves_in_history_window = Players {
        a: completed_actions.a - actions_at_reset.a,
        b: completed_actions.b - actions_at_reset.b,
    };
    for historical_position in historical.values() {
        for player in [Player::A, Player::B] {
            let distance = u64::from(manhattan(
                *historical_position.pawns.get(player),
                *position.pawns.get(player),
            ));
            if distance > 2 * *moves_in_history_window.get(player) {
                return Err(NormalDuelError::InvalidState);
            }
        }
    }
    let at_goal_a = position.pawns.a.r == config.goal_rows.a;
    let at_goal_b = position.pawns.b.r == config.goal_rows.b;
    if (at_goal_a && at_goal_b)
        || position.pawns.get(position.turn).r == *config.goal_rows.get(position.turn)
    {
        return Err(NormalDuelError::InvalidState);
    }
    let current_count = input
        .repetition_counts
        .iter()
        .find(|entry| entry.position_key == key)
        .expect("key was checked")
        .count;
    let expected_outcome = if input.ply == 0 {
        Outcome::Ongoing
    } else {
        adjudicate(
            config,
            if at_goal_a {
                Some(Player::A)
            } else if at_goal_b {
                Some(Player::B)
            } else {
                None
            },
            current_count,
            input.ply,
        )?
    };
    if input.ply == 0
        && key
            != initial_state
                .as_ref()
                .expect("ply-zero validation creates the initial state")
                .position_key
        || input.ply > config.ply_cap
        || current_count > u64::from(config.repetition_threshold)
        || input.repetition_counts.iter().any(|entry| {
            entry.position_key != key && entry.count >= u64::from(config.repetition_threshold)
        })
        || input.outcome != expected_outcome
    {
        return Err(NormalDuelError::InvalidState);
    }
    Ok(GameState {
        position,
        position_key: key,
        ply: input.ply,
        history_start_ply: input.history_start_ply,
        repetition_counts: input.repetition_counts.clone(),
        outcome: expected_outcome,
    })
}

pub fn legal_actions(config: &Config, state: &GameState) -> Result<Vec<Action>> {
    let state = validate_state(config, state)?;
    if !state.outcome.is_ongoing() {
        return Ok(Vec::new());
    }
    legal_position_actions(config, &state.position)
}

pub fn legal_action_codes(config: &Config, state: &GameState) -> Result<Vec<usize>> {
    legal_actions(config, state)?
        .iter()
        .map(|action| action.policy_code_validated(config))
        .collect()
}

pub fn legal_action_mask(config: &Config, state: &GameState) -> Result<Vec<u8>> {
    let mut mask = vec![0; policy_size(config)?];
    for code in legal_action_codes(config, state)? {
        mask[code] = 1;
    }
    Ok(mask)
}

fn action_is_legal(config: &Config, state: &GameState, action: &Action) -> Result<bool> {
    let board = Board::from_position(config, &state.position)?;
    match action {
        Action::Pawn { to } => Ok(in_bounds(config, *to)
            && legal_pawn_destinations_board(config, &state.position, board).contains(to)),
        Action::Wall { wall } => {
            let candidate =
                Wall::parse(config, wall).map_err(|_| NormalDuelError::InvalidAction)?;
            Ok(board.candidate_is_legal(config, &state.position, candidate))
        }
    }
}

fn apply_trusted_action(config: &Config, state: &GameState, action: &Action) -> Result<GameState> {
    let mover = state.position.turn;
    let mut position = state.position.clone();
    match action {
        Action::Pawn { to } => *position.pawns.get_mut(mover) = *to,
        Action::Wall { wall } => {
            position.walls.push(
                Wall::parse(config, wall)
                    .map_err(|_| NormalDuelError::InvalidAction)?
                    .text(),
            );
            *position.stock.get_mut(mover) -= 1;
        }
    }
    position.turn = mover.other();
    let position = normalize_position(config, &position)?;
    let position_key = position_key_normalized(config, &position);
    let mut counts: HashMap<String, u64> = if matches!(action, Action::Wall { .. }) {
        HashMap::from([(position_key.clone(), 1)])
    } else {
        state
            .repetition_counts
            .iter()
            .map(|entry| (entry.position_key.clone(), entry.count))
            .collect()
    };
    if matches!(action, Action::Pawn { .. }) {
        let count = counts.entry(position_key.clone()).or_default();
        *count = count.checked_add(1).ok_or(NormalDuelError::InvalidState)?;
    }
    let mut repetition_counts: Vec<_> = counts
        .into_iter()
        .map(|(position_key, count)| RepetitionCount {
            position_key,
            count,
        })
        .collect();
    repetition_counts
        .sort_by(|left, right| compare_position_keys(&left.position_key, &right.position_key));
    let ply = state.ply + 1;
    let current_count = repetition_counts
        .iter()
        .find(|entry| entry.position_key == position_key)
        .expect("inserted position key")
        .count;
    let goal_winner = if position.pawns.get(mover).r == *config.goal_rows.get(mover) {
        Some(mover)
    } else {
        None
    };
    let outcome = adjudicate(config, goal_winner, current_count, ply)?;
    Ok(GameState {
        position,
        position_key,
        ply,
        history_start_ply: if matches!(action, Action::Wall { .. }) {
            ply
        } else {
            state.history_start_ply
        },
        repetition_counts,
        outcome,
    })
}

/// Applies one structurally valid action using direct pawn/wall checks rather
/// than rebuilding the complete policy list. Both input and output are owned,
/// immutable value objects from the caller's perspective.
pub fn apply_legal_action(
    config: &Config,
    state: &GameState,
    action: &Action,
) -> Result<GameState> {
    let state = validate_state(config, state)?;
    if !state.outcome.is_ongoing() {
        return Err(NormalDuelError::TerminalState);
    }
    if !action_is_legal(config, &state, action)? {
        return Err(NormalDuelError::IllegalAction);
    }
    apply_trusted_action(config, &state, action)
}

/// Applies an untrusted action after checking it against canonical legal
/// action codes. Use [`apply_legal_action`] in search once an action came from
/// [`legal_actions`].
pub fn apply_action(config: &Config, state: &GameState, action: &Action) -> Result<GameState> {
    config.validate()?;
    if !state.outcome.is_ongoing() {
        return Err(NormalDuelError::TerminalState);
    }
    let state = validate_state(config, state)?;
    let code = action.policy_code_validated(config)?;
    if !legal_position_action_codes(config, &state.position)?.contains(&code) {
        return Err(NormalDuelError::IllegalAction);
    }
    apply_trusted_action(config, &state, action)
}

/// Replay canonical policy codes from the configured initial state. Each code
/// is first decoded and then checked by [`apply_action`], so this is a fixture
/// convenience rather than an unchecked search shortcut.
pub fn state_from_action_codes(config: &Config, action_codes: &[usize]) -> Result<GameState> {
    let mut state = create_initial_state(config)?;
    for &code in action_codes {
        let action = decode_action(config, code)?;
        state = apply_action(config, &state, &action)?;
    }
    Ok(state)
}

/// Build a deterministic reachable state using only canonical wall actions.
/// `lcg32-v1` initializes from `seed`, advances before every choice via
/// `state = state * 1664525 + 1013904223 (mod 2^32)`, then selects
/// `state % walls.len()`. Modulo bias is deliberately contractual.
pub fn seeded_wall_state(config: &Config, seed: u32, plies: u64) -> Result<GameState> {
    let mut state = create_initial_state(config)?;
    let total_stock = state.position.stock.a + state.position.stock.b;
    if plies > total_stock {
        return Err(NormalDuelError::IllegalAction);
    }
    let mut random = seed;
    for _ in 0..plies {
        random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let walls: Vec<_> = legal_actions(config, &state)?
            .into_iter()
            .filter(|action| matches!(action, Action::Wall { .. }))
            .collect();
        if walls.is_empty() {
            return Err(NormalDuelError::IllegalAction);
        }
        let index = (random as usize) % walls.len();
        state = apply_legal_action(config, &state, &walls[index])?;
    }
    Ok(state)
}

/// Compact mutable state for a future alpha-beta core. It contains the two
/// 64-bit wall grids and can revert a move without allocating or reconstructing
/// a Position. Repetition/outcome bookkeeping remains in the immutable public
/// [`GameState`] layer, keeping this low-level type unambiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchPosition {
    pawns: Players<Coord>,
    walls: WallBits,
    stock: Players<u64>,
    turn: Player,
    blocked_down: u128,
    blocked_right: u128,
    config_hash: u64,
}

#[derive(Debug, Clone)]
pub struct Undo {
    before: SearchPosition,
    after: SearchPosition,
}

impl SearchPosition {
    pub fn from_position(config: &Config, position: &Position) -> Result<Self> {
        let position = normalize_position(config, position)?;
        let board = Board::from_position(config, &position)?;
        Ok(Self {
            pawns: position.pawns,
            walls: board.walls,
            stock: position.stock,
            turn: position.turn,
            blocked_down: board.blocked_down,
            blocked_right: board.blocked_right,
            config_hash: config_fingerprint(config),
        })
    }

    /// Convert back to the public DTO using the same validated configuration
    /// that created this search position.
    pub fn to_position(&self, config: &Config) -> Result<Position> {
        config.validate()?;
        if self.config_hash != config_fingerprint(config) {
            return Err(NormalDuelError::InvalidConfig);
        }
        let position = Position {
            pawns: self.pawns,
            walls: self
                .walls
                .walls(config)
                .into_iter()
                .map(Wall::text)
                .collect(),
            stock: self.stock,
            turn: self.turn,
        };
        let normalized = normalize_position(config, &position)?;
        let rebuilt = Board::from_position(config, &normalized)?;
        if rebuilt.walls != self.walls
            || rebuilt.blocked_down != self.blocked_down
            || rebuilt.blocked_right != self.blocked_right
        {
            return Err(NormalDuelError::InvalidPosition);
        }
        Ok(normalized)
    }

    /// The caller must already have checked the action with the public legal
    /// generator. The returned token is consumed by [`Self::undo`].
    pub fn apply_unchecked(&mut self, config: &Config, action: &Action) -> Result<Undo> {
        config.validate()?;
        if self.config_hash != config_fingerprint(config) {
            return Err(NormalDuelError::InvalidConfig);
        }
        let before = *self;
        let prior_turn = self.turn;
        match action {
            Action::Pawn { to } => {
                if !in_bounds(config, *to) {
                    return Err(NormalDuelError::InvalidAction);
                }
                *self.pawns.get_mut(prior_turn) = *to;
            }
            Action::Wall { wall } => {
                let wall = Wall::parse(config, wall).map_err(|_| NormalDuelError::InvalidAction)?;
                if *self.stock.get(prior_turn) == 0 || self.walls.contains_valid(wall, config) {
                    return Err(NormalDuelError::IllegalAction);
                }
                if !self.walls.insert_valid(wall, config) {
                    return Err(NormalDuelError::InvalidAction);
                }
                let mut board = self.board();
                board.add_wall_edges(config, wall);
                self.blocked_down = board.blocked_down;
                self.blocked_right = board.blocked_right;
                *self.stock.get_mut(prior_turn) -= 1;
            }
        }
        self.turn = prior_turn.other();
        Ok(Undo {
            before,
            after: *self,
        })
    }

    /// Restore the exact state which produced `undo`. Tokens from another
    /// position/configuration, stale tokens, and out-of-order undos fail
    /// closed without mutating the receiver.
    pub fn undo(&mut self, config: &Config, undo: Undo) -> bool {
        if config.validate().is_err()
            || self.config_hash != config_fingerprint(config)
            || *self != undo.after
        {
            return false;
        }
        *self = undo.before;
        true
    }

    fn board(&self) -> Board {
        Board {
            walls: self.walls,
            blocked_down: self.blocked_down,
            blocked_right: self.blocked_right,
        }
    }

    /// The fully allocation-free form of [`legal_action_codes_fast`]: this type
    /// already holds the bitboards, so nothing is parsed or rebuilt.
    ///
    /// Writes ascending policy codes into `out` and returns the count. `out`
    /// must hold at least `config.policy_size()` entries.
    pub fn legal_action_codes_fast(&self, config: &Config, out: &mut [u16]) -> usize {
        self.legal_action_codes_fast_stats(config, out).0
    }

    /// [`SearchPosition::legal_action_codes_fast`] plus prefilter counters.
    pub fn legal_action_codes_fast_stats(
        &self,
        config: &Config,
        out: &mut [u16],
    ) -> (usize, FastLegalStats) {
        let mut stats = FastLegalStats::default();
        let count = self
            .board()
            .legal_codes_into(config, self.pawns, self.stock, self.turn, out, &mut stats);
        (count, stats)
    }

    /// The fully allocation-free form of [`encode_state_into`].
    ///
    /// `out` must be exactly `NN_INPUT_PLANES * config.cells()` long.
    pub fn encode_into(&self, config: &Config, out: &mut [f32]) -> Result<()> {
        config.validate()?;
        if config.rows != 9 || config.columns != 9 {
            return Err(NormalDuelError::InvalidConfig);
        }
        if out.len() != NN_INPUT_PLANES * config.cells() {
            return Err(NormalDuelError::InvalidPosition);
        }
        encode_board_into(
            config,
            &self.board(),
            self.pawns,
            self.stock,
            self.turn,
            out,
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompactPositionKey {
    pawns: Players<Coord>,
    walls: WallBits,
    stock: Players<u64>,
    turn: Player,
}

impl CompactPositionKey {
    fn from_search(position: &SearchPosition) -> Self {
        Self {
            pawns: position.pawns,
            walls: position.walls,
            stock: position.stock,
            turn: position.turn,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompactRepetition {
    key: CompactPositionKey,
    count: u64,
    epoch: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PreparedSearchStructural {
    config_hash: u64,
    position: CompactPositionKey,
    ply: u64,
    outcome: Outcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedOracleStructural {
    search: PreparedSearchStructural,
    repetitions: Vec<(CompactPositionKey, u64)>,
}

/// Allocation-free, mirror-canonical identity for transposition-table probes.
/// Equality intentionally excludes the orientation flag, so reflected states
/// verify as the same entry. Bounds are reusable only when
/// [`Self::bounds_reusable`] returns true; otherwise the entry is an ordering
/// hint because threefold repetition depends on history.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PreparedSearchIdentity {
    key: u64,
    mirrored: bool,
    bounds_reusable: bool,
    structural: PreparedSearchStructural,
}

impl PartialEq for PreparedSearchIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.structural == other.structural
    }
}

impl Eq for PreparedSearchIdentity {}

impl PreparedSearchIdentity {
    #[must_use]
    pub(crate) const fn key(&self) -> u64 {
        self.key
    }

    #[must_use]
    pub(crate) const fn mirrored(&self) -> bool {
        self.mirrored
    }

    /// True only for a fresh repetition epoch containing exactly the current
    /// position at count one. TT scores from all other epochs are
    /// history-dependent and must not be used as bounds.
    #[must_use]
    pub(crate) const fn bounds_reusable(&self) -> bool {
        self.bounds_reusable
    }
}

/// Context-complete identity used by the exact zero-wall solver. Unlike
/// [`PreparedSearchIdentity`], it includes the active repetition multiset and
/// must not be used in the alpha-beta hot path.
#[derive(Debug, Clone)]
pub(crate) struct PreparedOracleIdentity {
    key: u64,
    mirrored: bool,
    structural: PreparedOracleStructural,
}

impl PartialEq for PreparedOracleIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.structural == other.structural
    }
}

impl Eq for PreparedOracleIdentity {}

impl PreparedOracleIdentity {
    #[must_use]
    pub(crate) const fn key(&self) -> u64 {
        self.key
    }

    #[must_use]
    pub(crate) const fn mirrored(&self) -> bool {
        self.mirrored
    }
}

/// A complete game state converted once from the strict wire contract into the
/// allocation-light search representation. Public methods remain checked;
/// internal traversal consumes only action codes generated from this state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedGameState {
    position: SearchPosition,
    ply: u64,
    outcome: Outcome,
    repetitions: Vec<CompactRepetition>,
    epoch: u64,
}

#[derive(Debug)]
enum PreparedRepetitionUndo {
    Incremented(usize),
    Pushed,
    Wall,
}

#[derive(Debug)]
struct PreparedUndo {
    before_position: SearchPosition,
    before_ply: u64,
    before_outcome: Outcome,
    repetition: PreparedRepetitionUndo,
    after_position: SearchPosition,
    after_ply: u64,
}

#[derive(Clone)]
struct CodeList {
    codes: [u16; 209],
    len: usize,
}

impl CodeList {
    fn new() -> Self {
        Self {
            codes: [0; 209],
            len: 0,
        }
    }

    fn push(&mut self, code: usize) {
        debug_assert!(self.len < self.codes.len());
        debug_assert!(u16::try_from(code).is_ok());
        self.codes[self.len] = code as u16;
        self.len += 1;
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.codes[..self.len].iter().map(|&code| usize::from(code))
    }
}

fn compact_pawn_codes(
    config: &Config,
    position: &SearchPosition,
    board: Board,
    output: &mut CodeList,
) {
    let mover = *position.pawns.get(position.turn);
    let opponent = *position.pawns.get(position.turn.other());
    let mut found = [false; 81];
    for (dr, dc) in DIRECTIONS {
        let Some(adjacent) = offset(mover, dr, dc) else {
            continue;
        };
        if !in_bounds(config, adjacent) || board.edge_blocked(config, mover, adjacent) {
            continue;
        }
        if adjacent != opponent {
            found[square(config, adjacent)] = true;
            continue;
        }
        for (exit_r, exit_c) in DIRECTIONS {
            let Some(destination) = offset(opponent, exit_r, exit_c) else {
                continue;
            };
            if destination != mover
                && in_bounds(config, destination)
                && !board.edge_blocked(config, opponent, destination)
            {
                found[square(config, destination)] = true;
            }
        }
    }
    for (code, present) in found.into_iter().take(config.cells()).enumerate() {
        if present {
            output.push(code);
        }
    }
}

fn compact_legal_codes(config: &Config, state: &PreparedGameState) -> CodeList {
    let mut output = CodeList::new();
    if !state.outcome.is_ongoing() {
        return output;
    }
    let board = state.position.board();
    compact_pawn_codes(config, &state.position, board, &mut output);
    if *state.position.stock.get(state.position.turn) > 0 {
        for orientation in [Orientation::Horizontal, Orientation::Vertical] {
            for r in 0..config.rows - 1 {
                for c in 0..config.columns - 1 {
                    let wall = Wall { orientation, r, c };
                    if board.candidate_is_legal_parts(
                        config,
                        state.position.pawns,
                        state.position.stock,
                        state.position.turn,
                        wall,
                    ) {
                        output.push(wall.policy_code(config));
                    }
                }
            }
        }
    }
    output
}

fn mirror_wall_bits(config: &Config, walls: WallBits) -> WallBits {
    let mut mirrored = WallBits::default();
    let columns = usize::from(config.columns - 1);
    for (source, target) in [
        (walls.horizontal, &mut mirrored.horizontal),
        (walls.vertical, &mut mirrored.vertical),
    ] {
        for anchor in 0..config.anchors_per_axis() {
            if source & (1_u64 << anchor) != 0 {
                let row = anchor / columns;
                let column = anchor % columns;
                let reflected = row * columns + (columns - 1 - column);
                *target |= 1_u64 << reflected;
            }
        }
    }
    mirrored
}

fn mirror_compact_position_key(config: &Config, key: CompactPositionKey) -> CompactPositionKey {
    let mirror_coord = |coord: Coord| Coord {
        r: coord.r,
        c: config.columns - 1 - coord.c,
    };
    CompactPositionKey {
        pawns: Players {
            a: mirror_coord(key.pawns.a),
            b: mirror_coord(key.pawns.b),
        },
        walls: mirror_wall_bits(config, key.walls),
        stock: key.stock,
        turn: key.turn,
    }
}

fn compact_identity_order(
    config: &Config,
    key: CompactPositionKey,
) -> (usize, usize, u64, u64, u64, u64, u8) {
    (
        square(config, key.pawns.a),
        square(config, key.pawns.b),
        key.walls.horizontal,
        key.walls.vertical,
        key.stock.a,
        key.stock.b,
        u8::from(matches!(key.turn, Player::B)),
    )
}

fn compact_identity_hash(config: &Config, key: CompactPositionKey) -> u64 {
    let ordered = compact_identity_order(config, key);
    let mut hash = mix64(ordered.0 as u64);
    hash ^= mix64(ordered.1 as u64 ^ 0x9e37_79b9_7f4a_7c15);
    hash ^= mix64(ordered.2 ^ 0xbf58_476d_1ce4_e5b9);
    hash ^= mix64(ordered.3 ^ 0x94d0_49bb_1331_11eb);
    hash ^= mix64(ordered.4 ^ 0xd6e8_feb8_6659_fd93);
    hash ^= mix64(ordered.5 ^ 0xa076_1d64_78bd_642f);
    hash ^ mix64(u64::from(ordered.6) ^ 0xe703_7ed1_a0b4_28db)
}

impl PreparedGameState {
    pub fn from_game_state(config: &Config, state: &GameState) -> Result<Self> {
        let state = validate_state(config, state)?;
        let position = SearchPosition::from_position(config, &state.position)?;
        let mut repetitions = Vec::with_capacity(state.repetition_counts.len() + 8);
        for entry in state.repetition_counts {
            let historical = parse_position_key(config, &entry.position_key)?;
            let historical = SearchPosition::from_position(config, &historical)?;
            repetitions.push(CompactRepetition {
                key: CompactPositionKey::from_search(&historical),
                count: entry.count,
                epoch: 0,
            });
        }
        Ok(Self {
            position,
            ply: state.ply,
            outcome: state.outcome,
            repetitions,
            epoch: 0,
        })
    }

    pub fn legal_action_codes(&self, config: &Config) -> Result<Vec<usize>> {
        config.validate()?;
        if self.position.config_hash != config_fingerprint(config) {
            return Err(NormalDuelError::InvalidConfig);
        }
        Ok(compact_legal_codes(config, self).iter().collect())
    }

    /// Allocation-free transposition identity including board, side, stock,
    /// ply and outcome, canonicalized through the supported left/right mirror.
    /// The full repetition history is deliberately excluded from this hot-path
    /// key; callers must honor [`PreparedSearchIdentity::bounds_reusable`].
    pub(crate) fn search_identity(&self, config: &Config) -> PreparedSearchIdentity {
        let direct = CompactPositionKey::from_search(&self.position);
        let reflected = mirror_compact_position_key(config, direct);
        let direct_hash = compact_identity_hash(config, direct);
        let reflected_hash = compact_identity_hash(config, reflected);
        let mirrored = reflected_hash < direct_hash
            || (reflected_hash == direct_hash
                && compact_identity_order(config, reflected)
                    < compact_identity_order(config, direct));
        let position = if mirrored { reflected } else { direct };
        let structural = PreparedSearchStructural {
            config_hash: self.position.config_hash,
            position,
            ply: self.ply,
            outcome: self.outcome,
        };
        let current = CompactPositionKey::from_search(&self.position);
        let mut active_count = 0_usize;
        let mut current_is_one = false;
        for repetition in self
            .repetitions
            .iter()
            .filter(|entry| entry.epoch == self.epoch)
        {
            active_count += 1;
            current_is_one = repetition.key == current && repetition.count == 1;
        }
        let bounds_reusable = active_count == 1 && current_is_one;
        let outcome_domain: u64 = match self.outcome {
            Outcome::Ongoing => 0,
            Outcome::Win {
                winner: Player::A, ..
            } => 1,
            Outcome::Win {
                winner: Player::B, ..
            } => 2,
            Outcome::Draw {
                reason: DrawReason::ThreefoldRepetition,
            } => 3,
            Outcome::Draw {
                reason: DrawReason::PlyCap,
            } => 4,
        };
        let key = compact_identity_hash(config, position)
            ^ mix64(self.ply ^ outcome_domain.rotate_left(17));
        PreparedSearchIdentity {
            key,
            mirrored,
            bounds_reusable,
            structural,
        }
    }

    /// Context-complete, mirror-canonical identity for exact solvers. This
    /// allocates and sorts the active repetition context and should remain off
    /// the normal alpha-beta probe path.
    pub(crate) fn oracle_identity(&self, config: &Config) -> PreparedOracleIdentity {
        let search = self.search_identity(config);
        let canonicalize = |key| {
            if search.mirrored {
                mirror_compact_position_key(config, key)
            } else {
                key
            }
        };
        let mut repetitions: Vec<_> = self
            .repetitions
            .iter()
            .filter(|entry| entry.epoch == self.epoch)
            .map(|entry| (canonicalize(entry.key), entry.count))
            .collect();
        repetitions.sort_by_key(|(key, _)| compact_identity_order(config, *key));
        let structural = PreparedOracleStructural {
            search: search.structural,
            repetitions,
        };
        let mut context_hash = search.key;
        for (index, (key, count)) in structural.repetitions.iter().enumerate() {
            context_hash ^= mix64(
                compact_identity_hash(config, *key)
                    ^ count.rotate_left((index % 63) as u32)
                    ^ index as u64,
            );
        }
        PreparedOracleIdentity {
            key: mix64(context_hash),
            mirrored: search.mirrored,
            structural,
        }
    }

    /// Consume one matching undo token and restore the exact prior state.
    fn undo_generated_code(&mut self, undo: PreparedUndo) -> bool {
        self.undo_generated(undo)
    }

    fn apply_generated_code(&mut self, config: &Config, code: usize) -> Result<PreparedUndo> {
        if !self.outcome.is_ongoing() || code >= config.policy_size() {
            return Err(NormalDuelError::IllegalAction);
        }
        let before_position = self.position;
        let before_ply = self.ply;
        let before_outcome = self.outcome;
        let mover = self.position.turn;
        let mut placed_wall = false;
        if code < config.cells() {
            *self.position.pawns.get_mut(mover) = Coord {
                r: (code / config.columns as usize) as u8,
                c: (code % config.columns as usize) as u8,
            };
        } else {
            let anchors = config.anchors_per_axis();
            let offset = code - config.cells();
            let wall = Wall {
                orientation: if offset < anchors {
                    Orientation::Horizontal
                } else {
                    Orientation::Vertical
                },
                r: ((offset % anchors) / (config.columns as usize - 1)) as u8,
                c: ((offset % anchors) % (config.columns as usize - 1)) as u8,
            };
            if !self.position.walls.insert_valid(wall, config) {
                return Err(NormalDuelError::IllegalAction);
            }
            let mut board = self.position.board();
            board.add_wall_edges(config, wall);
            self.position.blocked_down = board.blocked_down;
            self.position.blocked_right = board.blocked_right;
            *self.position.stock.get_mut(mover) -= 1;
            placed_wall = true;
        }
        self.position.turn = mover.other();
        self.ply = self
            .ply
            .checked_add(1)
            .ok_or(NormalDuelError::InvalidState)?;
        let key = CompactPositionKey::from_search(&self.position);
        let (repetition, current_count) = if placed_wall {
            self.epoch = self
                .epoch
                .checked_add(1)
                .ok_or(NormalDuelError::InvalidState)?;
            self.repetitions.push(CompactRepetition {
                key,
                count: 1,
                epoch: self.epoch,
            });
            (PreparedRepetitionUndo::Wall, 1)
        } else if let Some(index) = self
            .repetitions
            .iter()
            .position(|entry| entry.epoch == self.epoch && entry.key == key)
        {
            self.repetitions[index].count += 1;
            (
                PreparedRepetitionUndo::Incremented(index),
                self.repetitions[index].count,
            )
        } else {
            self.repetitions.push(CompactRepetition {
                key,
                count: 1,
                epoch: self.epoch,
            });
            (PreparedRepetitionUndo::Pushed, 1)
        };
        self.outcome = if self.position.pawns.get(mover).r == *config.goal_rows.get(mover) {
            Outcome::Win {
                winner: mover,
                reason: GoalReason::Goal,
            }
        } else if current_count >= u64::from(config.repetition_threshold) {
            Outcome::Draw {
                reason: DrawReason::ThreefoldRepetition,
            }
        } else if self.ply >= config.ply_cap {
            Outcome::Draw {
                reason: DrawReason::PlyCap,
            }
        } else {
            Outcome::Ongoing
        };
        Ok(PreparedUndo {
            before_position,
            before_ply,
            before_outcome,
            repetition,
            after_position: self.position,
            after_ply: self.ply,
        })
    }

    fn undo_generated(&mut self, undo: PreparedUndo) -> bool {
        if self.position != undo.after_position || self.ply != undo.after_ply {
            return false;
        }
        match undo.repetition {
            PreparedRepetitionUndo::Incremented(index) => {
                let Some(entry) = self.repetitions.get_mut(index) else {
                    return false;
                };
                if entry.count <= 1 {
                    return false;
                }
                entry.count -= 1;
            }
            PreparedRepetitionUndo::Pushed => {
                self.repetitions.pop();
            }
            PreparedRepetitionUndo::Wall => {
                self.repetitions.pop();
                self.epoch -= 1;
            }
        }
        self.position = undo.before_position;
        self.ply = undo.before_ply;
        self.outcome = undo.before_outcome;
        true
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastThroughputProbe {
    pub root_action_codes: Vec<usize>,
    pub child_action_count: u64,
    pub perft_leaves: u64,
}

/// One compact child transition, materialized only for pre-timing differential
/// verification. Search itself continues to use allocation-free apply/undo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastTraceChild {
    pub action_code: usize,
    pub position_key: String,
    pub legal_action_codes: Vec<usize>,
}

/// Exact observable trace of one compact root and each of its direct children.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastTrace {
    pub root_action_codes: Vec<usize>,
    pub children: Vec<FastTraceChild>,
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn config_fingerprint(config: &Config) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in position_key_normalized(
        config,
        &Position {
            pawns: config.start,
            walls: Vec::new(),
            stock: config.initial_stock,
            turn: config.first_player,
        },
    )
    .bytes()
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn zobrist_feature(config_hash: u64, domain: u64, value: u64) -> u64 {
    // Hash the namespace and payload independently before combining them.
    // This prevents arbitrary stock values from crossing into another
    // feature's numeric range and creating deterministic XOR aliases.
    let namespace = mix64(config_hash ^ domain.wrapping_mul(0xd6e8_feb8_6659_fd93));
    mix64(namespace ^ mix64(value ^ 0xa076_1d64_78bd_642f))
}

fn zobrist_for(config: &Config, position: &Position, mirrored: bool) -> Result<u64> {
    let normalized = normalize_position(config, position)?;
    let board = Board::from_position(config, &normalized)?;
    let hash = config_fingerprint(config);
    let mirror_coord = |coord: Coord| Coord {
        r: coord.r,
        c: if mirrored {
            config.columns - 1 - coord.c
        } else {
            coord.c
        },
    };
    let mut result = zobrist_feature(hash, 1, 0)
        ^ zobrist_feature(
            hash,
            2,
            square(config, mirror_coord(normalized.pawns.a)) as u64,
        )
        ^ zobrist_feature(
            hash,
            3,
            square(config, mirror_coord(normalized.pawns.b)) as u64,
        );
    for wall in board.walls.walls(config) {
        let c = if mirrored {
            config.columns - 2 - wall.c
        } else {
            wall.c
        };
        let mirrored_wall = Wall { c, ..wall };
        let domain = match wall.orientation {
            Orientation::Horizontal => 4,
            Orientation::Vertical => 5,
        };
        result ^= zobrist_feature(hash, domain, mirrored_wall.anchor_index(config) as u64);
    }
    result ^= zobrist_feature(hash, 6, normalized.stock.a);
    result ^= zobrist_feature(hash, 7, normalized.stock.b);
    result ^= zobrist_feature(hash, 8, u64::from(matches!(normalized.turn, Player::B)));
    Ok(result)
}

/// Deterministic Zobrist-style transposition key covering configuration,
/// pawns, both 64-bit wall grids, stock, and side to move.
pub fn zobrist_key(config: &Config, position: &Position) -> Result<u64> {
    zobrist_for(config, position, false)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalZobrist {
    pub key: u64,
    /// True when callers must left/right mirror actions to enter the canonical
    /// transposition-table orientation.
    pub mirrored: bool,
}

/// Return the canonical left/right hash together with its orientation.
pub fn canonical_zobrist(config: &Config, position: &Position) -> Result<CanonicalZobrist> {
    let direct = zobrist_for(config, position, false)?;
    let mirrored = zobrist_for(config, position, true)?;
    Ok(if mirrored < direct {
        CanonicalZobrist {
            key: mirrored,
            mirrored: true,
        }
    } else {
        CanonicalZobrist {
            key: direct,
            mirrored: false,
        }
    })
}

/// Canonicalizes only the left/right reflected board; rotations and goal-side
/// swaps are intentionally never identified because they change game meaning.
pub fn canonical_zobrist_key(config: &Config, position: &Position) -> Result<u64> {
    Ok(canonical_zobrist(config, position)?.key)
}

/// Transform an action through the sole supported symmetry. Calling this
/// function twice returns the original canonical action.
pub fn mirror_action(config: &Config, action: &Action) -> Result<Action> {
    config.validate()?;
    match action {
        Action::Pawn { to } if in_bounds(config, *to) => Ok(Action::Pawn {
            to: Coord {
                r: to.r,
                c: config.columns - 1 - to.c,
            },
        }),
        Action::Pawn { .. } => Err(NormalDuelError::InvalidAction),
        Action::Wall { wall } => {
            let wall = Wall::parse(config, wall).map_err(|_| NormalDuelError::InvalidAction)?;
            Ok(Action::Wall {
                wall: Wall {
                    c: config.columns - 2 - wall.c,
                    ..wall
                }
                .text(),
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerftDivide {
    pub action: Action,
    pub action_code: usize,
    /// `child_leaves_by_depth[d] == P(apply(action), d)`; hence root depth d
    /// is the sum of these values at index d - 1.
    pub child_leaves_by_depth: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerftReport {
    pub depth: u8,
    pub leaves_by_depth: Vec<u64>,
    /// Counts scalar state evaluations using the reference report semantics.
    /// The root is charged once; `childLeavesByDepth[0]` is algebraic and does
    /// not charge a child; each deeper requested child depth is independent.
    pub node_visits: u64,
    pub divide: Vec<PerftDivide>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerftOptions {
    pub max_nodes: u64,
}

impl Default for PerftOptions {
    fn default() -> Self {
        Self {
            max_nodes: DEFAULT_MAX_PERFT_NODES,
        }
    }
}

impl PerftOptions {
    fn validate(self) -> Result<Self> {
        if self.max_nodes == 0 || self.max_nodes > MAX_PERFT_NODES_HARD_CAP {
            return Err(NormalDuelError::InvalidPerftBudget);
        }
        Ok(self)
    }
}

struct PerftCounter {
    visits: u64,
    max_nodes: u64,
}

impl PerftCounter {
    fn new(options: PerftOptions) -> Result<Self> {
        Ok(Self {
            visits: 0,
            max_nodes: options.validate()?.max_nodes,
        })
    }

    fn charge(&mut self) -> Result<()> {
        if self.visits >= self.max_nodes {
            return Err(NormalDuelError::PerftNodeBudget);
        }
        self.visits += 1;
        Ok(())
    }
}

fn perft_exact_counted(
    config: &Config,
    state: &GameState,
    depth: u8,
    counter: &mut PerftCounter,
) -> Result<u64> {
    counter.charge()?;
    if depth == 0 {
        return Ok(1);
    }
    if !state.outcome.is_ongoing() {
        return Ok(0);
    }
    if depth == 1 {
        return Ok(legal_actions(config, state)?.len() as u64);
    }
    let mut total = 0_u64;
    for action in legal_actions(config, state)? {
        total = total
            .checked_add(perft_exact_counted(
                config,
                &apply_legal_action(config, state, &action)?,
                depth - 1,
                counter,
            )?)
            .ok_or(NormalDuelError::PerftOverflow)?;
    }
    Ok(total)
}

/// Exact-depth perft with the reference engine's deterministic 400-node
/// default budget. Larger reviewed probes must use [`perft_with_options`].
pub fn perft(config: &Config, state: &GameState, depth: u8) -> Result<u64> {
    perft_with_options(config, state, depth, PerftOptions::default())
}

/// Exact-depth perft with an explicit deterministic budget. Use
/// `PerftOptions::default()` for the reference 400-node default.
pub fn perft_with_options(
    config: &Config,
    state: &GameState,
    depth: u8,
    options: PerftOptions,
) -> Result<u64> {
    if depth > MAX_PERFT_DEPTH {
        return Err(NormalDuelError::PerftDepth);
    }
    let state = validate_state(config, state)?;
    perft_exact_counted(config, &state, depth, &mut PerftCounter::new(options)?)
}

/// Perft report with the reference engine's 400-node default budget.
pub fn perft_report(config: &Config, state: &GameState, depth: u8) -> Result<PerftReport> {
    perft_report_with_options(config, state, depth, PerftOptions::default())
}

pub fn perft_report_with_options(
    config: &Config,
    state: &GameState,
    depth: u8,
    options: PerftOptions,
) -> Result<PerftReport> {
    if depth > MAX_PERFT_DEPTH {
        return Err(NormalDuelError::PerftDepth);
    }
    let state = validate_state(config, state)?;
    let mut counter = PerftCounter::new(options)?;
    counter.charge()?;
    if depth == 0 || !state.outcome.is_ongoing() {
        let mut leaves_by_depth = vec![1];
        leaves_by_depth.extend(std::iter::repeat(0).take(depth as usize));
        return Ok(PerftReport {
            depth,
            leaves_by_depth,
            node_visits: counter.visits,
            divide: Vec::new(),
        });
    }
    let mut divide = Vec::new();
    for action in legal_actions(config, &state)? {
        let child = apply_legal_action(config, &state, &action)?;
        let mut child_leaves_by_depth = Vec::with_capacity(depth as usize);
        for child_depth in 0..depth {
            child_leaves_by_depth.push(if child_depth == 0 {
                1
            } else {
                perft_exact_counted(config, &child, child_depth, &mut counter)?
            });
        }
        divide.push(PerftDivide {
            action_code: action.policy_code_validated(config)?,
            action,
            child_leaves_by_depth,
        });
    }
    let mut leaves_by_depth = vec![1];
    for current_depth in 1..=depth as usize {
        let total = divide.iter().try_fold(0_u64, |sum, entry| {
            sum.checked_add(entry.child_leaves_by_depth[current_depth - 1])
                .ok_or(NormalDuelError::PerftOverflow)
        })?;
        leaves_by_depth.push(total);
    }
    Ok(PerftReport {
        depth,
        leaves_by_depth,
        node_visits: counter.visits,
        divide,
    })
}

fn fast_perft_exact(
    config: &Config,
    state: &mut PreparedGameState,
    depth: u8,
    counter: &mut PerftCounter,
) -> Result<u64> {
    counter.charge()?;
    if depth == 0 {
        return Ok(1);
    }
    if !state.outcome.is_ongoing() {
        return Ok(0);
    }
    let codes = compact_legal_codes(config, state);
    if depth == 1 {
        return Ok(codes.len as u64);
    }
    let mut total = 0_u64;
    for code in codes.iter() {
        let undo = state.apply_generated_code(config, code)?;
        let child_result = fast_perft_exact(config, state, depth - 1, counter);
        if !state.undo_generated(undo) {
            return Err(NormalDuelError::InvalidState);
        }
        total = total
            .checked_add(child_result?)
            .ok_or(NormalDuelError::PerftOverflow)?;
    }
    Ok(total)
}

/// Convert and validate once, then run exact-depth perft entirely on compact
/// mutable state. This is the search/throughput path; wire-format repetition
/// and outcome semantics remain structurally exact.
pub fn fast_perft_with_options(
    config: &Config,
    state: &GameState,
    depth: u8,
    options: PerftOptions,
) -> Result<u64> {
    if depth > MAX_PERFT_DEPTH {
        return Err(NormalDuelError::PerftDepth);
    }
    let mut prepared = PreparedGameState::from_game_state(config, state)?;
    fast_perft_exact(
        config,
        &mut prepared,
        depth,
        &mut PerftCounter::new(options)?,
    )
}

/// Materialize the compact path's complete direct-child trace. This is meant
/// for differential verification outside timed search: it proves action order,
/// every child position key, and every child legal-action list.
pub fn fast_trace(config: &Config, state: &GameState) -> Result<FastTrace> {
    let mut prepared = PreparedGameState::from_game_state(config, state)?;
    let root_action_codes: Vec<_> = compact_legal_codes(config, &prepared).iter().collect();
    let mut children = Vec::with_capacity(root_action_codes.len());
    for &action_code in &root_action_codes {
        let undo = prepared.apply_generated_code(config, action_code)?;
        let position = prepared.position.to_position(config)?;
        let child = FastTraceChild {
            action_code,
            position_key: position_key_normalized(config, &position),
            legal_action_codes: compact_legal_codes(config, &prepared).iter().collect(),
        };
        if !prepared.undo_generated(undo) {
            return Err(NormalDuelError::InvalidState);
        }
        children.push(child);
    }
    Ok(FastTrace {
        root_action_codes,
        children,
    })
}

/// Fixed checksum-compatible benchmark unit: canonical root codes, every
/// child's canonical legal-action count, and exact-depth perft. Conversion and
/// validation occur once before compact traversal.
pub fn fast_throughput_probe(
    config: &Config,
    state: &GameState,
    depth: u8,
    options: PerftOptions,
) -> Result<FastThroughputProbe> {
    if depth > MAX_PERFT_DEPTH {
        return Err(NormalDuelError::PerftDepth);
    }
    let mut prepared = PreparedGameState::from_game_state(config, state)?;
    let root = compact_legal_codes(config, &prepared);
    let root_action_codes: Vec<_> = root.iter().collect();
    let mut child_action_count = 0_u64;
    for code in root.iter() {
        let undo = prepared.apply_generated_code(config, code)?;
        child_action_count = child_action_count
            .checked_add(compact_legal_codes(config, &prepared).len as u64)
            .ok_or(NormalDuelError::PerftOverflow)?;
        if !prepared.undo_generated(undo) {
            return Err(NormalDuelError::InvalidState);
        }
    }
    let perft_leaves = fast_perft_exact(
        config,
        &mut prepared,
        depth,
        &mut PerftCounter::new(options)?,
    )?;
    Ok(FastThroughputProbe {
        root_action_codes,
        child_action_count,
        perft_leaves,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard() -> Config {
        Config {
            ruleset: RULESET.into(),
            rows: 9,
            columns: 9,
            start: Players {
                a: Coord { r: 8, c: 4 },
                b: Coord { r: 0, c: 4 },
            },
            goal_rows: Players { a: 0, b: 8 },
            initial_stock: Players { a: 10, b: 10 },
            jump_rule: JUMP_RULE.into(),
            repetition_threshold: 3,
            ply_cap: 200,
            first_player: Player::A,
        }
    }

    #[test]
    fn policy_encoding_and_initial_perft_match_contract() {
        let config = standard();
        assert_eq!(policy_size(&config), Ok(209));
        assert_eq!(
            encode_action(
                &config,
                &Action::Wall {
                    wall: "H-0-0".into()
                }
            ),
            Ok(81)
        );
        assert_eq!(
            encode_action(
                &config,
                &Action::Wall {
                    wall: "V-0-0".into()
                }
            ),
            Ok(145)
        );
        for code in 0..209 {
            assert_eq!(
                encode_action(&config, &decode_action(&config, code).unwrap()),
                Ok(code)
            );
        }
        let initial = create_initial_state(&config).unwrap();
        assert_eq!(perft(&config, &initial, 0), Ok(1));
        assert_eq!(perft(&config, &initial, 1), Ok(131));
        assert_eq!(perft(&config, &initial, 2), Ok(16_677));
        let report = perft_report(&config, &initial, 2).unwrap();
        assert_eq!(report.leaves_by_depth, vec![1, 131, 16_677]);
        assert_eq!(report.node_visits, 132);
        assert_eq!(
            perft_report(&config, &initial, 3),
            Err(NormalDuelError::PerftNodeBudget)
        );
        assert_eq!(
            perft_with_options(
                &config,
                &initial,
                1,
                PerftOptions {
                    max_nodes: MAX_PERFT_NODES_HARD_CAP + 1,
                },
            ),
            Err(NormalDuelError::InvalidPerftBudget)
        );
    }

    #[test]
    fn permissive_exits_and_wall_geometry_are_enforced() {
        let config = standard();
        let position = Position {
            pawns: Players {
                a: Coord { r: 4, c: 4 },
                b: Coord { r: 3, c: 4 },
            },
            walls: Vec::new(),
            stock: Players { a: 10, b: 10 },
            turn: Player::A,
        };
        assert_eq!(
            legal_pawn_destinations(&config, &position).unwrap(),
            vec![
                Coord { r: 2, c: 4 },
                Coord { r: 3, c: 3 },
                Coord { r: 3, c: 5 },
                Coord { r: 4, c: 3 },
                Coord { r: 4, c: 5 },
                Coord { r: 5, c: 4 },
            ]
        );
        let with_wall = Position {
            walls: vec!["H-0-0".into()],
            stock: Players { a: 9, b: 10 },
            ..create_initial_state(&config).unwrap().position
        };
        assert!(!is_legal_wall(&config, &with_wall, "H-0-1").unwrap());
        assert!(!is_legal_wall(&config, &with_wall, "V-0-0").unwrap());
    }

    #[test]
    fn apply_is_immutable_and_resets_repetition_after_wall() {
        let config = standard();
        let initial = create_initial_state(&config).unwrap();
        let next = apply_action(
            &config,
            &initial,
            &Action::Wall {
                wall: "H-0-0".into(),
            },
        )
        .unwrap();
        assert_eq!(initial.ply, 0);
        assert_eq!(next.ply, 1);
        assert_eq!(next.history_start_ply, 1);
        assert_eq!(next.repetition_counts.len(), 1);
        assert_eq!(next.repetition_counts[0].count, 1);
        assert_eq!(next.position.stock.a, 9);
    }

    #[test]
    fn search_apply_undo_and_left_right_hash_canonicalization() {
        let config = standard();
        let state = create_initial_state(&config).unwrap();
        let mut search = SearchPosition::from_position(&config, &state.position).unwrap();
        let before = search.to_position(&config).unwrap();
        let undo = search
            .apply_unchecked(
                &config,
                &Action::Pawn {
                    to: Coord { r: 7, c: 4 },
                },
            )
            .unwrap();
        assert!(search.undo(&config, undo));
        assert_eq!(search.to_position(&config), Ok(before));
        let mut first = SearchPosition::from_position(&config, &state.position).unwrap();
        let mut second = SearchPosition::from_position(&config, &state.position).unwrap();
        let first_undo = first
            .apply_unchecked(
                &config,
                &Action::Pawn {
                    to: Coord { r: 7, c: 4 },
                },
            )
            .unwrap();
        second
            .apply_unchecked(
                &config,
                &Action::Pawn {
                    to: Coord { r: 8, c: 3 },
                },
            )
            .unwrap();
        let second_before_foreign_undo = second;
        assert!(!second.undo(&config, first_undo));
        assert_eq!(second, second_before_foreign_undo);

        let mut bits = WallBits::default();
        let invalid = Wall {
            orientation: Orientation::Horizontal,
            r: 9,
            c: 0,
        };
        assert!(!bits.insert(invalid, &config));
        assert!(!bits.remove(invalid, &config));
        assert_eq!(bits, WallBits::default());
        let left = Position {
            pawns: Players {
                a: Coord { r: 8, c: 3 },
                b: Coord { r: 0, c: 4 },
            },
            walls: vec!["V-1-2".into()],
            stock: Players { a: 9, b: 10 },
            turn: Player::B,
        };
        let right = Position {
            pawns: Players {
                a: Coord { r: 8, c: 5 },
                b: Coord { r: 0, c: 4 },
            },
            walls: vec!["V-1-5".into()],
            stock: Players { a: 9, b: 10 },
            turn: Player::B,
        };
        assert_eq!(
            canonical_zobrist_key(&config, &left),
            canonical_zobrist_key(&config, &right)
        );
        let wall_action = Action::Wall {
            wall: "V-1-2".into(),
        };
        assert_eq!(
            mirror_action(&config, &mirror_action(&config, &wall_action).unwrap()),
            Ok(wall_action)
        );
        let left_canonical = canonical_zobrist(&config, &left).unwrap();
        let right_canonical = canonical_zobrist(&config, &right).unwrap();
        assert_eq!(left_canonical.key, right_canonical.key);
        assert_ne!(
            left_canonical.mirrored, right_canonical.mirrored,
            "non-symmetric mirror pair uses opposite canonical orientations"
        );
    }

    #[test]
    fn public_wall_apis_fail_closed_for_invalid_configs() {
        let mut zero_dimensions = standard();
        zero_dimensions.rows = 0;
        zero_dimensions.columns = 0;
        let mut unsupported_ruleset = standard();
        unsupported_ruleset.ruleset = "normal-duel-v2".into();
        let wall = Wall {
            orientation: Orientation::Horizontal,
            r: 0,
            c: 0,
        };

        for invalid_config in [&zero_dimensions, &unsupported_ruleset] {
            assert_eq!(
                Wall::parse(invalid_config, "H-0-0"),
                Err(NormalDuelError::InvalidConfig)
            );
            let mut bits = WallBits {
                horizontal: 1,
                vertical: 0,
            };
            let before = bits;
            assert!(!bits.contains(wall, invalid_config));
            assert!(!bits.insert(wall, invalid_config));
            assert!(!bits.remove(wall, invalid_config));
            assert_eq!(bits, before);
        }
    }

    #[test]
    fn zero_dimension_helpers_and_search_conversion_are_total() {
        let config = standard();
        let initial = create_initial_state(&config).unwrap();
        let search = SearchPosition::from_position(&config, &initial.position).unwrap();

        for (rows, columns) in [(0, 0), (0, 9), (9, 0)] {
            let mut invalid = config.clone();
            invalid.rows = rows;
            invalid.columns = columns;
            assert_eq!(invalid.cells(), 0);
            assert_eq!(invalid.anchors_per_axis(), 0);
            assert_eq!(invalid.policy_size(), 0);
            assert_eq!(
                search.to_position(&invalid),
                Err(NormalDuelError::InvalidConfig)
            );
            assert_eq!(
                Action::Pawn {
                    to: Coord { r: 0, c: 0 }
                }
                .policy_code(&invalid),
                Err(NormalDuelError::InvalidConfig)
            );
        }
        let mut unsupported = config;
        unsupported.ruleset = "normal-duel-v2".into();
        assert_eq!(
            Action::Pawn {
                to: Coord { r: 7, c: 4 }
            }
            .policy_code(&unsupported),
            Err(NormalDuelError::InvalidConfig)
        );
    }

    #[test]
    fn search_conversion_rejects_unchecked_illegal_states_and_stale_edges() {
        let config = standard();
        let initial = create_initial_state(&config).unwrap();

        let mut overlapping = SearchPosition::from_position(&config, &initial.position).unwrap();
        overlapping
            .apply_unchecked(
                &config,
                &Action::Pawn {
                    to: initial.position.pawns.b,
                },
            )
            .unwrap();
        assert_eq!(
            overlapping.to_position(&config),
            Err(NormalDuelError::InvalidPosition)
        );

        let mut overlapping_walls =
            SearchPosition::from_position(&config, &initial.position).unwrap();
        overlapping_walls
            .apply_unchecked(
                &config,
                &Action::Wall {
                    wall: "H-0-0".into(),
                },
            )
            .unwrap();
        overlapping_walls
            .apply_unchecked(
                &config,
                &Action::Wall {
                    wall: "H-0-1".into(),
                },
            )
            .unwrap();
        assert_eq!(
            overlapping_walls.to_position(&config),
            Err(NormalDuelError::InvalidWallGeometry)
        );

        let mut stale_edges = SearchPosition::from_position(&config, &initial.position).unwrap();
        stale_edges.blocked_down ^= 1;
        assert_eq!(
            stale_edges.to_position(&config),
            Err(NormalDuelError::InvalidPosition)
        );
    }

    #[test]
    fn adjudication_rejects_non_js_safe_counters_before_precedence() {
        let config = standard();
        for (count, ply) in [
            (0, 1),
            (1, 0),
            (MAX_JS_SAFE_INTEGER + 1, 1),
            (1, MAX_JS_SAFE_INTEGER + 1),
        ] {
            assert_eq!(
                adjudicate(&config, Some(Player::A), count, ply),
                Err(NormalDuelError::InvalidAdjudication)
            );
        }
        assert!(adjudicate(&config, None, MAX_JS_SAFE_INTEGER, MAX_JS_SAFE_INTEGER).is_ok());
    }

    #[test]
    fn miri_core_smoke() {
        let config = standard();
        let initial = create_initial_state(&config).unwrap();
        let mut prepared = PreparedGameState::from_game_state(&config, &initial).unwrap();
        let root_codes = compact_legal_codes(&config, &prepared);
        assert_eq!(root_codes.len, 131);

        let pawn_code = root_codes.iter().next().unwrap();
        let pawn_undo = prepared.apply_generated_code(&config, pawn_code).unwrap();
        assert!(!compact_legal_codes(&config, &prepared)
            .iter()
            .collect::<Vec<_>>()
            .is_empty());
        assert!(prepared.undo_generated(pawn_undo));

        let wall_code = root_codes
            .iter()
            .find(|&code| code >= config.cells())
            .unwrap();
        let wall_undo = prepared.apply_generated_code(&config, wall_code).unwrap();
        assert_eq!(prepared.position.stock.a, config.initial_stock.a - 1);
        assert!(prepared.undo_generated(wall_undo));
        assert_eq!(
            prepared.position.to_position(&config).unwrap(),
            initial.position,
            "compact apply/undo restores the exact root"
        );

        assert_eq!(
            fast_perft_with_options(&config, &initial, 1, PerftOptions::default()),
            Ok(131)
        );
    }

    #[test]
    fn zobrist_stock_domains_do_not_alias_arbitrary_stock_values() {
        let mut config = standard();
        config.initial_stock = Players { a: 2_000, b: 2_000 };
        let common = create_initial_state(&config).unwrap().position;
        let first = Position {
            stock: Players { a: 1_010, b: 5 },
            ..common.clone()
        };
        let second = Position {
            stock: Players { a: 1_005, b: 10 },
            ..common
        };
        assert_ne!(
            zobrist_key(&config, &first).unwrap(),
            zobrist_key(&config, &second).unwrap()
        );
    }

    #[test]
    fn seeded_wall_helper_matches_lcg32_reference_opening() {
        let mut blitz = standard();
        blitz.rows = 7;
        blitz.columns = 7;
        blitz.start = Players {
            a: Coord { r: 6, c: 3 },
            b: Coord { r: 0, c: 3 },
        };
        blitz.goal_rows = Players { a: 0, b: 6 };
        let after_one = seeded_wall_state(&blitz, 0, 1).unwrap();
        assert_eq!(after_one.position.walls, vec!["H-1-1"]);
    }

    #[test]
    fn state_validation_rejects_first_move_and_history_teleports() {
        let config = standard();
        let initial = create_initial_state(&config).unwrap();

        let goal_teleport_position = Position {
            pawns: Players {
                a: Coord { r: 0, c: 3 },
                b: config.start.b,
            },
            walls: Vec::new(),
            stock: config.initial_stock,
            turn: Player::B,
        };
        let goal_teleport_key = position_key(&config, &goal_teleport_position).unwrap();
        let mut teleport_counts = vec![
            RepetitionCount {
                position_key: initial.position_key.clone(),
                count: 1,
            },
            RepetitionCount {
                position_key: goal_teleport_key.clone(),
                count: 1,
            },
        ];
        teleport_counts
            .sort_by(|left, right| compare_position_keys(&left.position_key, &right.position_key));
        let forged_goal = GameState {
            position: goal_teleport_position,
            position_key: goal_teleport_key,
            ply: 1,
            history_start_ply: 0,
            repetition_counts: teleport_counts,
            outcome: Outcome::Win {
                winner: Player::A,
                reason: GoalReason::Goal,
            },
        };
        assert_eq!(
            validate_state(&config, &forged_goal),
            Err(NormalDuelError::InvalidState)
        );

        let after_a = apply_action(
            &config,
            &initial,
            &Action::Pawn {
                to: Coord { r: 7, c: 4 },
            },
        )
        .unwrap();
        let after_b = apply_action(
            &config,
            &after_a,
            &Action::Pawn {
                to: Coord { r: 1, c: 4 },
            },
        )
        .unwrap();
        let impossible_historical = Position {
            pawns: Players {
                a: Coord { r: 3, c: 3 },
                b: Coord { r: 5, c: 5 },
            },
            walls: Vec::new(),
            stock: config.initial_stock,
            turn: Player::B,
        };
        let impossible_key = position_key(&config, &impossible_historical).unwrap();
        let mut forged_history = after_b.clone();
        forged_history.repetition_counts = vec![
            RepetitionCount {
                position_key: initial.position_key,
                count: 1,
            },
            RepetitionCount {
                position_key: impossible_key,
                count: 1,
            },
            RepetitionCount {
                position_key: after_b.position_key.clone(),
                count: 1,
            },
        ];
        forged_history
            .repetition_counts
            .sort_by(|left, right| compare_position_keys(&left.position_key, &right.position_key));
        assert_eq!(
            validate_state(&config, &forged_history),
            Err(NormalDuelError::InvalidState)
        );

        let mut forged_reset = after_b;
        forged_reset.history_start_ply = forged_reset.ply;
        forged_reset.repetition_counts = vec![RepetitionCount {
            position_key: forged_reset.position_key.clone(),
            count: 1,
        }];
        assert_eq!(
            validate_state(&config, &forged_reset),
            Err(NormalDuelError::InvalidState)
        );
    }

    #[test]
    fn malformed_wall_actions_map_to_invalid_action() {
        let config = standard();
        let initial = create_initial_state(&config).unwrap();
        let malformed = Action::Wall {
            wall: "H-9-0".into(),
        };
        assert_eq!(
            encode_action(&config, &malformed),
            Err(NormalDuelError::InvalidAction)
        );
        assert_eq!(
            apply_action(&config, &initial, &malformed),
            Err(NormalDuelError::InvalidAction)
        );
        assert_eq!(
            apply_legal_action(&config, &initial, &malformed),
            Err(NormalDuelError::InvalidAction)
        );
        // A wall-query takes raw wall text, so malformed input remains a
        // negative wall result rather than an Action-shape error.
        assert_eq!(
            is_legal_wall(&config, &initial.position, "H-9-0"),
            Ok(false)
        );
    }

    #[test]
    fn apply_action_validates_config_before_terminal_precedence() {
        let config = standard();
        let mut terminal = create_initial_state(&config).unwrap();
        terminal.outcome = Outcome::Draw {
            reason: DrawReason::PlyCap,
        };
        let action = Action::Pawn {
            to: Coord { r: 7, c: 4 },
        };
        assert_eq!(
            apply_action(&config, &terminal, &action),
            Err(NormalDuelError::TerminalState)
        );

        let mut unsupported = config.clone();
        unsupported.ruleset = "normal-duel-v2".into();
        assert_eq!(
            apply_action(&unsupported, &terminal, &action),
            Err(NormalDuelError::InvalidConfig)
        );

        let mut invalid_dimensions = config;
        invalid_dimensions.rows = 0;
        invalid_dimensions.columns = 0;
        assert_eq!(
            apply_action(&invalid_dimensions, &terminal, &action),
            Err(NormalDuelError::InvalidConfig)
        );
    }

    #[test]
    fn json_boundary_accepts_exponent_form_safe_integers() {
        let expected = standard();
        let mut config_json = serde_json::to_value(&expected).unwrap();
        config_json["initialStock"]["A"] = serde_json::from_str("1e1").unwrap();
        config_json["start"]["A"]["r"] = serde_json::from_str("8e0").unwrap();
        config_json["plyCap"] = serde_json::from_str("2e2").unwrap();
        assert_eq!(validate_config_json(&config_json), Ok(expected));

        config_json["plyCap"] = serde_json::from_str("2.5").unwrap();
        assert_eq!(
            validate_config_json(&config_json),
            Err(NormalDuelError::InvalidConfig)
        );
    }
}
