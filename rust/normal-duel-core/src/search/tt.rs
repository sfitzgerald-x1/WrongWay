use crate::{Config, PreparedSearchIdentity};

use super::budget::SearchBudget;

// Keeping each allocation modest lets a deadline be observed while a large
// table is being set up, rather than only after one monolithic allocation.
const BUCKETS_PER_CHUNK: usize = 1 << 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Entry {
    pub(crate) identity: PreparedSearchIdentity,
    pub(crate) depth: u8,
    /// Root-relative score. Tables are created per search invocation and the
    /// verified identity contains absolute ply, so one identity cannot be
    /// reused at a different root-relative mate distance. No cross-root TT is
    /// retained; an implementation that adds one must normalize mate scores.
    pub(crate) score: i32,
    pub(crate) bound: Option<Bound>,
    canonical_action: Option<u16>,
    generation: u16,
}

impl Entry {
    pub(crate) fn best_action(
        self,
        config: &Config,
        identity: PreparedSearchIdentity,
    ) -> Option<usize> {
        let canonical = usize::from(self.canonical_action?);
        Some(if identity.mirrored() {
            mirror_code(config, canonical)
        } else {
            canonical
        })
    }
}

pub(crate) struct TranspositionTable {
    buckets: Vec<Box<[[Option<Entry>; 2]]>>,
    bucket_count: usize,
    generation: u16,
}

impl TranspositionTable {
    #[cfg(test)]
    pub(crate) fn new(capacity: usize) -> Self {
        Self::build(capacity, || false).expect("an unconditional TT build cannot stop")
    }

    /// Builds the table in bounded chunks so a wall-clock search can abandon
    /// setup promptly. Node-budget mode observes the same polls but, because
    /// setup does not add nodes, remains deterministic.
    pub(crate) fn new_with_budget<B: SearchBudget>(
        capacity: usize,
        budget: &mut B,
        nodes: u64,
    ) -> Option<Self> {
        Self::build(capacity, || budget.exhausted(nodes))
    }

    fn build(capacity: usize, mut exhausted: impl FnMut() -> bool) -> Option<Self> {
        debug_assert!(capacity >= 2);
        // Capacity is an upper bound on entries. Odd capacities round down,
        // retaining two real ways in every allocated bucket.
        let bucket_count = capacity / 2;
        let mut buckets = Vec::with_capacity(bucket_count.div_ceil(BUCKETS_PER_CHUNK));
        let mut remaining = bucket_count;
        while remaining != 0 {
            if exhausted() {
                return None;
            }
            let chunk_len = remaining.min(BUCKETS_PER_CHUNK);
            buckets.push(vec![[None, None]; chunk_len].into_boxed_slice());
            remaining -= chunk_len;
        }
        Some(Self {
            buckets,
            bucket_count,
            generation: 0,
        })
    }

    pub(crate) fn next_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    fn index(&self, key: u64) -> usize {
        key as usize % self.bucket_count
    }

    fn bucket(&self, index: usize) -> &[Option<Entry>; 2] {
        &self.buckets[index / BUCKETS_PER_CHUNK][index % BUCKETS_PER_CHUNK]
    }

    fn bucket_mut(&mut self, index: usize) -> &mut [Option<Entry>; 2] {
        &mut self.buckets[index / BUCKETS_PER_CHUNK][index % BUCKETS_PER_CHUNK]
    }

    pub(crate) fn probe(&self, identity: PreparedSearchIdentity) -> Option<Entry> {
        self.bucket(self.index(identity.key()))
            .iter()
            .flatten()
            .find(|entry| entry.identity == identity)
            .copied()
    }

    fn usefulness(entry: Entry) -> (u8, u8) {
        let bound = match entry.bound {
            None => 0,
            Some(Bound::Upper | Bound::Lower) => 1,
            Some(Bound::Exact) => 2,
        };
        (entry.depth, bound)
    }

    fn replacement_lane(
        &self,
        bucket: [Option<Entry>; 2],
        identity: PreparedSearchIdentity,
    ) -> usize {
        if bucket[0].is_some_and(|entry| entry.identity == identity) {
            return 0;
        }
        if bucket[1].is_some_and(|entry| entry.identity == identity) {
            return 1;
        }
        if bucket[0].is_none() {
            return 0;
        }
        if bucket[1].is_none() {
            return 1;
        }
        let first = bucket[0].expect("occupied lane");
        let second = bucket[1].expect("occupied lane");
        match (
            first.generation != self.generation,
            second.generation != self.generation,
        ) {
            (true, false) => 0,
            (false, true) => 1,
            (true, true) | (false, false) => {
                if Self::usefulness(first) <= Self::usefulness(second) {
                    0
                } else {
                    1
                }
            }
        }
    }

    pub(crate) fn store(
        &mut self,
        config: &Config,
        identity: PreparedSearchIdentity,
        depth: u8,
        score: i32,
        bound: Option<Bound>,
        best_action: Option<usize>,
    ) {
        let index = self.index(identity.key());
        let lane = self.replacement_lane(*self.bucket(index), identity);
        let canonical_action = best_action.map(|code| {
            let canonical = if identity.mirrored() {
                mirror_code(config, code)
            } else {
                code
            };
            u16::try_from(canonical).expect("normal-duel policy fits in u16")
        });
        self.bucket_mut(index)[lane] = Some(Entry {
            identity,
            depth,
            score,
            bound,
            canonical_action,
            generation: self.generation,
        });
    }
}

pub(crate) fn mirror_code(config: &Config, code: usize) -> usize {
    if code < config.cells() {
        let row = code / usize::from(config.columns);
        let column = code % usize::from(config.columns);
        return row * usize::from(config.columns) + (usize::from(config.columns) - 1 - column);
    }
    let anchors = config.anchors_per_axis();
    let offset = code - config.cells();
    let orientation_offset = if offset >= anchors { anchors } else { 0 };
    let anchor = offset % anchors;
    let columns = usize::from(config.columns - 1);
    let row = anchor / columns;
    let column = anchor % columns;
    config.cells() + orientation_offset + row * columns + (columns - 1 - column)
}

#[cfg(test)]
mod tests {
    use super::{mirror_code, Bound, TranspositionTable};
    use crate::{
        apply_legal_action, create_initial_state, decode_action, Config, Coord, Player, Players,
        PreparedGameState, JUMP_RULE, REPETITION_THRESHOLD, RULESET,
    };

    fn config() -> Config {
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
            repetition_threshold: REPETITION_THRESHOLD,
            ply_cap: 512,
            first_player: Player::A,
        }
    }

    #[test]
    fn action_mirror_is_an_involution_for_full_policy() {
        let config = config();
        for code in 0..config.policy_size() {
            assert_eq!(mirror_code(&config, mirror_code(&config, code)), code);
        }
    }

    #[test]
    fn canonical_entry_recovers_the_best_action_in_local_orientation() {
        let config = config();
        let initial = create_initial_state(&config).unwrap();
        let left =
            apply_legal_action(&config, &initial, &decode_action(&config, 75).unwrap()).unwrap();
        let right =
            apply_legal_action(&config, &initial, &decode_action(&config, 77).unwrap()).unwrap();
        let left = PreparedGameState::from_game_state(&config, &left).unwrap();
        let right = PreparedGameState::from_game_state(&config, &right).unwrap();
        let left_identity = left.search_identity(&config);
        let right_identity = right.search_identity(&config);
        assert_eq!(left_identity, right_identity);
        assert_ne!(left_identity.mirrored(), right_identity.mirrored());

        let mut table = TranspositionTable::new(64);
        table.store(&config, left_identity, 3, 42, Some(Bound::Exact), Some(3));
        let recovered = table
            .probe(right_identity)
            .unwrap()
            .best_action(&config, right_identity);
        assert_eq!(recovered, Some(5));
    }

    #[test]
    fn bounds_are_disabled_after_pawn_history_and_reenabled_by_a_wall() {
        let config = config();
        let initial = create_initial_state(&config).unwrap();
        let initial_prepared = PreparedGameState::from_game_state(&config, &initial).unwrap();
        assert!(initial_prepared.search_identity(&config).bounds_reusable());

        let pawn =
            apply_legal_action(&config, &initial, &decode_action(&config, 75).unwrap()).unwrap();
        let pawn_prepared = PreparedGameState::from_game_state(&config, &pawn).unwrap();
        assert!(!pawn_prepared.search_identity(&config).bounds_reusable());

        let wall_code = crate::legal_action_codes(&config, &pawn)
            .unwrap()
            .into_iter()
            .find(|&code| code >= config.cells())
            .unwrap();
        let wall = apply_legal_action(&config, &pawn, &decode_action(&config, wall_code).unwrap())
            .unwrap();
        let wall_prepared = PreparedGameState::from_game_state(&config, &wall).unwrap();
        assert!(wall_prepared.search_identity(&config).bounds_reusable());
    }

    #[test]
    fn oracle_identity_distinguishes_same_board_and_ply_with_different_history() {
        let config = config();
        let initial = create_initial_state(&config).unwrap();
        let play = |codes: [usize; 4]| {
            let mut state = initial.clone();
            for code in codes {
                state = apply_legal_action(&config, &state, &decode_action(&config, code).unwrap())
                    .unwrap();
            }
            PreparedGameState::from_game_state(&config, &state).unwrap()
        };
        let left_cycle = play([75, 3, 76, 4]);
        let right_cycle = play([77, 5, 76, 4]);
        assert_eq!(
            left_cycle.search_identity(&config),
            right_cycle.search_identity(&config)
        );
        assert_ne!(
            left_cycle.oracle_identity(&config),
            right_cycle.oracle_identity(&config)
        );
        assert!(!left_cycle.search_identity(&config).bounds_reusable());
        assert!(!right_cycle.search_identity(&config).bounds_reusable());
    }

    #[test]
    fn two_way_bucket_keeps_two_verified_colliding_identities() {
        let config = config();
        let initial = create_initial_state(&config).unwrap();
        let left =
            apply_legal_action(&config, &initial, &decode_action(&config, 75).unwrap()).unwrap();
        let mut initial_identity = PreparedGameState::from_game_state(&config, &initial)
            .unwrap()
            .search_identity(&config);
        let mut left_identity = PreparedGameState::from_game_state(&config, &left)
            .unwrap()
            .search_identity(&config);
        // Force a hash collision while retaining distinct structural
        // verification payloads.
        left_identity.key = initial_identity.key;
        assert_ne!(initial_identity, left_identity);

        let mut table = TranspositionTable::new(2);
        table.store(
            &config,
            initial_identity,
            4,
            17,
            Some(Bound::Exact),
            Some(75),
        );
        table.store(&config, left_identity, 2, -9, Some(Bound::Lower), Some(3));
        assert_eq!(table.probe(initial_identity).unwrap().score, 17);
        assert_eq!(table.probe(left_identity).unwrap().score, -9);

        // A third identity in the same bucket evicts the shallower, less
        // useful lane while the deeper exact result survives.
        let forward =
            apply_legal_action(&config, &initial, &decode_action(&config, 67).unwrap()).unwrap();
        let mut forward_identity = PreparedGameState::from_game_state(&config, &forward)
            .unwrap()
            .search_identity(&config);
        forward_identity.key = initial_identity.key;
        table.store(&config, forward_identity, 3, 5, Some(Bound::Upper), Some(4));
        assert!(table.probe(initial_identity).is_some());
        assert!(table.probe(left_identity).is_none());
        assert!(table.probe(forward_identity).is_some());

        // A matching identity is updated in place, never displacing its
        // colliding neighbor.
        table.store(
            &config,
            initial_identity,
            5,
            23,
            Some(Bound::Exact),
            Some(77),
        );
        assert_eq!(table.probe(initial_identity).unwrap().score, 23);
        assert!(table.probe(forward_identity).is_some());

        // Keep the compiler honest that the test mutates only the opaque hash
        // field, not structural verification.
        initial_identity.key = initial_identity.key.wrapping_add(1);
        assert!(table.probe(initial_identity).is_none());
    }

    #[test]
    fn stale_generation_is_replaced_before_current_useful_entries() {
        let config = config();
        let initial = create_initial_state(&config).unwrap();
        let child = |code| {
            let state =
                apply_legal_action(&config, &initial, &decode_action(&config, code).unwrap())
                    .unwrap();
            PreparedGameState::from_game_state(&config, &state)
                .unwrap()
                .search_identity(&config)
        };
        let old = PreparedGameState::from_game_state(&config, &initial)
            .unwrap()
            .search_identity(&config);
        let survivor = child(75);
        let incoming = child(67);
        let mut table = TranspositionTable::new(2);
        table.store(&config, old, 8, 1, Some(Bound::Exact), Some(3));
        table.next_generation();
        table.store(&config, survivor, 1, 2, Some(Bound::Upper), Some(5));
        table.store(&config, incoming, 2, 3, Some(Bound::Lower), Some(4));
        assert!(table.probe(old).is_none());
        assert!(table.probe(survivor).is_some());
        assert!(table.probe(incoming).is_some());
    }
}
