"""
board.py — Stateless geometry module for AFWIP.

Responsibilities:
  - Absolute position mapping (relative BandID + Side → integer distance)
  - Band distance and range checks
  - Token spawn location validation
  - Movement reachability
  - WEZ intersection for missile defense
  - Contingency Location generation roll

No game state lives here. All functions are pure.
"""

from __future__ import annotations
from typing import Optional
from afwip.core.constants import (
    BandID,
    Side,
    TokenType,
    PostureType
)
from afwip.core.tokens import TOKEN_REGISTRY

ON_MAP_BANDS: frozenset[BandID] = frozenset({
    BandID.BAND_A,
    BandID.BAND_B,
    BandID.BAND_C,
    BandID.BAND_D,
    BandID.BAND_E,
})

CHINA_SHOOT_PATH: dict[BandID, frozenset[BandID]] = {
    BandID.PRC_AIRBASE: frozenset({BandID.BAND_E, BandID.BAND_D, BandID.BAND_C, BandID.BAND_B, BandID.BAND_A}),
    BandID.PRC_STANDOFF: frozenset({BandID.BAND_E, BandID.BAND_D, BandID.BAND_C, BandID.BAND_B, BandID.BAND_A}),
    BandID.BAND_E: frozenset({BandID.BAND_D, BandID.BAND_C, BandID.BAND_B, BandID.BAND_A}),
    BandID.BAND_D: frozenset({BandID.BAND_C, BandID.BAND_B, BandID.BAND_A}),
    BandID.BAND_C: frozenset({BandID.BAND_B, BandID.BAND_A}),
    BandID.BAND_B: frozenset({BandID.BAND_A}),
    BandID.BAND_A: frozenset(),
}

US_SHOOT_PATH: dict[BandID, frozenset[BandID]] = {
    BandID.US_AIRBASE: frozenset({BandID.BAND_A, BandID.BAND_B, BandID.BAND_C, BandID.BAND_D, BandID.BAND_E}),
    BandID.US_STANDOFF: frozenset({BandID.BAND_A, BandID.BAND_B, BandID.BAND_C, BandID.BAND_D, BandID.BAND_E}),
    BandID.US_CONTINGENCY_LOCATION: frozenset({BandID.BAND_A, BandID.BAND_B, BandID.BAND_C, BandID.BAND_D, BandID.BAND_E}),
    BandID.BAND_A: frozenset({BandID.BAND_B, BandID.BAND_C, BandID.BAND_D, BandID.BAND_E}),
    BandID.BAND_B: frozenset({BandID.BAND_C, BandID.BAND_D, BandID.BAND_E}),
    BandID.BAND_C: frozenset({BandID.BAND_D, BandID.BAND_E}),
    BandID.BAND_D: frozenset({BandID.BAND_E}),
    BandID.BAND_E: frozenset(),
}

ADJACENCY: dict[BandID, frozenset[BandID]] = {
    BandID.US_STANDOFF: frozenset({BandID.BAND_A, BandID.US_AIRBASE, BandID.US_CONTINGENCY_LOCATION}),
    BandID.US_AIRBASE: frozenset({BandID.BAND_A, BandID.US_STANDOFF}),
    BandID.US_CONTINGENCY_LOCATION: frozenset({BandID.BAND_A, BandID.US_STANDOFF}),
    BandID.BAND_A: frozenset({BandID.US_AIRBASE, BandID.US_CONTINGENCY_LOCATION, BandID.US_STANDOFF, BandID.BAND_B}),
    BandID.BAND_B: frozenset({BandID.BAND_A, BandID.BAND_C}),
    BandID.BAND_C: frozenset({BandID.BAND_B, BandID.BAND_D}),
    BandID.BAND_D: frozenset({BandID.BAND_C, BandID.BAND_E}),
    BandID.BAND_E: frozenset({BandID.BAND_D, BandID.PRC_AIRBASE, BandID.PRC_STANDOFF}),
    BandID.PRC_AIRBASE: frozenset({BandID.BAND_E, BandID.PRC_STANDOFF}),
    BandID.PRC_STANDOFF: frozenset({BandID.BAND_E, BandID.PRC_AIRBASE})
}


def _build_distance_table() -> dict[frozenset[BandID], int]:
    table: dict[frozenset[BandID], int] = {}
    for start in ADJACENCY:
        dist: dict[BandID, int] = {start: 0}
        queue: list[BandID] = [start]

        while queue:
            current = queue.pop(0)
            for neighbor in ADJACENCY[current]:
                if neighbor not in dist:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)

        for end, d in dist.items():
            key = frozenset({start, end})
            table[key] = d
    
    return table

DISTANCES = _build_distance_table()

# Location classification
 
US_BASE_LOCATIONS: frozenset[BandID] = frozenset({
    BandID.US_AIRBASE,
    BandID.US_CONTINGENCY_LOCATION,
})
 
PRC_BASE_LOCATIONS: frozenset[BandID] = frozenset({
    BandID.PRC_AIRBASE,
})
 
BASE_LOCATIONS: frozenset[BandID] = US_BASE_LOCATIONS | PRC_BASE_LOCATIONS
 
STANDOFF_LOCATIONS: frozenset[BandID] = frozenset({
    BandID.US_STANDOFF,
    BandID.PRC_STANDOFF,
})
 
ALL_GRAPH_LOCATIONS: frozenset[BandID] = (
    ON_MAP_BANDS | BASE_LOCATIONS | STANDOFF_LOCATIONS
)
 
# geometry
 
def band_distance(band_a: BandID, band_b: BandID) -> int:
    """
    Shortest path distance between two locations.
    """
    key = frozenset({band_a, band_b})
    if key not in DISTANCES:
        raise ValueError(f"No path between {band_a} and {band_b}")
    return DISTANCES[key]
 
 
def in_range(band_a: BandID, band_b: BandID, max_range: int) -> bool:
    """Return True if band_b is within max_range bands of band_a."""
    return band_distance(band_a, band_b) <= max_range
 
# Movement reachability
 
def reachable_bands(current_band: BandID, move_range: int) -> frozenset[BandID]:
    """
    All bands reachable from current_band within move_range steps.
    Uses DISTANCES table — O(n) scan over graph nodes but graph is tiny.
    Returns empty frozenset for move_range = 0 (immobile tokens: ADA).
    Includes US_CONTINGENCY_LOCATION — rules layer filters for voluntary moves.
    """
    if move_range == 0:
        return frozenset()
    return frozenset(
        band for band in ADJACENCY
        if band != current_band
        and DISTANCES.get(frozenset({current_band, band}), 999) <= move_range
    )
 
 
def valid_move_destinations(current_band: BandID, move_range: int, movement_bands: frozenset[BandID], squadron_location: BandID) -> frozenset[BandID]:
    """
    Reachable bands for a voluntary move action.
 
    Filters applied on top of raw reachability:
      - US_CONTINGENCY_LOCATION always excluded (Winchester return only)
      - STANDOFF excluded unless can_use_standoff=True (bombers and AEW only)
 
    can_use_standoff should be derived from token_score_type at call site:
        can_use_standoff = profile.token_score_type in {
            TokenScoreType.BOMBER, TokenScoreType.AEW
        }
    """
    unreachable_bases = BASE_LOCATIONS - {squadron_location}
    reachable = reachable_bands(current_band, move_range) - unreachable_bases
    reachable = reachable & movement_bands
    return reachable
 
# Missile defense / WEZ
def in_wez(defender_band: BandID, attacker_band: BandID, attacker_side: Side, target_band: BandID, wez_range: int) -> bool:
    """Returns whether missile defense is possible for defender_band, attacker_band corresponding to tokens."""
    path_map = US_SHOOT_PATH if attacker_side == Side.US else CHINA_SHOOT_PATH
    passed_over_bands = (
        path_map.get(attacker_band, frozenset()) - path_map.get(target_band, frozenset())
    ) | {target_band}

    return any(
        DISTANCES.get(frozenset({defender_band, band}), 999) <= wez_range
        for band in passed_over_bands
    )
    

# Spawn / placement validation
 
def valid_spawn_locations(token_type: TokenType, side: Side, posture: PostureType) -> frozenset[BandID]:
    """
    BandIDs where a token may be placed on squadron activation.
    Base set comes from TokenProfile.spawn_bands.
    Posture modifiers applied on top:
      - SURGE: removes US_CONTINGENCY_LOCATION for all US tokens
        ("Contingency Locations cannot be used")
    """

    profile = TOKEN_REGISTRY[token_type]
    bands: frozenset[BandID] = profile.spawn_bands

    if side == Side.US and posture == PostureType.SURGE:
        bands = bands - {BandID.US_CONTINGENCY_LOCATION}

    return bands


# Contingency Location generation roll
 
def contingency_tokens_generated(roll: int, squadron_size: int) -> int:
    """
    Tokens that successfully generate from US_CONTINGENCY_LOCATION.
    Result = min(roll, squadron_size).
    Grounded tokens remain at CL and are vulnerable to attack.
 
    Bypass: pass squadron_size for both args when:
      - ACE posture is active (always generate max)
      - Flying Crew Chief enabler is in play (always generate max for one squadron)
    """
    return min(roll, squadron_size)

# Side utilities
 
def opponent(side: Side) -> Side:
    """Return the opposing Side."""
    return Side.PRC if side == Side.US else Side.US
 
 
def own_airbase(side: Side) -> BandID:
    """Return the primary airbase BandID for the given side."""
    return BandID.US_AIRBASE if side == Side.US else BandID.PRC_AIRBASE
 
 
def own_standoff(side: Side) -> BandID:
    """Return the standoff BandID for the given side."""
    return BandID.US_STANDOFF if side == Side.US else BandID.PRC_STANDOFF
 
 
def own_base_locations(side: Side) -> frozenset[BandID]:
    """Return all base locations belonging to side."""
    return US_BASE_LOCATIONS if side == Side.US else PRC_BASE_LOCATIONS
 
 
def is_own_base(band: BandID, side: Side) -> bool:
    """Return True if band is a base location belonging to side."""
    return band in own_base_locations(side)
 
 
def is_enemy_base(band: BandID, side: Side) -> bool:
    """Return True if band is a base location belonging to the opposing side."""
    return is_own_base(band, opponent(side))
 
 
def is_on_map(band: BandID) -> bool:
    """Return True if band is a standard on-map range band."""
    return band in ON_MAP_BANDS