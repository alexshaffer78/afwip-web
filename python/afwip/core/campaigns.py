"""
campaigns.py — Pre-defined campaign definitions and draft-legality rules.

Each Campaign (from the Player Guide) constrains setup: how many ATO cycles,
which Mission and Posture cards are allowed, whether Enabler Cards are drafted at
all, any forced squadrons, and which cards/units are banned. `RulesEngine` reads
the profile for `GameState.campaign` to validate `setup_missions` /
`select_posture`.

Bans are expressed structurally so they stay in sync with the registries:
squadron bans by generated `TokenType`, enabler bans by `EnablerClass` and/or a
name substring (to catch e.g. US "SOF ..." cards that are filed under AIR_FORCE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from afwip.core.constants import (
    Side,
    TokenType,
    PostureType,
    MissionType,
    EnablerClass,
    CAMPAIGN_ATO_CYCLES,
)
from afwip.core.cards import SQUADRON_REGISTRY, ENABLER_REGISTRY


# Units referenced by campaign bans.
FIFTH_GEN_AIRCRAFT: frozenset[TokenType] = frozenset({
    TokenType.F_22, TokenType.F_35A, TokenType.J_20B,
})
BOMBERS: frozenset[TokenType] = frozenset({TokenType.B_52, TokenType.H_6K})
LONG_RANGE_BOMBERS: frozenset[TokenType] = frozenset({TokenType.B_52, TokenType.H_6K})


@dataclass(frozen=True)
class CampaignProfile:
    number: int
    name: str
    ato_cycles: int

    # None means "all allowed".
    allowed_missions: Optional[frozenset[MissionType]] = None
    allowed_postures: Optional[frozenset[PostureType]] = None

    enablers_allowed: bool = True
    # side -> squadron card_ids each side must INCLUDE in its draft (Campaign 1);
    # the rest of the Standard-posture roster is drafted normally around them.
    forced_squadrons: dict[Side, frozenset[int]] = field(default_factory=dict)

    # Bans.
    banned_squadron_tokens: frozenset[TokenType] = frozenset()
    banned_enabler_classes: frozenset[EnablerClass] = frozenset()
    banned_enabler_name_substrings: tuple[str, ...] = ()

    # Scoring extras (handled in the scoring layer; recorded here for reference).
    air_kill_bonus_vp: bool = False          # Campaign 4: +1 VP per air kill, max +2/ATO
    intact_squadron_bonus: bool = False      # Campaign 5: +1 VP per intact squadron at end

    # -- legality helpers ---------------------------------------------------

    def mission_allowed(self, mission_type: MissionType) -> bool:
        return self.allowed_missions is None or mission_type in self.allowed_missions

    def posture_allowed(self, posture_type: PostureType) -> bool:
        return self.allowed_postures is None or posture_type in self.allowed_postures

    def squadron_allowed(self, card_id: int) -> bool:
        return SQUADRON_REGISTRY[card_id].token_type not in self.banned_squadron_tokens

    def enabler_allowed(self, card_id: int) -> bool:
        profile = ENABLER_REGISTRY[card_id]
        if profile.enabler_class in self.banned_enabler_classes:
            return False
        upper = profile.name.upper()
        return not any(sub.upper() in upper for sub in self.banned_enabler_name_substrings)


CAMPAIGN_REGISTRY: dict[int, CampaignProfile] = {
    1: CampaignProfile(
        number=1,
        name="Meeting Engagement",
        ato_cycles=CAMPAIGN_ATO_CYCLES[1],
        allowed_missions=frozenset({MissionType.ATTRITION}),
        allowed_postures=frozenset({PostureType.STANDARD}),
        enablers_allowed=False,
        # US must include the F-16 squadron (card 10); PRC the J-10 squadron
        # (card 60). Both still draft a normal Standard-posture roster around it.
        forced_squadrons={Side.US: frozenset({10}), Side.PRC: frozenset({60})},
    ),
    2: CampaignProfile(
        number=2,
        name="Tournament",
        ato_cycles=CAMPAIGN_ATO_CYCLES[2],
        allowed_missions=frozenset({MissionType.ATTRITION}),
        allowed_postures=frozenset({PostureType.STANDARD}),
    ),
    3: CampaignProfile(
        number=3,
        name="Prolonged Combat",
        ato_cycles=CAMPAIGN_ATO_CYCLES[3],
    ),
    4: CampaignProfile(
        number=4,
        name="The World Watches",
        ato_cycles=CAMPAIGN_ATO_CYCLES[4],
        # No long-range missile or SOF enablers; no long-range bomber squadrons.
        banned_squadron_tokens=LONG_RANGE_BOMBERS,
        banned_enabler_classes=frozenset({EnablerClass.MISSILE, EnablerClass.SOF}),
        # "SOF" catches US SOF cards filed under AIR_FORCE; "TOMAHAWK" is the
        # US long-range-missile enabler (the PRC ones are the MISSILE class).
        banned_enabler_name_substrings=("SOF", "TOMAHAWK"),
        air_kill_bonus_vp=True,
    ),
    5: CampaignProfile(
        number=5,
        name="Reserves",
        ato_cycles=CAMPAIGN_ATO_CYCLES[5],
        # No 5th-gen aircraft, no bombers, no long-range ADA for the PRC.
        banned_squadron_tokens=FIFTH_GEN_AIRCRAFT | BOMBERS | frozenset({TokenType.LONG_RANGE_ADA_PRC}),
        intact_squadron_bonus=True,
    ),
}


def get_campaign(number: int) -> CampaignProfile:
    """Return the CampaignProfile for a campaign number (falls back to free-play)."""
    if number in CAMPAIGN_REGISTRY:
        return CAMPAIGN_REGISTRY[number]
    return CampaignProfile(number=number, name="Free Play",
                           ato_cycles=CAMPAIGN_ATO_CYCLES.get(number, 1))
