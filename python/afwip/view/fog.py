"""
fog.py — Viewer-relative visibility predicates for the view layer.

The single place the web stack decides what a viewer may see. Semantics match
the engine's information model as already encoded in afwip/env.py and
afwip/tui.py (pinned by lockstep tests in tests/test_view.py):

  - Token position and acquisition value are public; identity/flags are hidden
    until the token is acquired (flipped face-up for the ATO cycle).
  - An enemy Squadron Card is known once it has EVER been activated (a public
    act; identity is never re-hidden) or once destroyed. Face-down cards are
    reported only as aggregate counts.
  - Enemy enablers in hand are hidden unless revealed during Intel; cards
    played/spent openly are public.

`viewer=None` means the PUBLIC view (both sides' hidden info masked) — used
for the hotseat handoff screen. `reveal=True` is the omniscient debug/spectator
view.
"""

from __future__ import annotations

from typing import Optional

from afwip.core.constants import Side
from afwip.core.state import CardZone, EnablerCardState, SquadronState, TokenInstance


def token_known(token: TokenInstance, viewer: Optional[Side], reveal: bool) -> bool:
    """Whether the viewer sees this token's identity and status flags."""
    if reveal:
        return True
    if viewer is None:                      # public view: only acquired tokens
        return token.acquired
    return token.side == viewer or token.acquired


def squadron_known(squad: SquadronState, viewer: Optional[Side], reveal: bool) -> bool:
    """Whether the viewer sees this Squadron Card's identity."""
    if reveal:
        return True
    if viewer is not None and squad.side == viewer:
        return True
    return squad.ever_activated or squad.is_destroyed


def hand_card_known(card: EnablerCardState, viewer: Optional[Side], reveal: bool) -> bool:
    """Whether the viewer sees this in-hand enabler's identity."""
    if reveal:
        return True
    if viewer is not None and card.side == viewer:
        return True
    return card.revealed_to_opponent


def played_card_public(card: EnablerCardState) -> bool:
    """Cards played, spent, or discarded openly are public information."""
    return card.zone in (CardZone.PLAYED, CardZone.ACTIVE,
                         CardZone.REMOVED, CardZone.DISCARDED)
