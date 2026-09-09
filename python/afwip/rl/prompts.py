"""
prompts.py — Build the token-efficient system prompt for LLM trajectory play.

The system prompt is: a condensed rules brief + the side's doctrine essentials +
the output-format contract. These live as small markdown files under
`afwip/rl/prompts/` (derived from `pme_materials/`, NOT the full 80-90 KB
doctrine corpora — kept short on purpose). The full state + legal actions are
sent per turn as the user message (see `afwip.rl.state_text`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from afwip.core.constants import Side, TokenScoreType
from afwip.core.cards import ENABLER_REGISTRY
from afwip.core.tokens import TOKEN_REGISTRY

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

_DOCTRINE_FILE = {
    Side.US: "usaf_doctrine_brief.md",
    Side.PRC: "prc_doctrine_brief.md",
}


def _read(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def token_reference() -> str:
    """Per-unit capability reference (what each token can ATTACK), from the
    engine so it never drifts. This is the fix for air-to-air vs surface-strike
    confusion (e.g. drones/bombers have NO air-to-air and cannot screen)."""
    lines = ["# Token capability reference — what each unit can ATTACK.",
             "A unit can attack ONLY the way listed. 'air-to-air' = can shoot "
             "enemy AIRCRAFT; 'surface-strike rN' = can hit ships / bases / "
             "standoff aircraft at range N. **A unit with NO air-to-air cannot "
             "screen against or shoot down aircraft** (bombers, drones, AEW). "
             "Standoff aircraft (bombers/AEW loitering in `*_STANDOFF`) can be hit "
             "by a surface/standoff strike OR by air-to-air from the ADJACENT "
             "front band — standoff is a 6th band one step beyond it, so an r1 "
             "fighter must be in that band to shoot in."]
    for tt in TOKEN_REGISTRY:
        p = TOKEN_REGISTRY[tt]
        st = p.token_score_type
        standoff = any("STANDOFF" in b.name for b in (p.movement_bands or []))
        head = f"{tt.value} ({st.name.lower()}, x{p.token_count}, av{p.acquisition_value}):"
        if st == TokenScoreType.ADA:
            caps = []
            if p.air_atk_range is not None:
                caps.append(f"air-to-air r{p.air_atk_range} (on your turn, shoots "
                            f"down an acquired enemy aircraft in range — single-hit "
                            f"kill; not passive)")
            if p.has_missile_defense:
                caps.append("missile defense (also disadvantages enemy attacks "
                            "crossing its WEZ)")
            desc = ("static air-defense, no surface/ship strike: "
                    + " + ".join(caps)
                    + "; itself killed only by a surface/base strike, never air-to-air")
        elif st == TokenScoreType.AEW:
            desc = "sensor — no attack" + ("; loiters in standoff" if standoff else "")
        elif st == TokenScoreType.EC_130:
            desc = "electronic warfare — no attack"
        elif st == TokenScoreType.UAS and p.surf_atk_range is None:
            desc = "recon drone — no attack"
        else:
            caps = []
            if p.air_atk_range is not None:
                caps.append("air-to-air")
            if p.surf_atk_range is not None:
                caps.append(f"surface-strike r{p.surf_atk_range}")
            if p.has_missile_defense:
                caps.append("missile defense")
            desc = " + ".join(caps) if caps else "no attack"
            if p.air_atk_range is None and p.surf_atk_range is not None:
                desc += " (NO air-to-air — cannot screen or shoot aircraft)"
            if standoff:
                desc += "; loiters in standoff"
        lines.append(f"{head} {desc}")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def enabler_reference() -> str:
    """A compact reference of every Enabler Card's effect, generated from the
    engine so it never drifts. Both sides are listed (you draft yours and face
    theirs). Small (~1.3k tokens) and part of the cached system prompt."""
    lines = ["# Enabler Card reference — `id NAME [type]: effect`.",
             "Play at most ONE per turn (responses excepted). "
             "'Play immediately after …' = a RESPONSE card: hold it until that "
             "trigger and play it out of turn. Cyber Rate 4 = instant win.",
             "[type] tags: **single** = spent for the whole campaign once played "
             "(don't waste it — play only when it does real work; otherwise SAVE "
             "it for a later ATO); **multi** = returns to your hand every ATO, so "
             "use it freely. **enduring** = its effect lasts the whole ATO cycle."]
    for side in (Side.US, Side.PRC):
        lines.append(f"\n## {side.value} enablers")
        for cid in sorted(ENABLER_REGISTRY):
            p = ENABLER_REGISTRY[cid]
            if p.side != side:
                continue
            tags = ["single" if p.single_use else "multi"]
            if p.is_enduring:
                tags.append("enduring")
            if p.is_response:
                tags.append("response")
            eff = " ".join((p.effect_text or "").split())
            lines.append(f"{cid} {p.name} [{', '.join(tags)}]: {eff}")
    return "\n".join(lines)


@lru_cache(maxsize=4)
def system_prompt(side: Side) -> str:
    """The full system prompt for `side`: rules + that side's doctrine + the
    Enabler reference + the output format.

    Stable per side (cached) so OpenAI prompt-caching discounts it — which is why
    the full Enabler reference is affordable to include.
    """
    side = side if isinstance(side, Side) else Side(side)
    return "\n\n---\n\n".join([
        f"You are the {side.value} commander in AFWIP. Play doctrine-aligned, "
        f"legal moves.",
        _read("rules_brief.md"),
        _read(_DOCTRINE_FILE[side]),
        token_reference(),
        enabler_reference(),
        _read("output_format.md"),
    ])


def approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for budgeting/logging."""
    return max(1, len(text) // 4)
