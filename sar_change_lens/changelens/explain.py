"""'Why was this area flagged?' - explanations built ONLY from measured evidence.

Default: a deterministic template. Every number in it comes from the detector.
Optional: an LLM (Claude) may rewrite the template in plainer language, but its
output is checked: if it contains any number that is not in the evidence, or
drops the disclaimer, it is thrown away and the template is shown instead.
The LLM never sees imagery and never decides what is flagged.
"""

from __future__ import annotations

import json
import re

DISCLAIMER_SENTENCE = ("This is an automated screening result from a student prototype, "
                       "not a confirmed flood or an official warning.")

FLOOD_ALTERNATIVES = ("radar shadow on steep slopes, wet snow, freshly flooded rice paddies "
                      "or irrigation, very smooth surfaces such as tarmac or dry sand, or "
                      "wind-free water that was already present")
FOREST_ALTERNATIVES = ("drier soil or vegetation after rain, harvest of crops or plantations, "
                       "seasonal leaf change, or freeze/thaw")


def _fmt(value, nd=1, unit=""):
    return "n/a" if value is None else f"{value:.{nd}f}{unit}"


def evidence_for(props: dict, scene: dict) -> dict:
    """Collect the facts one explanation may use (also what an LLM is given)."""
    return {
        "id": props.get("id"),
        "hazard": props.get("hazard"),
        "label": props.get("label"),
        "area_km2": props.get("area_km2"),
        "pixels": props.get("pixels"),
        "mean_delta_db": props.get("mean_delta_db"),
        "mean_pre_db": props.get("mean_pre_db"),
        "mean_post_db": props.get("mean_post_db"),
        "mean_hand_m": props.get("mean_hand_m"),
        "confidence": props.get("confidence"),
        "confidence_label": props.get("confidence_label"),
        "pre_date": scene.get("pre_date"),
        "post_date": scene.get("post_date"),
        "polarization": scene.get("polarization"),
        "dataset": scene.get("dataset"),
        "threshold_db": scene.get("threshold_db"),
        "delta_db_required": scene.get("delta_db_required"),
        "hand_max_m": scene.get("hand_max_m"),
        "scene_issues": scene.get("scene_issues", []),
    }


def template_explanation(ev: dict) -> str:
    pol = ev.get("polarization") or "radar"
    lines = []
    if ev.get("hazard") == "forest_disturbance":
        lines.append(f"{ev['id']} is a candidate forest disturbance (experimental).")
        lines.append(f"- Cross-polarized ({pol}) backscatter, which comes mostly from branches and "
                     f"trunks, fell by {_fmt(abs(ev['mean_delta_db'] or 0))} dB between "
                     f"{ev['pre_date']} and {ev['post_date']} (the check needs "
                     f"{_fmt(ev['delta_db_required'])} dB).")
        alternatives = FOREST_ALTERNATIVES
    else:
        lines.append(f"{ev['id']} was flagged as a possible flood-related change.")
        lines.append(f"- Radar signal: {pol} backscatter fell by {_fmt(abs(ev['mean_delta_db'] or 0))} dB "
                     f"on average between {ev['pre_date']} and {ev['post_date']} (the detector needs "
                     f"at least {_fmt(ev['delta_db_required'])} dB).")
        lines.append(f"- Water-like afterwards: the post-event value, {_fmt(ev['mean_post_db'])} dB, is "
                     f"darker than this scene's learned water/land boundary of {_fmt(ev['threshold_db'])} dB. "
                     "Calm open water reflects the radar pulse away from the satellite, so it looks dark.")
        lines.append(f"- Not water before: the pre-event value was {_fmt(ev['mean_pre_db'])} dB, "
                     "in the land-like range.")
        alternatives = FLOOD_ALTERNATIVES
    if ev.get("mean_hand_m") is not None:
        lines.append(f"- Terrain: on average {_fmt(ev['mean_hand_m'])} m above the nearest drainage "
                     f"(areas above {_fmt(ev['hand_max_m'], 0)} m are masked).")
    else:
        lines.append("- Terrain: not checked (no height-above-drainage layer was supplied).")
    lines.append(f"- Size: {_fmt(ev['area_km2'], 3)} km2 ({ev['pixels']} pixels).")
    lines.append(f"Confidence: {ev['confidence_label']} ({_fmt(ev['confidence'], 2)}). This is an "
                 "evidence score combining the checks above, not a probability.")
    if ev.get("scene_issues"):
        lines.append("Scene caveats: " + "; ".join(ev["scene_issues"]) + ".")
    lines.append(f"Other things that can look like this: {alternatives}.")
    lines.append(DISCLAIMER_SENTENCE)
    return "\n".join(lines)


_NUM = re.compile(r"\d+(?:\.\d+)?")


def _allowed_numbers(ev: dict) -> set[str]:
    allowed = set()

    def add(v):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            for nd in range(0, 4):
                allowed.add(f"{abs(v):.{nd}f}")
            allowed.add(str(abs(v)))
        elif isinstance(v, str):
            allowed.update(_NUM.findall(v))
        elif isinstance(v, (list, tuple)):
            for x in v:
                add(x)

    for value in ev.values():
        add(value)
    allowed.update(_NUM.findall(DISCLAIMER_SENTENCE + template_explanation(ev)))
    return allowed


def check_grounded(text: str, ev: dict) -> list[str]:
    """Problems that make an LLM rewrite unacceptable (empty list = acceptable)."""
    problems = []
    unknown = sorted(set(_NUM.findall(text)) - _allowed_numbers(ev))
    if unknown:
        problems.append(f"numbers not in the evidence: {unknown}")
    if DISCLAIMER_SENTENCE not in text:
        problems.append("disclaimer sentence missing or altered")
    return problems


SYSTEM_PROMPT = (
    "You rewrite short explanations of satellite-radar screening results for non-specialist "
    "readers such as local planners and students. Use only the facts in the JSON you are given. "
    "Do not add place names, causes, damage, casualties, rainfall, or any number that is not in "
    "the JSON; keep numbers exactly as written. Say 'possible' or 'candidate', never 'confirmed'. "
    "Keep it under 130 words, and end with this sentence copied exactly: " + DISCLAIMER_SENTENCE
)


def llm_explanation(ev: dict, client=None, model: str = "claude-opus-5") -> tuple[str, dict]:
    """Optional plain-language rewrite with Claude. Returns (text, info).

    Falls back to the template on any error, refusal, or grounding failure, so
    the dashboard never depends on the network. Needs `pip install anthropic`
    and credentials (ANTHROPIC_API_KEY or `ant auth login`).
    """
    template = template_explanation(ev)
    try:
        import anthropic
    except ImportError:
        return template, {"source": "template", "reason": "anthropic package not installed"}
    try:
        client = client or anthropic.Anthropic()
        response = client.beta.messages.create(
            model=model,
            max_tokens=2000,
            output_config={"effort": "low"},  # a short, constrained rewrite
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # re-run on Anthropic's recommended model if declined
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content":
                       "Evidence JSON:\n" + json.dumps(ev, default=str) +
                       "\n\nDraft explanation to rewrite:\n" + template}],
        )
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        return template, {"source": "template", "reason": f"API error: {exc.__class__.__name__}"}
    except anthropic.AnthropicError as exc:  # e.g. no credentials configured
        return template, {"source": "template", "reason": f"client error: {exc.__class__.__name__}"}

    if response.stop_reason == "refusal":
        return template, {"source": "template", "reason": "model declined"}
    text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
    problems = check_grounded(text, ev)
    if not text or problems:
        return template, {"source": "template", "reason": "LLM output rejected", "problems": problems}
    return text, {"source": "llm", "model": response.model}
