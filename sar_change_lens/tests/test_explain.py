"""Explanations must use only measured evidence."""

from types import SimpleNamespace

from changelens.explain import (DISCLAIMER_SENTENCE, check_grounded, evidence_for, llm_explanation,
                                template_explanation)

PROPS = {"id": "FL-001", "hazard": "flood", "label": "Possible flood-related change", "area_km2": 1.4844,
         "pixels": 3711, "mean_delta_db": -13.59, "mean_pre_db": -8.02, "mean_post_db": -21.61,
         "mean_hand_m": 6.8, "confidence": 0.7, "confidence_label": "High"}
SCENE = {"pre_date": "2026-07-01", "post_date": "2026-07-13", "polarization": "HH",
         "dataset": "NISAR L2 GCOV", "threshold_db": -16.15, "delta_db_required": 3.0,
         "hand_max_m": 15.0, "scene_issues": []}
EV = evidence_for(PROPS, SCENE)


def test_template_uses_evidence_and_disclaimer():
    text = template_explanation(EV)
    assert "13.6 dB" in text and "2026-07-01" in text and "-21.6 dB" in text
    assert DISCLAIMER_SENTENCE in text
    assert "not a probability" in text
    assert check_grounded(text, EV) == []


def test_grounding_rejects_invented_numbers_and_missing_disclaimer():
    bad = "About 5000 people were affected by this 1.5 km2 flood. " + DISCLAIMER_SENTENCE
    assert any("numbers not in the evidence" in p for p in check_grounded(bad, EV))
    assert any("disclaimer" in p for p in check_grounded("A 1.5 km2 patch got darker.", EV))


class FakeClient:
    def __init__(self, text, stop_reason="end_turn"):
        msg = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason,
                              model="fake-model")
        create = lambda **kwargs: msg  # noqa: E731
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def test_llm_rewrite_accepted_only_when_grounded():
    good = ("Radar shows this 1.484 km2 area got 13.6 dB darker between 2026-07-01 and 2026-07-13, "
            "which is typical of new open water. Confidence is High (0.70). " + DISCLAIMER_SENTENCE)
    text, info = llm_explanation(EV, client=FakeClient(good))
    assert info["source"] == "llm" and text == good

    hallucinated = "Flooding from 250 mm of rain damaged 40 homes. " + DISCLAIMER_SENTENCE
    text, info = llm_explanation(EV, client=FakeClient(hallucinated))
    assert info["source"] == "template" and text == template_explanation(EV)


def test_llm_refusal_falls_back_to_template():
    text, info = llm_explanation(EV, client=FakeClient("", stop_reason="refusal"))
    assert info["source"] == "template" and info["reason"] == "model declined"
