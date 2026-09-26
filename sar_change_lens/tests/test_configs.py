"""The case templates must parse, carry valid data labels, and refuse to run until filled in."""

import glob
import os

import yaml

from changelens.pipeline import main
from changelens.provenance import DATA_LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def labels_in(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "label":
                yield v
            else:
                yield from labels_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from labels_in(v)


def test_templates_parse_and_use_known_labels():
    paths = glob.glob(os.path.join(ROOT, "config", "cases", "*.yaml"))
    assert paths
    for p in paths:
        with open(p) as fh:
            cfg = yaml.safe_load(fh)
        assert cfg["status"] == "template"
        assert set(labels_in(cfg)) <= set(DATA_LABELS), p


def test_template_refuses_to_run(tmp_path):
    p = os.path.join(ROOT, "config", "cases", "TEMPLATE_nisar_gcov_flood.yaml")
    assert main(["--case", p, "--out", str(tmp_path)]) == 2
