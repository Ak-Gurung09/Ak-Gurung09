from find_pairs import pair_candidates


def rec(t, group, i):
    return {"start": t, "group": group, "id": i, "url": f"https://example.invalid/{i}"}


def test_pairs_stay_within_one_geometry_and_bracket_the_event():
    recs = [rec("2026-07-02T01:00:00", ("T77", 10, "ASC"), "a1"),
            rec("2026-07-14T01:00:00", ("T77", 10, "ASC"), "a2"),
            rec("2026-07-26T01:00:00", ("T77", 10, "ASC"), "a3"),
            rec("2026-07-20T13:00:00", ("T12", 40, "DES"), "d1")]  # no "before" in this group
    pairs = pair_candidates(recs, "2026-07-20")
    assert len(pairs) == 1
    p = pairs[0]
    assert p["pre"]["id"] == "a2" and p["post"]["id"] == "a3" and p["days_apart"] == 12.0


def test_no_pairs_outside_window():
    recs = [rec("2026-01-01T00:00:00", "g", "x"), rec("2026-07-21T00:00:00", "g", "y")]
    assert pair_candidates(recs, "2026-07-20", max_days_before=36) == []


def test_records_without_time_are_ignored():
    recs = [rec("2026-07-14T01:00:00", "g", "a"), rec(None, "g", "b"), rec("2026-07-26T01:00:00", "g", "c")]
    assert [p["post"]["id"] for p in pair_candidates(recs, "2026-07-20")] == ["c"]
