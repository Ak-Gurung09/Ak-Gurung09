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


def test_sentinel1_groups_by_track_and_direction_even_if_frames_differ():
    # Same relative orbit 19, descending; frame numbers differ between dates (normal for S1).
    recs = [rec("2025-09-29T00:10:00", (19, "DESCENDING"), "S1A_pre"),
            rec("2025-10-11T00:10:00", (19, "DESCENDING"), "S1A_post"),
            rec("2025-10-06T12:20:00", (85, "ASCENDING"), "S1C_post_only")]
    pairs = pair_candidates(recs, "2025-10-05", max_days_after=12)
    assert len(pairs) == 1 and pairs[0]["pre"]["id"] == "S1A_pre" and pairs[0]["post"]["id"] == "S1A_post"


def test_s1_raw_download_is_refused():
    import pytest
    from find_pairs import main
    with pytest.raises(SystemExit):
        main(["s1-grd", "--bbox", "84.85", "26.75", "85.55", "27.20", "--event", "2025-10-05", "--download", "0"])
