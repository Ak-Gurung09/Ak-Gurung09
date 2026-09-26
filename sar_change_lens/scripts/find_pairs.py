"""Find pre/post acquisition pairs over an AOI, and optionally download them.

    # NISAR L2 GCOV PROVISIONAL (primary pathway), searched with asf_search:
    python scripts/find_pairs.py nisar-gcov --bbox 89.5 25.5 90.5 26.5 --event 2026-08-01

    # OPERA RTC-S1 or DSWx-S1 (fallback / reference), searched with earthaccess:
    python scripts/find_pairs.py opera-rtc  --bbox -122.5 48.2 -122.0 48.6 --event 2025-12-10
    python scripts/find_pairs.py opera-dswx --bbox -122.5 48.2 -122.0 48.6 --event 2025-12-10

Searching needs no login. Downloading (--download) needs a free NASA Earthdata
Login account: run `python -c "import earthaccess; earthaccess.login(persist=True)"`
once, or create ~/.netrc. Pairs are only formed within the same track/frame
and pass direction, because comparing different viewing geometries creates
false change.

NOTE: the network calls in this script could not be exercised in the
environment where it was written; the pairing logic is unit-tested with
mocked records (tests/test_find_pairs.py). Run it early on day one.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta

SHORT_NAMES = {  # from the official collection pages; verify in Earthdata Search on day one
    "nisar-gcov": "NISAR_L2_GCOV_PROVISIONAL_V1",
    "opera-rtc": "OPERA_L2_RTC-S1_V1",
    "opera-dswx": "OPERA_L3_DSWX-S1_V1",
}


def _time(rec: dict) -> datetime:
    return datetime.fromisoformat(str(rec["start"])[:19])


def pair_candidates(records: list[dict], event: str, max_days_before: int = 36,
                    max_days_after: int = 24) -> list[dict]:
    """Pick, per geometry group, the last acquisition before and the first on/after the event.

    records: dicts with keys start (ISO time), group (hashable geometry key:
    track/frame/direction for NISAR, burst or MGRS tile for OPERA), id, url.
    """
    t_event = datetime.fromisoformat(event)
    groups = defaultdict(list)
    for r in records:
        if r.get("start"):  # skip records without an acquisition time
            groups[r["group"]].append(r)
    pairs = []
    for group, recs in groups.items():
        recs.sort(key=_time)
        before = [r for r in recs if t_event - timedelta(days=max_days_before) <= _time(r) < t_event]
        after = [r for r in recs if t_event <= _time(r) <= t_event + timedelta(days=max_days_after)]
        if before and after:
            pre, post = before[-1], after[0]
            pairs.append({"group": group, "pre": pre, "post": post,
                          "days_apart": round((_time(post) - _time(pre)).total_seconds() / 86400, 1)})
    pairs.sort(key=lambda p: p["days_apart"])
    return pairs


def _search_nisar(bbox, start, end) -> list[dict]:
    import asf_search as asf
    wkt = (f"POLYGON(({bbox[0]} {bbox[1]},{bbox[2]} {bbox[1]},{bbox[2]} {bbox[3]},"
           f"{bbox[0]} {bbox[3]},{bbox[0]} {bbox[1]}))")
    results = asf.search(dataset="NISAR", processingLevel="GCOV", dataMaturity="PROVISIONAL",
                         intersectsWith=wkt, start=start, end=end)
    out = []
    for r in results:
        p = r.properties
        group = (p.get("pathNumber"), p.get("frameNumber"), p.get("flightDirection"))
        out.append({"start": p.get("startTime"), "group": group, "id": p.get("fileID") or p.get("sceneName"),
                    "url": p.get("url"), "product": r})
    return out


def _search_earthaccess(short_name, bbox, start, end) -> list[dict]:
    import earthaccess
    results = earthaccess.search_data(short_name=short_name, bounding_box=tuple(bbox), temporal=(start, end))
    out = []
    for g in results:
        umm = g["umm"]
        gid = umm.get("GranuleUR", "")
        # OPERA granule names embed the burst ID (RTC-S1) or MGRS tile (DSWx-S1)
        # as the 4th underscore-separated field, e.g. OPERA_L2_RTC-S1_T035-073251-IW2_...
        parts = gid.split("_")
        group = parts[3] if len(parts) > 3 else gid
        start_time = umm.get("TemporalExtent", {}).get("RangeDateTime", {}).get("BeginningDateTime")
        links = g.data_links()
        out.append({"start": start_time, "group": group, "id": gid, "url": links[0] if links else None,
                    "product": g})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("product", choices=sorted(SHORT_NAMES))
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    ap.add_argument("--event", required=True, help="event date YYYY-MM-DD")
    ap.add_argument("--days-before", type=int, default=36)
    ap.add_argument("--days-after", type=int, default=24)
    ap.add_argument("--download", type=int, metavar="N", help="download the N-th listed pair to data/raw/")
    args = ap.parse_args(argv)

    t = datetime.fromisoformat(args.event)
    start = (t - timedelta(days=args.days_before)).strftime("%Y-%m-%d")
    end = (t + timedelta(days=args.days_after)).strftime("%Y-%m-%d")
    try:
        if args.product == "nisar-gcov":
            recs = _search_nisar(args.bbox, start, end)
        else:
            recs = _search_earthaccess(SHORT_NAMES[args.product], args.bbox, start, end)
    except Exception as exc:  # network, auth, or API change: explain and stop
        print(f"Search failed ({exc.__class__.__name__}: {exc}). Use the offline fallback: "
              "pre-downloaded granules in data/raw/ (see docs, section 16).", file=sys.stderr)
        return 2

    pairs = pair_candidates(recs, args.event, args.days_before, args.days_after)
    print(f"{len(recs)} granules, {len(pairs)} same-geometry pre/post pairs for {SHORT_NAMES[args.product]}")
    for i, p in enumerate(pairs):
        print(f"[{i}] group={p['group']} {p['days_apart']} d apart\n    pre : {p['pre']['id']}\n"
              f"    post: {p['post']['id']}")
    if args.download is not None and pairs:
        import earthaccess
        earthaccess.login()
        chosen = pairs[args.download]
        urls = [chosen["pre"]["url"], chosen["post"]["url"]]
        earthaccess.download(urls, "data/raw")
        print("Downloaded to data/raw/. Record granule IDs and URLs in your case YAML.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
