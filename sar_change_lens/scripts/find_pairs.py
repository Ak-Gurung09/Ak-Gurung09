"""Find pre/post acquisition pairs over an AOI, and optionally download them.

    # NISAR L2 GCOV PROVISIONAL (primary pathway), searched with asf_search:
    python scripts/find_pairs.py nisar-gcov --bbox 89.5 25.5 90.5 26.5 --event 2026-08-01

    # OPERA RTC-S1 or DSWx-S1 (fallback / reference), searched with earthaccess:
    python scripts/find_pairs.py opera-rtc  --bbox -122.5 48.2 -122.0 48.6 --event 2025-12-10
    python scripts/find_pairs.py opera-dswx --bbox -122.5 48.2 -122.0 48.6 --event 2025-12-10

    # Sentinel-1 anywhere in the world (e.g. Nepal 2025, before NISAR's calibrated
    # record), turned into RTC backscatter by NASA ASF's HyP3 On Demand service:
    python scripts/find_pairs.py s1-grd --bbox 84.85 26.75 85.55 27.20 --event 2025-10-05 --days-after 12
    python scripts/find_pairs.py s1-grd --bbox 84.85 26.75 85.55 27.20 --event 2025-10-05 --days-after 12 --hyp3-rtc 0

Searching needs no login. Downloading (--download) and HyP3 processing
(--hyp3-rtc) need a free NASA Earthdata Login: run
`python -c "import earthaccess; earthaccess.login(persist=True)"` once, or
create ~/.netrc, or set EARTHDATA_TOKEN to an Earthdata Login user token.
Pairs are only formed within the same viewing geometry (track and pass
direction), because comparing different geometries creates false change.

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
    "s1-grd": "Sentinel-1 IW GRD_HD (ESA Copernicus, via ASF)",
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


def _search_s1(bbox, start, end) -> list[dict]:
    """Sentinel-1 IW GRD scenes. Grouped by relative orbit + pass direction: scenes on
    the same track share the viewing geometry even when their frame numbers differ."""
    import asf_search as asf
    wkt = (f"POLYGON(({bbox[0]} {bbox[1]},{bbox[2]} {bbox[1]},{bbox[2]} {bbox[3]},"
           f"{bbox[0]} {bbox[3]},{bbox[0]} {bbox[1]}))")
    results = asf.search(dataset=asf.DATASET.SENTINEL1, processingLevel=asf.PRODUCT_TYPE.GRD_HD,
                         beamMode=asf.BEAMMODE.IW, intersectsWith=wkt, start=start, end=end)
    out = []
    for r in results:
        p = r.properties
        group = (p.get("pathNumber"), p.get("flightDirection"))
        out.append({"start": p.get("startTime"), "group": group, "id": p.get("sceneName") or p.get("fileID"),
                    "url": p.get("url"), "product": r})
    return out


def submit_hyp3_rtc(pair: dict, out_dir: str = "data/raw") -> list[str]:
    """Process a Sentinel-1 GRD pair to RTC gamma-0 power GeoTIFFs with ASF HyP3, then download.

    Settings match what changelens expects: gamma0 radiometry, linear power
    scale, 30 m pixels, Copernicus DEM. Uses the free monthly HyP3 credits.
    """
    import os
    import zipfile

    import hyp3_sdk

    token = os.environ.get("EARTHDATA_TOKEN")
    hyp3 = hyp3_sdk.HyP3(token=token) if token else hyp3_sdk.HyP3()  # else ~/.netrc
    print(f"HyP3 credits remaining: {hyp3.check_credits()}")
    batch = hyp3_sdk.Batch()
    for role in ("pre", "post"):
        granule = pair[role]["id"]
        batch += hyp3.submit_rtc_job(granule, name=f"changelens_{role}", radiometry="gamma0",
                                     scale="power", resolution=30)
    print("Submitted; HyP3 jobs usually take minutes to an hour. Waiting...")
    batch = hyp3.watch(batch)
    extracted = []
    for zpath in batch.download_files(out_dir):
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(out_dir)
            extracted += [os.path.join(out_dir, n) for n in zf.namelist() if n.endswith(".tif")]
    return extracted


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
    ap.add_argument("--hyp3-rtc", type=int, metavar="N",
                    help="s1-grd only: RTC-process the N-th pair with ASF HyP3 and download it to data/raw/")
    args = ap.parse_args(argv)
    if args.product == "s1-grd" and args.download is not None:
        ap.error("raw Sentinel-1 GRD is not what the pipeline reads; use --hyp3-rtc N instead")
    if args.hyp3_rtc is not None and args.product != "s1-grd":
        ap.error("--hyp3-rtc only applies to s1-grd")

    t = datetime.fromisoformat(args.event)
    start = (t - timedelta(days=args.days_before)).strftime("%Y-%m-%d")
    end = (t + timedelta(days=args.days_after)).strftime("%Y-%m-%d")
    try:
        if args.product == "nisar-gcov":
            recs = _search_nisar(args.bbox, start, end)
        elif args.product == "s1-grd":
            recs = _search_s1(args.bbox, start, end)
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
    if args.hyp3_rtc is not None and pairs:
        tifs = submit_hyp3_rtc(pairs[args.hyp3_rtc])
        print("RTC GeoTIFFs:\n  " + "\n  ".join(sorted(tifs)))
        print("Put the *_VV.tif paths and the Sentinel-1 scene names into your case YAML.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
