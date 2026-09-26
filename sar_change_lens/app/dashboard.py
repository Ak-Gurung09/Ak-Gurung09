"""ChangeLens dashboard: reads outputs/<case>/ written by the pipeline.

    streamlit run app/dashboard.py

The dashboard never runs detection itself and never shows a map for a case
whose pipeline run failed. Set CHANGELENS_OUTPUTS to point at another folder.
"""

from __future__ import annotations

import base64
import glob
import json
import os

import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

OUT_ROOT = os.environ.get("CHANGELENS_OUTPUTS",
                          os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"))

HAZARDS = {
    "Flood / surface-water expansion": {"key": "flood", "status": "MVP"},
    "Forest disturbance": {"key": "forest_disturbance", "status": "Experimental demo"},
    "Landslide": {"key": "landslide", "status": "Future extension"},
    "Ice / glacier change": {"key": "ice", "status": "Future extension"},
}
NOT_SUPPORTED = {
    "landslide": ("Landslides are usually small, sit on steep slopes where radar layover and shadow "
                  "distort the image, and are best detected with interferometric coherence or "
                  "displacement (e.g. NISAR GUNW, OPERA DISP-S1) plus a DEM and an inventory for "
                  "checking. A before/after backscatter difference alone would produce many false "
                  "alarms, so this prototype does not show landslide results."),
    "ice": ("Ice and glacier change needs offset tracking or interferometry to measure flow "
            "(e.g. NISAR pixel-offset products, NASA ITS_LIVE velocities). Backscatter over ice "
            "mostly tracks melt and snow wetness, not hazard, so this prototype does not "
            "show ice results."),
}
LAYERS = {
    "Change classes": "classes",
    "Confidence of flagged areas": "confidence",
    "Change intensity (after minus before, dB)": "delta_db",
    "Before image (radar backscatter)": "pre_db",
    "After image (radar backscatter)": "post_db",
}
INK_SECONDARY = "#52514e"


def load_cases(root: str) -> dict:
    cases = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "summary.json"))):
        try:
            with open(path) as fh:
                s = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        cases[s.get("case_id", os.path.basename(os.path.dirname(path)))] = {
            "summary": s, "dir": os.path.dirname(path)}
    return cases


def embed_html(html: str, height: int) -> None:
    """Embed our own generated map HTML (st.iframe on new Streamlit, components.html before)."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


def data_url(path: str) -> str:
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def build_map(case_dir: str, s: dict, layer: str, show_outlines: bool) -> folium.Map:
    (south, west), (north, east) = s["map_bounds_latlon"]
    # OpenStreetMap tiles need no key (light use, with attribution, per the OSM tile policy).
    m = folium.Map(location=[(south + north) / 2, (west + east) / 2], tiles="OpenStreetMap",
                   control_scale=True)
    png = os.path.join(case_dir, "previews", f"{LAYERS[layer]}.png")
    if os.path.exists(png):
        folium.raster_layers.ImageOverlay(image=data_url(png), bounds=s["map_bounds_latlon"],
                                          name=layer, opacity=0.9 if "image" in layer else 1.0).add_to(m)
    gj_path = os.path.join(case_dir, "detections.geojson")
    if show_outlines and os.path.exists(gj_path):
        with open(gj_path) as fh:
            gj = json.load(fh)
        if gj["features"]:
            folium.GeoJson(
                gj, name="Flagged areas",
                style_function=lambda f: {"color": "#0b0b0b", "weight": 2, "fillOpacity": 0},
                highlight_function=lambda f: {"weight": 4},
                tooltip=folium.GeoJsonTooltip(
                    fields=["id", "label", "confidence_label", "confidence", "area_km2", "mean_delta_db"],
                    aliases=["ID", "Label", "Confidence", "Score", "Area (km2)", "Mean change (dB)"]),
            ).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    m.fit_bounds([[south, west], [north, east]])
    return m


def offline_image(case_dir: str, layer: str) -> Image.Image:
    """Static preview that needs no internet: the layer drawn over the after-image.

    Used when the venue Wi-Fi fails (Leaflet and basemap tiles come from the web).
    """
    prev = os.path.join(case_dir, "previews")
    base = Image.open(os.path.join(prev, "post_db.png")).convert("RGBA")
    canvas = Image.new("RGBA", base.size, (252, 252, 251, 255))
    key = LAYERS[layer]
    if key in ("pre_db", "post_db"):
        canvas.alpha_composite(Image.open(os.path.join(prev, f"{key}.png")).convert("RGBA"))
    else:
        faded = base.copy()
        faded.putalpha(faded.getchannel("A").point(lambda a: a * 0.45))
        canvas.alpha_composite(faded)
        canvas.alpha_composite(Image.open(os.path.join(prev, f"{key}.png")).convert("RGBA"))
    scale = max(1, 640 // max(canvas.size))
    return canvas.resize((canvas.width * scale, canvas.height * scale), Image.NEAREST)


def legend_html(s: dict, layer: str) -> str:
    if LAYERS[layer] == "classes":
        items = [(v["color"], v["label"]) for v in s["legend"].values()]
        items.append(("transparent", "No significant change (not coloured)"))
    elif LAYERS[layer] == "confidence":
        items = [("#cde2fb", "Low (0 - 0.4)"), ("#3987e5", "Medium (0.4 - 0.7)"), ("#0d366b", "High (0.7 - 1)")]
    elif LAYERS[layer] == "delta_db":
        items = [("#2a78d6", "Darker after (e.g. new open water)"),
                 ("#e34948", "Brighter after (unclassified)"),
                 ("transparent", "Change smaller than 1 dB (not coloured)")]
    else:
        items = [("#1a1a1a", "Dark: low backscatter (smooth water, shadow)"),
                 ("#e6e6e6", "Bright: rough or structured surfaces")]
    chips = "".join(
        f'<span style="display:inline-flex;align-items:center;margin:0 14px 6px 0;color:{INK_SECONDARY}">'
        f'<span style="width:14px;height:14px;border-radius:3px;background:{c};margin-right:6px;'
        f'border:1px solid rgba(11,11,11,0.25)"></span>{t}</span>' for c, t in items)
    return f'<div style="font-size:0.9rem;line-height:1.4">{chips}</div>'


def main() -> None:
    st.set_page_config(page_title="ChangeLens SAR screening", layout="wide")
    st.title("ChangeLens: radar surface-change screening")
    st.warning("Screening and visualisation prototype built at a hackathon. It is **not** an official "
               "flood map, emergency warning or damage assessment. Always check official sources.")

    cases = load_cases(OUT_ROOT)
    if not cases:
        st.info(f"No results found in `{OUT_ROOT}`. Run `python -m changelens.pipeline --dry-run` "
                "or process a case first.")
        return

    with st.sidebar:
        st.header("Study area")
        demo_only = st.toggle("Demo mode (prepared cases only)", value=True,
                              help="Shows only cases that finished processing without errors.")
        names = [c for c, v in cases.items() if not demo_only or v["summary"].get("status") in ("ok", "degraded")]
        if not names:
            st.error("No successfully processed cases to show.")
            return
        case_id = st.selectbox("Case", names, format_func=lambda c: cases[c]["summary"].get("title", c))
        hazard_name = st.radio("Hazard view", list(HAZARDS),
                               format_func=lambda h: f"{h}  ({HAZARDS[h]['status']})")
        layer = st.radio("Map layer", list(LAYERS))
        show_outlines = st.checkbox("Outline flagged areas", value=True)
        map_mode = st.radio("Map display", ["Interactive map (needs internet)", "Offline image"],
                            help="Use the offline image if the venue Wi-Fi fails.")

    case = cases[case_id]
    s = case["summary"]
    if s.get("status") == "failed":
        st.error(f"Processing failed for this case, so no map is shown: {s.get('error')}")
        return
    if "synthetic" in s.get("mode", "").lower():
        st.error("SYNTHETIC DATA: this case tests the software only. It is not a satellite observation, "
                 "and it is deliberately placed over the ocean.")

    hz = HAZARDS[hazard_name]["key"]
    if hz in NOT_SUPPORTED:
        st.subheader(f"{hazard_name}: not available in this prototype")
        st.write(NOT_SUPPORTED[hz])
        return
    if hz != s["hazard"]:
        st.info(f"This case was processed for **{s['hazard_label']}**. Choose that hazard view, "
                "or pick a case processed for the hazard you selected.")
        return

    sc = s["scene"]
    top = st.columns(4)
    top[0].metric("Before", sc["pre_date"])
    top[1].metric("After", sc["post_date"])
    top[2].metric("Flagged areas", s["detections"], help=s["hazard_label"])
    top[3].metric("Flagged area (km2)", s["detected_area_km2"])
    st.caption(f"{sc.get('dataset')} | polarization {sc.get('polarization')} | "
               f"scene quality: **{s['quality']['overall']}**")

    left, right = st.columns([3, 2])
    with left:
        if map_mode.startswith("Offline"):
            (south, west), (north, east) = s["map_bounds_latlon"]
            st.image(offline_image(case["dir"], layer), width="stretch",
                     caption=f"Offline preview, north up, no basemap. Extent: lat {south:.4f} to {north:.4f}, "
                             f"lon {west:.4f} to {east:.4f}.")
        else:
            embed_html(build_map(case["dir"], s, layer, show_outlines).get_root().render(), height=560)
        st.markdown(legend_html(s, layer), unsafe_allow_html=True)

    with right:
        st.subheader("Why was this area flagged?")
        expl = s.get("explanations", {})
        if expl:
            pick = st.selectbox("Flagged area", list(expl))
            st.text(expl[pick]["text"])
            st.caption(f"Explanation source: {expl[pick].get('source', 'template')} "
                       "(built only from the measured values below)")
        else:
            st.write("Nothing was flagged in this case. For an unchanged pair that is the expected result.")

        with st.expander("Uncertainty and limitations", expanded=True):
            issues = s["quality"]["issues"] or ["no scene-level warnings"]
            st.markdown("**This scene:** " + "; ".join(issues))
            for note in s.get("confidence_notes", {}).get("notes", []):
                st.markdown(f"- {note}")
            st.markdown(
                "- Confidence is an evidence score (model separation, strength of darkening, terrain), "
                "**not** a probability of flooding.\n"
                "- Radar sees open water well but misses water under dense forest or between buildings.\n"
                "- Wind-roughened water can look like land; shadow, wet snow, flooded rice fields and "
                "smooth tarmac can look like water.\n"
                "- Two images give two snapshots: water that rose and fell between them is missed.")

        with st.expander("Data provenance"):
            st.dataframe(pd.DataFrame([{k: d.get(k) for k in ("role", "dataset", "granule_id", "label", "url")}
                                       for d in s["datasets"]]), hide_index=True)
            wm = s["water_model"]
            st.markdown(f"Water/land boundary learned from this scene: **{wm['threshold_db']} dB** "
                        f"(method: {wm['method']}, separation D = {wm['ashman_d']}). "
                        "Full parameters and file checksums: `provenance.json`.")

        if s.get("validation"):
            with st.expander("Agreement with reference layer"):
                v = s["validation"]
                st.markdown(f"Reference: {v['reference']}. {v['note']}")
                st.dataframe(pd.DataFrame([v["metrics"]]), hide_index=True)
                if v.get("by_confidence"):
                    st.caption("Does higher confidence mean more agreement? (share of flagged pixels "
                               "the reference also marks as new water)")
                    st.dataframe(pd.DataFrame(v["by_confidence"]), hide_index=True)

    st.subheader("Flagged areas (table view)")
    gj_path = os.path.join(case["dir"], "detections.geojson")
    with open(gj_path) as fh:
        feats = json.load(fh)["features"]
    if feats:
        cols = ["id", "label", "confidence_label", "confidence", "area_km2", "mean_delta_db",
                "mean_post_db", "centroid_lat", "centroid_lon"]
        st.dataframe(pd.DataFrame([{c: f["properties"].get(c) for c in cols} for f in feats]), hide_index=True)
    else:
        st.write("No flagged areas.")

    dl = st.columns(3)
    for col, (name, mime) in zip(dl, (("report.md", "text/markdown"), ("detections.geojson", "application/geo+json"),
                                      ("summary.json", "application/json"))):
        path = os.path.join(case["dir"], name)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                col.download_button(f"Download {name}", fh.read(), file_name=f"{case_id}_{name}", mime=mime)


main()
