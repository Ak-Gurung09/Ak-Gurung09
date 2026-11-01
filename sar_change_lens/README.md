# ChangeLens: NISAR radar surface-change screening

> **Screening and visualisation prototype for the NASA Space Apps Challenge 2026.
> Not an official flood map, emergency warning or damage assessment.**

ChangeLens compares a **before** and an **after** radar image of the same place from the
NASA-ISRO **NISAR** satellite (L2 GCOV, L-band). It flags **possible flood-related change**:
land that became open-water-like. Each flagged area comes with a confidence level, an
uncertainty panel, full data provenance and a plain-language explanation built only from
measured values.

- **Primary hazard:** flood / surface-water expansion.
- **Experimental:** candidate forest disturbance.
- **Not supported, on purpose:** landslides and ice. See the plan for why.

**Full design, data inventory, validation plan, 48-hour schedule and pitch:**
[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate      # Python 3.10-3.12
pip install -r requirements-dev.txt               # rasterio wheels include GDAL
pytest                                            # 51 tests, about 15 s
python -m changelens.pipeline --dry-run           # SYNTHETIC test data (placed over the ocean)
python -m changelens.pipeline --dry-run --no-change
streamlit run app/dashboard.py                    # opens the dashboard
```

## Using real NASA data

1. Create a free **NASA Earthdata Login** at <https://urs.earthdata.nasa.gov/>, then run
   `python -c "import earthaccess; earthaccess.login(persist=True)"`.
2. Find a before/after pair from the same track, frame and pass direction:
   ```bash
   python scripts/find_pairs.py nisar-gcov --bbox W S E N --event YYYY-MM-DD
   python scripts/find_pairs.py nisar-gcov --bbox W S E N --event YYYY-MM-DD --download 0
   ```
   The fallback for North America is `opera-rtc`. The reference layer is `opera-dswx`.

   **Nepal, 2025 (the team's chosen case).** NISAR has no calibrated 2025 data, so this case
   uses Sentinel-1, processed to backscatter by NASA ASF HyP3. See
   `config/cases/nepal_bara_rautahat_oct2025_s1.yaml` and section 4.3 of the plan.
   ```bash
   python scripts/find_pairs.py s1-grd --bbox 84.85 26.75 85.60 27.20 --event 2025-10-05 --days-after 12
   python scripts/find_pairs.py s1-grd --bbox 84.85 26.75 85.60 27.20 --event 2025-10-05 --days-after 12 --hyp3-rtc 0
   python scripts/find_pairs.py opera-dswx --bbox 84.85 26.75 85.60 27.20 --event 2025-10-05 --days-after 12
   ```
3. Copy `config/cases/TEMPLATE_nisar_gcov_flood.yaml`, fill in every `TODO` (paths, granule
   IDs, URLs), and set `status: ready`.
4. Run the case:
   ```bash
   python -m changelens.pipeline --case config/cases/<your_case>.yaml
   ```
5. Open the dashboard. If processing failed, the dashboard shows the error and **no map**.

## What is and isn't verified

**Verified by automated tests (synthetic data only):**

- georeferencing (CRS, transforms, windows, reprojection);
- no-data handling;
- detection of an injected flood (IoU > 0.9);
- no detections on an unchanged pair;
- HAND masking of a hillside false-positive trap;
- fail-safe behaviour and the dashboard rendering.

**Not yet verified:** a run on a real NISAR granule. Network access to NASA archives was not
available where this was written. The NISAR reader follows the documented HDF5 layout and is
tested against a mock file. Do the four day-one checks in the plan (section 9) on the first
real granule.

**Thresholds are documented starting values, not tuned values.** The confidence level is an
evidence score, not a probability.

## Data labels

Every dataset in `provenance.json` and on the dashboard carries one of these labels:

- *Primary NASA SAR data*
- *NASA-derived SAR product*
- *NASA-affiliated or partner data*
- *Supporting non-SAR data*
- *Optional future data*
- *SYNTHETIC test data (not an observation)*

## Layout

- `changelens/`: the pipeline (readers, alignment, speckle filter, detector, confidence,
  polygons, explanations, validation, rendering, provenance).
- `app/dashboard.py`: the Streamlit dashboard.
- `scripts/find_pairs.py`: data discovery and download.
- `config/cases/`: case templates.
- `tests/`: the test suite.
