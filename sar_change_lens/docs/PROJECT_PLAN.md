# ChangeLens: NISAR radar surface-change screening

**A 48-hour NASA Space Apps Challenge 2026 MVP plan, from data inventory to judge Q&A**

> **Verification date: 26 September 2026.** Facts about datasets, access and the 2026
> challenge were checked against official NASA / JPL / ASF / PO.DAAC / Space Apps pages
> through web search on that date. Several of those sites (spaceappschallenge.org,
> nisar-docs.asf.alaska.edu, earthdata.nasa.gov, cmr.earthdata.nasa.gov) could not be
> opened directly from the environment this plan was written in, because of its network
> policy. Where that happened, the wording comes from the search engine's extract of the
> official page. **Re-open every link in [section 19](#19-sources-and-assumptions) before
> the event.** Status labels in this document:
> **Verified** means an official page says so. **To confirm** means an official page
> suggests it but it must be checked on day one. **Assumption** means our judgement.

---

## Contents

1. [Executive summary](#1-executive-summary)
2. [Scientific feasibility assessment](#2-scientific-feasibility-assessment)
3. [Official NASA data inventory](#3-official-nasa-data-inventory)
4. [MVP recommendation](#4-mvp-recommendation)
5. [Hazard comparison table](#5-hazard-comparison-table)
6. [AI change-engine design](#6-ai-change-engine-design)
7. [System architecture](#7-system-architecture)
8. [Dashboard design](#8-dashboard-design)
9. [Data and processing pipeline](#9-data-and-processing-pipeline)
10. [Validation plan](#10-validation-plan)
11. [Project folder structure](#11-project-folder-structure)
12. [Implementation roadmap](#12-implementation-roadmap)
13. [48-hour schedule](#13-48-hour-schedule)
14. [Team roles](#14-team-roles)
15. [Judge-facing pitch and demo](#15-judge-facing-pitch-and-demo)
16. [Risks and fallback plans](#16-risks-and-fallback-plans)
17. [NASA-data authenticity checklist](#17-nasa-data-authenticity-checklist)
18. [Future extensions](#18-future-extensions)
19. [Sources and assumptions](#19-sources-and-assumptions)
20. [Closing statement: what this project can and cannot claim](#20-closing-statement-what-this-project-can-and-cannot-claim)

---

## 1. Executive summary

**What we build.** *ChangeLens* is an interactive web app. It compares two radar images of the
same place, taken before and after an event by the NASA-ISRO NISAR satellite, and flags
**"possible flood-related change"**: land that became open-water-like. Every flagged area
comes with:

- a confidence level (an evidence score, explained on screen);
- an uncertainty and limitations panel;
- a data-provenance table (which NASA granules, which processing steps, file checksums);
- a plain-language "Why was this area flagged?" explanation built only from measured values.

It is a **screening and visualisation tool, not an emergency warning system**, and it says
so on every screen and in every export.

**Why this scope.** The concept diagram asks one AI engine to cover floods, landslides,
forests and ice. Physically these are different measurements (see [section 2](#2-scientific-feasibility-assessment)).
A before/after backscatter comparison is well suited to **open-water floods**, is only
partially suited to **forest disturbance**, and is **not reliable** for landslides or ice
without interferometry, offset tracking, DEMs and inventories. So we narrow:

| Role in MVP | Hazard | Why |
|---|---|---|
| **Primary (working MVP)** | Flood / surface-water expansion | Strong, well-understood radar signal (calm water is dark). NASA provides an independent reference product (OPERA DSWx-S1). |
| **Secondary (limited, labelled experimental)** | Candidate forest disturbance | L-band cross-polarisation (HV) is sensitive to canopy loss, but moisture and season confound a single pair. One documented case at most. |
| **Future extensions only** | Landslide, ice/glacier change | Need coherence/InSAR, DEMs, offset tracking and inventories; unreliable as a 48-hour before/after product. |

**Challenge fit (to confirm on the official page).** Search extracts of the 2026 challenge list
show a NISAR challenge whose statement reads: *"build an interactive application that uses
radar remote sensing data from the NASA-ISRO Synthetic Aperture Radar (NISAR) mission to track
and visualize one or more types of surface change at locations around the world"*. ChangeLens
answers that directly, with one type of change done carefully. We could not open the challenge
page itself, so the team must confirm the title, requirements and suggested resources there
before freezing scope ([section 13](#13-48-hour-schedule), hour 0).

> **Update, 1 November 2026:** the team chose **Nepal with 2025 data**. NISAR has no
> calibrated data for 2025, so that case uses Sentinel-1 processed by NASA ASF HyP3. See
> [section 4.3](#43-team-decision-1-november-2026-nepal-2025-data) for the pathway and
> what it costs in challenge fit.

**Data pathway.**

- **Primary data.** NISAR **L2 GCOV, PROVISIONAL** (L-band, radiometrically terrain-corrected
  backscatter, 10 or 20 m, HDF5). Public through ASF DAAC since **20 July 2026**, for
  acquisitions from **17 June 2026**. **Verified**, with caveats in [section 3](#3-official-nasa-data-inventory).
- **Independent reference.** NASA JPL **OPERA DSWx-S1** (validated surface-water product from
  Sentinel-1, near-global, 30 m).
- **Fallback input, same code.** NASA JPL **OPERA RTC-S1** (Sentinel-1 backscatter, North America).

**AI, honestly scoped.**

1. The detector is physics-based, with an **unsupervised Gaussian mixture model**. It learns
   each scene's "water-like" and "land-like" backscatter populations, with no labels.
2. **Evidence-based confidence and uncertainty** come from model separation, change strength
   relative to the scene's own noise, and a terrain check.
3. A **grounded explanation layer** is template-based by default. An optional LLM rewrite is
   rejected automatically if it adds any number not in the evidence.

The LLM never detects anything.

**What already exists in this repository.**

- A working, tested starter pipeline and dashboard: 51 automated tests, a dry-run mode on
  clearly-labelled synthetic data, and fail-safe error handling.
- A NISAR GCOV reader, tested against a mock file that follows the documented layout.
- **Not yet done:** running on a real NISAR granule. That is the day-one task, because this
  build environment could not reach NASA archives.

**Buildable in 48 hours?** **Yes, with the scope frozen as above.** The critical risk is
finding a clean NISAR before/after pair, which is why there is a hard decision gate at hour 6
and a ready fallback.

---

## 2. Scientific feasibility assessment

### 2.1 What SAR measures

A synthetic aperture radar (SAR) is an **active** microwave instrument. It sends pulses
sideways toward the ground and records the echo's **amplitude** (how much energy came back)
and **phase** (where in its wave cycle the echo arrived). Moving along its orbit, it combines
many echoes into a sharp image. That is the "synthetic aperture": a long virtual antenna.

The normalised **backscatter coefficient** (σ⁰, or γ⁰ after terrain normalisation, usually
shown in **decibels, dB**) depends on four things:

- **Surface roughness relative to the wavelength.** Smooth surfaces mirror energy away.
- **Dielectric properties**, mostly water content. Wet soil and vegetation reflect more.
- **Geometry.** Slopes facing the radar are bright; slopes facing away are dark or in shadow.
- **Structure.** Canopies cause volume scattering. Trunks or buildings standing in water cause
  "double bounce".

NISAR's NASA-built radar is **L-band (wavelength ≈ 24 cm)**. ISRO's radar on the same
spacecraft is S-band (≈ 9–10 cm). Sentinel-1 (ESA) is **C-band (≈ 5.6 cm)**.

### 2.2 Why SAR works through clouds and at night

- **At night:** radar brings its own illumination, so it does not need sunlight.
- **Through clouds:** microwaves of centimetre-to-decimetre wavelength are far longer than
  cloud droplets, so clouds barely attenuate them.

This matters for floods: the storms that cause them usually hide the ground from optical
satellites. NASA's own NISAR visualisation of the January 2026 Mozambique floods makes exactly
this point: Sentinel-2 saw cloud while NISAR saw the water.

Caveat: very heavy rain can slightly affect shorter wavelengths. At L-band the effect is small.

### 2.3 Backscatter (amplitude) vs. InSAR (phase) vs. coherence

| Quantity | What it tells you | Product type | Used in MVP? |
|---|---|---|---|
| **Backscatter intensity change** (e.g. log-ratio of γ⁰ between dates) | The *kind* of surface changed: water appeared, canopy removed, soil got wetter | Calibrated / RTC backscatter (NISAR GCOV, OPERA RTC-S1) | **Yes, core** |
| **Interferometric phase (InSAR)** | Millimetre-to-centimetre movement of the ground along the radar line of sight | Interferograms, displacement time series (NISAR GUNW, ARIA-S1-GUNW, OPERA DISP-S1) | No (future: landslides) |
| **Coherence** (similarity of the complex signal between dates, 0–1) | Whether the surface was disturbed or changed randomly (damage, landslide, vegetation growth, snow) | Coherence layers in GUNW / GCOV-derived pairs, ARIA damage proxy maps | No (future) |
| **Pixel offsets** | Large displacements such as glacier flow (metres per day) | NISAR ROFF/GOFF, ITS_LIVE | No (future: ice) |

### 2.4 The four hazards are physically different

| Hazard | Physical radar signal | Simplest scientifically valid method | Also needs | Simple before/after enough? |
|---|---|---|---|---|
| **Open-water flood** | Calm water acts as a mirror (specular reflection), so backscatter drops sharply, often by 5–15 dB | Before/after backscatter change plus water-likeness in the "after" image | A terrain mask (HAND, height above nearest drainage), a permanent-water reference, an independent product for checking | **Yes, for open water in low-lying, sparsely built terrain** |
| **Flooded vegetation / urban flooding** | Double bounce makes backscatter *increase*, especially in L-band HH | Polarimetric / ratio analysis, land-cover context | Land cover, expert interpretation | No: experimental only (future) |
| **Landslide** | Small (tens of metres), steep terrain; backscatter change is subtle and geometric distortion is strong; coherence drops | Coherence loss or InSAR displacement plus slope masks | DEM, layover/shadow masks, optical imagery, inventories (NASA COOLR) | **No** |
| **Forest disturbance** | Canopy loss removes volume scattering, so cross-pol (HV/VH) backscatter decreases | Time-series anomaly vs. a multi-date baseline (as OPERA DIST-S1 does) | Several pre-event dates, moisture/season context, fire data | **Partially:** a single pair gives *candidates* only |
| **Ice / glacier / snow** | Flow needs offset tracking or InSAR. Wet snow lowers backscatter strongly. Calving fronts need boundary mapping | Offset tracking (ITS_LIVE, NISAR offsets), wet-snow ratio methods, boundary digitising | Specialised processing, seasonal context | **No:** backscatter change mostly tracks melt, not hazard |

### 2.5 Common sources of false positives and false negatives

| Cause | What it looks like | Why | Mitigation in this MVP |
|---|---|---|---|
| **Terrain shadow** | Dark patch ("water") on mountainsides | The radar cannot illuminate slopes facing away | HAND mask (> 15 m above drainage is masked). Product layover/shadow masks where available |
| **Layover** | Very bright, smeared slopes | Slopes facing the radar compress into few pixels | Masked in the product. Excluded as no-data |
| **Speckle** | Salt-and-pepper "changes" | Coherent-imaging interference noise | Lee filter, 2 ha minimum mapping unit |
| **Agriculture** | New "water" in fields; brightness swings | Irrigation, rice-paddy flooding, ploughing, harvest | Stated limitation. Low-confidence display. Choose AOIs carefully. Show one known agricultural false positive in the demo |
| **Vegetation moisture / rain** | Brightening or darkening without flooding | Wetter leaves and soil change the dielectric constant | Require *water-like* backscatter, not just change |
| **Rough water** | Flood *missed*: water looks like land | Wind waves scatter energy back to the radar (L-band is less sensitive than C-band, but not immune) | Stated false-negative risk |
| **Wet snow** | Large "new water" areas in mountains or cold regions | Liquid water in snow absorbs microwaves, so it is very dark | HAND mask, seasonal caution, avoid snow AOIs |
| **Dry sand, tarmac, smooth surfaces** | Permanent "water" | Smooth surfaces are dark | Stable in both dates, so not *new* water |
| **Seasonal change** | Widespread moderate change | Phenology, freeze/thaw, soil moisture | Short date gaps (≤ 24 days preferred). Scene-quality warning if longer |
| **Registration / geometry errors** | Thin false-change lines along edges | Misaligned grids, or different orbits (ascending vs. descending) | Same track, frame and direction only. Explicit reprojection to one grid. Edge pixels mostly fall below the minimum mapping unit |
| **Radio-frequency interference (L-band)** | Streaks or bright stripes | Ground radars and communications transmitting at L-band | Visual QA of the "before" and "after" layers. Documented as a NISAR limitation |
| **Water under canopy / between buildings** | Flood *missed* or appears as brightening | Double bounce instead of specular reflection | Stated false-negative risk. "Other increase" class is shown but unclassified |

### 2.6 Verdict

- **Can be done credibly with before/after change detection in 48 hours:** open-water flood
  extent in flat, low-lying, sparsely built terrain.
- **Can be demonstrated as experimental candidates:** stand-replacing forest disturbance (a burn
  scar or clear-cut) from an L-band HV decrease, clearly labelled.
- **Need coherence, interferometry, DEMs, optical imagery, weather data or field validation:**
  landslides, ground deformation, urban and vegetated flooding, glacier flow, damage.

---

## 3. Official NASA data inventory

### 3.1 Source table

Labels used throughout this project, in the docs, the dashboard and `provenance.json`:

- **Primary NASA SAR data**
- **NASA-derived SAR product**
- **NASA-affiliated or partner data**
- **Supporting non-SAR data**
- **Optional future data**

| Source | Organization | Product / data type | SAR or derived product | Hazard relevance | Access method | 48-hour practicality | Limitations | Official link | Label |
|---|---|---|---|---|---|---|---|---|---|
| **NISAR L2 GCOV, PROVISIONAL** | NASA JPL and ISRO. The L-band SAR is NASA's; distributed by NASA ASF DAAC | Geocoded polarimetric covariance: RTC γ⁰ power for HH, HV (quad-pol in some areas). 10 or 20 m (frequency A), 80 m (frequency B). HDF5 | Calibrated, radiometrically terrain-corrected SAR (Level 2) | **Flood (core)**, inundated vegetation, forest disturbance, ice backscatter | ASF Vertex. `asf_search` (`dataset='NISAR', processingLevel='GCOV', dataMaturity='PROVISIONAL'`). `earthaccess` short name `NISAR_L2_GCOV_PROVISIONAL_V1`. **Earthdata Login to download** | **Medium–high.** Analysis-ready, but large custom HDF5: read the AOI window only. Download two granules by hour 6 | PROVISIONAL (validated at a limited set of sites). Only acquisitions **≥ 2026-06-17**. 12-day repeat. Known radiometric banding in some 40 MHz joint-mode products. L-band RFI | [NISAR GCOV guide](https://nisar-docs.asf.alaska.edu/gcov/); [catalog](https://www.earthdata.nasa.gov/data/catalog/asf-nisar-l2-gcov-provisional-v1-1) | Primary NASA SAR data |
| **NISAR L2 GUNW, PROVISIONAL** | NASA JPL / ISRO via ASF | Geocoded unwrapped interferogram plus coherence | InSAR (phase) product | Landslide / deformation (future) | As above | Low for this MVP | Needs InSAR expertise. Atmospheric and ionospheric effects, unwrapping errors | [Catalog](https://www.earthdata.nasa.gov/fr/data/catalog/asf-nisar-l2-gunw-provisional-v1-1) | Primary NASA SAR data (future use) |
| **NISAR BETA collections** (Jan/Feb 2026 releases) | NASA / ASF | Pre-calibration L1–L3 products | SAR | Familiarisation only | ASF | Low | Documented as *"not intended for use in scientific research"* | [GCOV beta catalog](https://www.earthdata.nasa.gov/data/catalog/asf-nisar-l2-gcov-beta-v1-1) | Optional future data (do **not** use for results) |
| **NISAR pixel offsets** (ROFF/GOFF) | NASA JPL / ISRO via ASF | Offset-tracking displacement | SAR-derived | Glacier flow (future) | ASF | Low | Provisional availability **to confirm** | [Products overview](https://nisar-docs.asf.alaska.edu/products-overview/) | Optional future data |
| **OPERA RTC-S1 V1** | NASA JPL OPERA, from ESA Copernicus Sentinel-1 | RTC γ⁰ backscatter, 30 m, one COG per burst per polarization (VV/VH) | NASA-derived SAR product | **Flood (fallback input)**, disturbance | ASF DAAC. `earthaccess` `OPERA_L2_RTC-S1_V1`. Earthdata Login | **High** (small COGs, simple format) | **North America only**: USA and territories, Canada within 200 km of the border, Mexico to Panama. C-band. Burst-sized tiles | [Catalog](https://www.earthdata.nasa.gov/data/catalog/asf-opera-l2-rtc-s1-v1-1); [product guide](https://hyp3-docs.asf.alaska.edu/guides/opera_rtc_product_guide/) | NASA-derived SAR product |
| **OPERA DSWx-S1 V1** | NASA JPL OPERA; distributed by PO.DAAC | Surface-water classes: 0 not water, 1 open water, 3 inundated vegetation, 250 HAND-masked, 251 layover/shadow, 255 fill. 30 m MGRS tiles, GeoTIFF | NASA-derived SAR product (surface-water extent) | **Flood reference / benchmark** | PO.DAAC. `earthaccess` `OPERA_L3_DSWX-S1_V1`. Earthdata Login | **High** | Released 2024-10-25; forward production since Sept 2024. Near-global. 6–12-day revisit. Maps water bodies > ~3 ha and > 200 m wide. **A model output, not ground truth** | [PO.DAAC dataset](https://podaac.jpl.nasa.gov/dataset/OPERA_L3_DSWX-S1_V1); [product spec](https://d2pn8kiwq2w21t.cloudfront.net/documents/OPERA_DSWx-S1_Product_Spec_v1.0.0_D-108761_RevA_2024-08-16_.pdf) | NASA-derived SAR product |
| **OPERA DSWx-HLS V1** | NASA JPL OPERA (Landsat 8/9 + Sentinel-2) | Optical surface-water extent | Not SAR | Cloud-free visual cross-check | PO.DAAC | Medium (clouds during floods) | Cloud-limited | [PO.DAAC](https://podaac.jpl.nasa.gov/dataset/OPERA_L3_DSWX-HLS_V1) | Supporting non-SAR data |
| **OPERA DSWx-NI** | NASA JPL OPERA | NISAR-based water extent (planned 30 m MGRS, 12-day) | NASA-derived SAR product | Flood (ideal NISAR-native reference) | PO.DAAC, *when released* | **Unknown.** A 2025 OPERA workshop slide projected October 2026 | **Not confirmed released** as of 2026-09-26 | [DSWx suite](https://www.jpl.nasa.gov/go/opera/products/dswx-product-suite/) | Optional future data |
| **OPERA DIST-ALERT-S1 V1** | NASA JPL OPERA | Disturbance alerts from Sentinel-1 backscatter time series. COG, 10 layers incl. status and confidence | NASA-derived SAR product | Forest disturbance reference (secondary) | ASF. Earthdata Login | Medium: **to confirm** granules exist for your AOI and date | The catalog entry lists 2026 Sentinel data; NASA pages said production was scheduled for 2026 | [Catalog](https://www.earthdata.nasa.gov/data/catalog/asf-opera-l3-dist-alert-s1-v1-1) | NASA-derived SAR product (availability to confirm) |
| **OPERA DIST-ALERT-HLS V1** | NASA JPL OPERA | Optical disturbance alerts | Not SAR | Forest disturbance cross-check | LP DAAC | Medium | Cloud-limited | [Catalog](https://www.earthdata.nasa.gov/data/catalog/lpcloud-opera-l3-dist-alert-hls-v1-1) | Supporting non-SAR data |
| **OPERA DISP-S1 V1** | NASA JPL OPERA | InSAR displacement time series, 30 m, from 2016 | NASA-derived SAR product | Slow-moving landslides, subsidence (future) | ASF (incl. ASF displacement portal) | Low for MVP | North America. Needs time-series interpretation | [Catalog](https://www.earthdata.nasa.gov/data/catalog/asf-opera-l3-disp-s1-v1-1) | NASA-derived SAR product (future) |
| **ARIA-S1-GUNW** | NASA JPL ARIA, hosted by ASF | InSAR interferograms (NetCDF, 90 m) plus coherence and corrections | NASA-derived SAR product | Deformation, landslides (future) | ASF Vertex, on-demand | Low for MVP | Mostly tectonically active areas | [Product guide](https://hyp3-docs.asf.alaska.edu/guides/gunw_product_guide/) | NASA-derived SAR product (future) |
| **ARIA Flood Proxy Maps / Damage Proxy Maps** | NASA JPL ARIA with the NASA Disasters Program | Event maps from Sentinel-1 amplitude / coherence change | NASA-derived SAR product (flood / damage *proxy*) | Reference layer for past activated events | NASA Disasters mapping portal | Medium (only for activated events) | ARIA's own guidance: flood proxies are *less reliable over urban areas*; damage proxies are less reliable over vegetation and cannot separate flood from damage | [Example](https://disasters.nasa.gov/our-impact/news/aria-flood-extent-map-harvey-sentinel-1-sar-data) | NASA-derived SAR product |
| **NASA Disasters Program activations** | NASA Earth Science Applied Sciences | Event pages and GIS layers (e.g. OPERA DSWx-S1 for Alaska flooding Oct 2025, Washington atmospheric rivers Dec 2025; DIST-S1 for Texas flooding Jul 2025) | Access point to NASA-derived products | **Case-study selection** and reference | Web and ArcGIS REST services | High for browsing | Only activated events | [disasters.nasa.gov](https://disasters.nasa.gov/) | NASA-derived SAR product (per layer) |
| **ASF HyP3 On Demand** | NASA ASF DAAC | On-demand RTC and InSAR processing of Sentinel-1, worldwide | NASA-derived SAR product (from partner data) | Fallback RTC outside North America | Vertex or `hyp3_sdk`. Earthdata Login. Free basic allotment of 8,000 credits/month | Medium: jobs queue, so submit early | C-band. Credit limits | [HyP3 docs](https://hyp3-docs.asf.alaska.edu/) | NASA-derived SAR product |
| **Sentinel-1 A/C/D** | ESA / European Commission (Copernicus); mirrored by ASF | C-band SLC/GRD Level-1 SAR | Raw / Level-1 SAR | Inputs to OPERA, ARIA and HyP3 | ASF, Copernicus Data Space | Medium | Not NASA data. S-1C launched 2024-12-05. S-1D launched 2025-11-04 (calibrated data from 2026-04-17). S-1A operations planned to end 2026-06-29 | [ASF Sentinel-1](https://hyp3-docs.asf.alaska.edu/sentinel1/) | NASA-affiliated or partner data |
| **UAVSAR** | NASA JPL (airborne L-band) | Campaign SAR imagery; simulated NISAR products | Primary SAR (airborne) | Learning L-band appearance | UAVSAR site / ASF | Low | Campaign coverage only | [Simulated NISAR](https://uavsar.jpl.nasa.gov/science/documents/nisar-sample-products.html) | Optional future data |
| **ASF GLO-30 HAND** | ASF, derived from the Copernicus GLO-30 DEM | Height above nearest drainage, 30 m COG, CC0 licence | Not SAR | **Flood plausibility mask** | AWS Open Data (no login) | **High** | DEM-derived (2021 DEM release). Weaker in flat deltas and cities | [Registry](https://registry.opendata.aws/glo-30-hand/) | Supporting non-SAR data |
| **Copernicus GLO-30 DEM** | ESA / Airbus | 30 m elevation | Not SAR product (radar-derived DEM) | Slope, terrain context | AWS Open Data | High | Partner data | [Registry](https://registry.opendata.aws/copernicus-dem/) | Supporting non-SAR data |
| **NASA FIRMS** | NASA LANCE | MODIS/VIIRS active-fire detections | Not SAR | Confirms a fire happened (forest demo) | Web map; API needs a free MAP_KEY | High | Point detections (375 m–1 km), not burn perimeters | [FIRMS](https://firms.modaps.eosdis.nasa.gov/) | Supporting non-SAR data |
| **NASA Global Landslide Catalog / COOLR** | NASA GSFC | Landslide reports and inventories | Not SAR | Landslide validation (future) | Landslide Viewer | Medium | Report-based. Location precision varies | [landslides.nasa.gov/viewer](https://gpm.nasa.gov/landslides/data.html) | Optional future data |
| **ITS_LIVE (NASA MEaSUREs)** | NASA JPL | Glacier and ice-sheet velocities from optical + Sentinel-1 offset tracking | NASA-derived product (partly SAR) | Ice (future) | its-live.jpl.nasa.gov, NSIDC | Medium | Velocity, not hazard | [ITS_LIVE](https://its-live.jpl.nasa.gov/) | Optional future data |

### 3.2 Answers to the seven verification questions

1. **Publicly accessible?** Yes, for everything in the table.
   - NISAR L-band products are distributed free through ASF DAAC and NASA Earthdata.
   - OPERA products are free.
   - HAND is on AWS Open Data.
   - Sentinel-1 is free and open (Copernicus).
2. **Account or credentials?**
   - **Downloading** from ASF DAAC or PO.DAAC needs a **free NASA Earthdata Login**.
     Searching and browsing do not.
   - HyP3 needs Earthdata Login.
   - HAND on AWS needs nothing.
   - The FIRMS API needs a free MAP_KEY.
   - The optional LLM explanation needs an Anthropic API key. It is off by default.
   - **Every team member should create an Earthdata Login at hour 0.**
3. **Can suitable data be found and downloaded within 48 hours?** Yes, if pairs are found by
   hour 6.
   - NISAR granules are large, so either download two granules early or read only the AOI
     window. The NISAR guide documents streaming with `asf_search` + `fsspec`.
   - OPERA COGs are small.
   - If local event rules allow preparation before the hackathon, scout candidate pairs in
     advance. If not, the schedule assumes scouting during hours 2–6.
4. **Is NISAR sufficiently available, processed and documented?** Yes, for this MVP, with caveats.
   - Available and processed: GCOV PROVISIONAL is calibrated, terrain-corrected and geocoded.
   - Documented: the NISAR Data User Guide, ARSET 2026 NISAR training, and an Earthdata
     tutorial all exist.
   - Caveat: it is *provisional*.
   - Caveat: the calibrated record starts **17 June 2026**, so by November there is only
     about five months of data. There is no multi-year baseline, and the event must fall in
     that window.
   - Caveat: files are a custom HDF5 layout.
   - Caveat: DSWx-NI (NISAR water product) is not confirmed released.
5. **Are Sentinel-1, OPERA or ARIA products more practical?**
   - For **historical** events and **North America**, OPERA RTC-S1 + DSWx-S1 are easier
     (COGs, validated, records back to 2016 for RTC). That is why they are our fallback and
     reference.
   - For the **NISAR challenge**, NISAR must be the primary input.
6. **Limitations of the three core datasets:**

   | | NISAR L2 GCOV (PROVISIONAL) | OPERA RTC-S1 | OPERA DSWx-S1 |
   |---|---|---|---|
   | Band / wavelength | L-band (~24 cm) | C-band (~5.6 cm) | C-band (derived from RTC-S1) |
   | Pixel spacing | 10 or 20 m (freq. A) | 30 m | 30 m |
   | Polarization | Mostly dual-pol (HH, HV) over land; quad-pol in some areas | Usually VV + VH | Classification layer |
   | Revisit | 12-day repeat cycle, ascending + descending passes | 6–12 days, depending on region and constellation | 6–12 days |
   | Coverage | Near-global land and ice | North America | Near-global |
   | Record | Calibrated from 2026-06-17 | 2016 onward | Forward production since Sept 2024 |
   | Format | Custom HDF5 | COG per burst | GeoTIFF per MGRS tile |
   | Maturity | Provisional | Validated | Validated |

7. **What kind of data is each?** See the "SAR or derived product" column above. In short:
   - GCOV and RTC-S1 are **calibrated, terrain-corrected backscatter**.
   - GUNW, DISP-S1 and ARIA-S1-GUNW are **InSAR / displacement**.
   - DSWx-S1 is a **surface-water extent** classification.
   - ARIA FPM/DPM are **flood / damage proxies**.
   - DIST-ALERT is a **disturbance** product.
   - Sentinel-1 SLC/GRD is **Level-1 SAR from a partner**.

---

## 4. MVP recommendation

| Element | Decision |
|---|---|
| **One primary hazard** | Flood / surface-water expansion ("possible flood-related change") |
| **One secondary hazard** | Candidate forest disturbance (experimental, one case, fixed low confidence) |
| **Future only** | Landslide, ice/glacier (the dashboard explains *why* instead of showing a map) |
| **One data pathway** | NISAR L2 GCOV PROVISIONAL (HH) before/after pair from the same track, frame and direction. Fallback: OPERA RTC-S1 (VV) in North America, same code |
| **One core change-detection method** | Log-ratio change plus water-likeness from a per-scene Gaussian mixture model, with HAND mask and minimum mapping unit |
| **One AI component that adds real value** | Unsupervised, per-scene probabilistic water model that yields calibrated-looking *and explainable* probabilities without labels, plus grounded explanations |
| **One dashboard** | Streamlit + Leaflet (folium), with an offline image mode |
| **One strong demo scenario** | "A flood happened under cloud. Here is what NISAR saw, here is why each area was flagged, and here is how much to trust it." Also show one **negative (unchanged)** pair and one **known false positive** |

### 4.1 Choosing the case studies (hours 2–6)

Pick in this order, and record everything in `config/cases/<case>.yaml`:

1. **NISAR primary case.** A documented flood, or a strong seasonal inundation, that happened
   **after 17 June 2026**, with:
   - two PROVISIONAL GCOV acquisitions from the **same track, frame and pass direction**, one
     before the water rose and one near the peak (≤ 24 days apart ideally);
   - flat, low-lying terrain with a river to anchor the water class;
   - an OPERA DSWx-S1 tile within a few days of each NISAR date.

   *Candidate regions (assumption, to be checked with `scripts/find_pairs.py`):*
   - 2026 monsoon-season floodplains in South Asia (e.g. the Brahmaputra in Assam / Bangladesh),
     which NISAR's India-focused observation plan should cover well. Beware rice paddies (a
     known false-positive source).
   - Any 2026 NASA Disasters Program flood activation after mid-June.
2. **Fallback case (North America, validated inputs).** A NASA Disasters flood activation
   covered by OPERA RTC-S1 and DSWx-S1, e.g. the **December 2025 Washington atmospheric-river
   floods** or the **October 2025 western Alaska flooding** (both have Disasters Program OPERA
   DSWx-S1 layers).
3. **Negative case.** The same NISAR track over a period or area with no expected change.
   Expected result: nothing flagged.
4. **Experimental forest case (only if time).** A documented 2026 wildfire with NISAR HV
   before/after, confirmed with NASA FIRMS detections.

### 4.2 Presenting preprocessed data honestly

Preprocessing is normal in Earth observation. Hiding it is not. In the demo:

- **Show the real inputs.** Put the granule IDs, ASF URLs, acquisition times and
  `identification` metadata on screen (Data provenance panel), plus `provenance.json` with
  SHA-256 checksums of every output.
- **Say it plainly:** "We downloaded these two NISAR granules from ASF on [date] and processed
  them with this pipeline. The dashboard reads the results so the demo does not depend on
  venue Wi-Fi."
- **Re-run live.** Run `python -m changelens.pipeline --case config/cases/<case>.yaml` on the
  cached AOI subset. It takes seconds for a small AOI. Or change one parameter (e.g.
  `delta_db`) and show the map update.
- **Never** demo on synthetic data without the red **SYNTHETIC DATA** banner. The dashboard
  enforces the banner, and the dry-run scene sits over open ocean on purpose.

### 4.3 Team decision (1 November 2026): Nepal, 2025 data

The team chose **Nepal** as the study area and **2025** as the data year. That choice
changes the data pathway.

**Event.** Flooding of the **Terai plains in Bara and Rautahat districts** (Madhesh Province)
after extreme rain on **3–5 October 2025**. More than 330 mm of rain was reported in
Rautahat, and the same storm caused deadly landslides in Ilam and Koshi River flooding
further east. The case file is
[`config/cases/nepal_bara_rautahat_oct2025_s1.yaml`](../config/cases/nepal_bara_rautahat_oct2025_s1.yaml).

**Why the Terai and not the hills.** Radar flood mapping needs flat, low-lying ground. In
Nepal's hills and mountains, layover, shadow and narrow valleys defeat a before/after
method. That rules out:

- the Ilam landslides (October 2025);
- the Rasuwa glacial flood (8 July 2025).

Alternative Terai events in 2025 (to scout with `find_pairs.py`):

- **Kailali**, far west. International Charter activation 970 on 1 July 2025.
- **Jhapa**, east. Flooding on 4–5 October 2025.

**Data pathway for 2025.**

| Role | Data | Label |
|---|---|---|
| Before/after radar | Copernicus Sentinel-1 IW GRD (ESA), turned into RTC γ⁰ power at 30 m by **NASA ASF HyP3 On Demand** (`find_pairs.py s1-grd … --hyp3-rtc N`) | NASA-affiliated or partner data (ESA observation, NASA processing) |
| Reference | NASA JPL **OPERA DSWx-S1**. Near-global by design; **confirm Nepal tiles exist** for the dates | NASA-derived SAR product |
| Terrain mask | ASF GLO-30 HAND | Supporting non-SAR data |
| Optional rain context | NASA **GPM IMERG** rainfall for 3–5 October 2025 | Supporting non-SAR data |

**Why not NISAR.**

- NISAR's calibrated (PROVISIONAL) record starts **17 June 2026**.
- Its pre-calibration BETA data covers a subset of **17 October 2025 – 20 January 2026**,
  which is *after* this flood, and NASA marks it "not intended for use in scientific research".
- OPERA RTC-S1 covers North America only, so it cannot serve Nepal either.

**What this costs, said plainly.**

1. **Challenge fit.** The 2026 Space Apps challenge this plan targets asks for **NISAR**
   data. A Nepal-2025-only project uses **no NISAR data**. To keep the fit, add a second
   Nepal Terai case from the **2026 monsoon (July–September 2026)** using NISAR GCOV
   PROVISIONAL. The pipeline already supports it (`find_pairs.py nisar-gcov`), and the two
   cases together show the same method on C-band (2025) and L-band (2026).
2. **NASA-data authenticity.** The detector's input becomes ESA observations processed by
   NASA. NASA's role is the processing (ASF HyP3), the reference (OPERA DSWx-S1), the
   terrain layer (ASF) and optional rainfall (GPM). Say this exactly; do not call
   Sentinel-1 "NASA data".
3. **Validation strength.** DSWx-S1 is also made from Sentinel-1, so agreement with it is a
   *consistency check*, not independent validation. A Nepali or international flood map for
   these dates (Charter, UNOSAT, ICIMOD) would be a more independent comparison, if one exists.
4. **Timing risk.** Terai floodwater can drain within days. Sentinel-1 revisits each track
   every 6–12 days, so the first post-event pass may miss the peak. Check the dates
   `find_pairs.py` returns before committing to the event.
5. **Look-alikes.** In early October the Terai's rice paddies may still be wet, so expect
   agricultural false positives. Show one in the demo rather than hide it.

---

## 5. Hazard comparison table

Scores are 1–10. For **Processing difficulty** and **Risk of misleading users**, *higher is
worse*. They are inverted (11 − score) in the total.

| Criterion | A: Flood | B: Landslide | C: Forest disturbance | D: Ice / glacier |
|---|---|---|---|---|
| Data availability | **9** | 5 | 7 | 6 |
| Scientific validity within 48 h | **8** | 3 | 5 | 4 |
| Processing difficulty (higher = harder) | **4** | 8 | 5 | 7 |
| AI usefulness | **7** | 5 | 6 | 4 |
| Visualization potential | **9** | 7 | 7 | 8 |
| NASA-data authenticity | **9** | 8 | 9 | 8 |
| Judge appeal | 8 | **8** | 7 | 7 |
| Risk of misleading users (higher = worse) | **4** | 8 | 6 | 7 |
| MVP reliability | **8** | 3 | 6 | 4 |
| **Total (difficulty & risk inverted, max 90)** | **72** | **45** | **58** | **49** |

The scores rest on the following reasoning.

**A. Flood.**

- Calibrated NISAR data and a validated NASA reference (DSWx-S1) exist.
- The physics is simple enough to explain to judges in one sentence.
- An unchanged pair gives a clean negative test.
- Main risks: look-alikes (shadow, wet snow, paddies) and missed water under canopy. All are
  explainable and partly mitigated.

**B. Landslide.**

- Dramatic, and InSAR products exist, so judge appeal is high.
- But landslides are small and sit in the worst radar geometry (steep slopes).
- A before/after amplitude difference would flag many false positives.
- Validation needs inventories that rarely exist for a fresh event.
- High risk of misleading users.

**C. Forest disturbance.**

- NISAR's L-band HV is genuinely good at this.
- But a single pair confounds disturbance with moisture and season.
- NISAR's calibrated record is too short for a proper baseline.
- It is credible as a labelled experimental secondary.

**D. Ice.**

- Visually striking, but the real hazard signals (flow speed, calving) need offset tracking.
- Backscatter change on ice mostly reflects melt.
- Users could easily read "change" as "ice loss".

**Recommendation.** A is the primary MVP and C is the experimental secondary. B and D are
future extensions, shown in the dashboard as "not available in this prototype, and here is
why", which judges tend to respect.

---

## 6. AI change-engine design

### 6.1 Approaches compared

| # | Approach | Required data | Training | Scientific risks | Interpretability | Compute | 48-h feasibility | Suitability for flood | Likely judge perception |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Threshold-based before/after** (fixed dB thresholds) | Calibrated pre/post pair | None | Fixed thresholds do not transfer between sensors, scenes or seasons; speckle; look-alikes | Very high | Trivial | Yes | Good baseline | "Simple": acceptable if tied to physics and validated |
| 2 | **Statistical anomaly detection** (change z-score vs. the scene's own noise; CFAR; time-series baseline) | Pair (or time series) | None | Needs a sound noise model. NISAR's record is too short for long baselines | High | Low | Yes | Good for gating "significant" change | Solid, scientific |
| 3 | **Pixel/patch classical ML** (random forest, gradient boosting) | Labelled pixels | Labels needed | **Label circularity** (training on DSWx-S1, which is derived from the same data, just copies it). Overfits one event. Spatial autocorrelation inflates scores | Medium (feature importance) | Low | Only if labels exist | Useful later | "Real ML", but judges will ask where the labels came from |
| 4 | **Unsupervised clustering** (Gaussian mixture, Kittler–Illingworth, split-based thresholding) | The scene itself | None (fitted per scene) | Assumes two populations. Fails when water is scarce (we add plausibility checks and a fallback) | High (means, spreads, boundary) | Low | **Yes** | **Very good**: standard practice in SAR flood mapping | "Learns each scene without labels": strong if explained |
| 5 | **Pretrained segmentation model** (e.g. networks trained on the Sen1Floods11 C-band Sentinel-1 dataset) | Inputs matching the training data | None / fine-tune | **Domain shift** to L-band NISAR, different polarisations and preprocessing. Opaque. Licence checks | Low | Medium (GPU helpful) | Risky | Only as a comparison on the Sentinel-1 fallback | Impressive if it works, damaging if unexplained |
| 6 | **Deep learning from scratch** | Thousands of labelled chips | GPU-hours | No NISAR flood labels exist yet. Overfitting | Low | High | **No** | Not in 48 h | Sceptical: "where is your training data?" |
| 7 | **AI explanation layer** (LLM) | Detector evidence (JSON) | None | Hallucination, overclaiming | High *if grounded and checked* | API call | Yes, with template fallback | Communication value | Good *if* clearly separated from detection |
| 8 | **Hybrid** (1 + 2 + 4 as detector, 7 for explanation; 3 later) | Pair + HAND + reference | None for MVP | Addresses the others' weaknesses | High | Low | **Yes** | **Best** | "Scientific core, AI where it adds value": the strongest story |

### 6.2 Recommended hybrid architecture (as implemented)

```text
NISAR L2 GCOV (HDF5) ──or── OPERA RTC-S1 (COG)          io_readers.py
        │  read AOI window only, keep CRS + transform, fill/NaN -> no-data
        ▼
Calibration / preprocessing          (already done by NASA: RTC gamma-0 power)
        ▼
Co-registration / alignment          align.py      same grid or explicit reprojection;
        │                                          refuse if overlap < 20 %
        ▼
Speckle reduction                    speckle.py    masked Lee filter (no-data safe)
        ▼
Change metric                        detect.py     dB, log-ratio = post_dB - pre_dB
        ▼
Unsupervised water model  (AI #1)    detect.py     2-component Gaussian mixture on pooled
        │                                          pre+post dB -> P(water-like) per pixel;
        │                                          plausibility + Otsu cross-check + fallback
        ▼
Change classification                detect.py     new water / persistent / receded /
        │                                          other decrease / other increase / masked
        ▼
Terrain + size filters               detect.py     HAND > 15 m masked; objects < 2 ha dropped
        ▼
Confidence & uncertainty             confidence.py evidence score + scene-quality issues
        ▼
Objects + evidence                   vectorize.py  polygons (EPSG:4326) with per-object stats
        ▼
Grounded explanation      (AI #2)    explain.py    template; optional LLM rewrite that is
        │                                          rejected if it adds any number
        ▼
Validation vs. reference             validate.py   OPERA DSWx-S1 agreement (+ by confidence)
        ▼
Map layers, report, provenance       render.py, pipeline.py -> outputs/<case>/ -> app/dashboard.py
```

### 6.3 The detection rule

Let γ⁰ be terrain-corrected backscatter power and x = 10·log₁₀(γ⁰) after Lee filtering.

1. **Water model.** Fit a 2-component Gaussian mixture to the pooled pre- and post-event dB
   values. The darker component is "water-like" (μ_w, s_w) and the brighter one "land-like"
   (μ_l, s_l). The posterior P_w(x) is forced to be monotonic, so darker never means "less
   water-like". (Unequal variances otherwise make very bright pixels look like water. The
   tests caught this.)
2. **Change.** Δ = x_post − x_pre.
3. **Possible flood-related change**, all of which must hold:
   - P_w(x_post) ≥ 0.5
   - P_w(x_pre) < 0.5
   - Δ ≤ −3 dB
   - HAND ≤ 15 m
   - the connected object is ≥ 2 ha
4. **Plausibility checks.**
   - The water mean must be ≤ −15 dB and its weight ≥ 0.5 %. Otherwise use a fixed,
     documented fallback threshold and cap confidence at 0.5. Scenes without water must not
     "invent" it (tested).
   - Separability is reported as Ashman's D = √2·|μ_l − μ_w| / √(s_w² + s_l²); D < 2 is
     flagged.
   - Otsu's threshold must agree with the mixture boundary within 3 dB, or the scene is flagged.

All thresholds are parameters in `DetectionParams` ([detect.py](../changelens/detect.py)) and
are written to `provenance.json`. They are **documented starting values, not tuned values**.
Tune them only on a case you do not also use to report agreement.

### 6.4 Confidence and uncertainty

Per flagged pixel, the confidence is C = (E_class × E_change × E_terrain)^(1/3), where:

- **E_class** = P_w(post) × (1 − P_w(pre)). The model is sure it is water now and was not before.
- **E_change** = logistic((−Δ − 3 dB) / σ_noise). σ_noise is the scene's own robust spread of
  Δ over stable land (1.4826 × MAD), so a 6 dB drop counts for more in a quiet scene than in a
  noisy one.
- **E_terrain** = 1 at HAND ≤ 5 m, falling to 0.5 at 15 m, and 0.75 when no HAND layer exists
  (the check was not possible, so it gets no full credit).

The geometric mean means one weak link pulls the score down.

Scene caps: weak bimodality → ≤ 0.6; fallback threshold → ≤ 0.5; methods disagree → ≤ 0.7.

Labels: **High ≥ 0.7, Medium ≥ 0.4, Low < 0.4.**

**Wording rule.** This is an *evidence score*, not a probability of flooding. The dashboard
says so. Once a case is validated, `validate.agreement_by_confidence` shows whether "High"
pixels actually agree with DSWx-S1 more often than "Low" ones. That is the only honest way to
say the score means something *for that case*.

Scene-level uncertainty is shown in plain language:

- water and land overlap;
- thresholds disagree;
- images more than 24 days apart;
- more than 30 % of the AOI masked;
- background change noise above 2 dB.

### 6.5 Explanation layer and LLM guardrails

- **The default is deterministic.** `template_explanation()` states the dB drop and the
  dates, the post-event value against the scene's learned boundary, the pre-event value, the
  HAND check, the size, the confidence ("not a probability"), look-alike causes, and a fixed
  disclaimer sentence.
- **The LLM rewrite is optional** (`--llm`, off by default). Claude is given *only* the
  evidence JSON and the template, and is told not to add places, causes, damage or numbers.
  The reply is **rejected automatically** if it:
  - contains any number not present in the evidence; or
  - drops or alters the disclaimer sentence; or
  - the model declines; or
  - the API errors.

  In all of those cases the template is shown. The request enables server-side refusal
  fallback (`fallbacks="default"`), so a declined request is retried on Anthropic's
  recommended model before falling back to the template.
- **The LLM never sees imagery, never chooses what is flagged, and never changes a number.**

### 6.6 What we deliberately do not do

- No LLM or vision model as a detector.
- No model trained on DSWx-S1 labels and then "validated" against DSWx-S1. That would be
  circular.
- No "accuracy" number without the reference, dates, masks and pixel counts next to it.
  Overall accuracy is not computed at all, because the "no change" majority class inflates it.
- No landslide or ice maps from backscatter differences.

---

## 7. System architecture

### 7.1 Components and why each is needed

| Component | Why |
|---|---|
| Python 3.10–3.12 | Common ground for undergraduates; the whole scientific stack |
| **rasterio** (pip wheels bundle GDAL) | Read/write GeoTIFF/COG with CRS, windowed reads, reprojection, polygonising. The wheels avoid a separate GDAL install, the classic hackathon time-sink |
| **h5py** | NISAR products are HDF5 |
| **NumPy / SciPy** | Array maths; `ndimage` for filtering and connected components |
| **pyproj / shapely** | CRS transforms; polygon union and centroid |
| **scikit-learn** | Gaussian mixture model |
| **PyYAML** | Human-readable case files that also serve as provenance |
| **Pillow** | Transparent PNG map overlays |
| **Streamlit** | A whole dashboard in one Python file; no JavaScript build |
| **folium (Leaflet)** | Interactive map with tooltips; plus an offline image mode |
| **pandas** | Tables (detections, provenance, validation) |
| **asf_search** (ASF) / **earthaccess** (NASA) | Official search and download clients; handle Earthdata Login |
| anthropic (optional) | Grounded plain-language rewrite of explanations |
| *GeoPandas* | *Optional.* Our vector work is small (a GeoJSON of polygons), so shapely + json suffice. Add it if the team prefers |
| *PyTorch* | *Not needed.* No deep-learning component in the MVP |
| *Cloud infrastructure* | *Not needed.* Everything runs on a laptop for a < 50 × 50 km AOI. Earthdata Cloud (AWS us-west-2) is a future option |

### 7.2 Data flow

```text
 NASA Earthdata / ASF DAAC / PO.DAAC            AWS Open Data
  NISAR GCOV (HDF5), OPERA RTC-S1 (COG),        GLO-30 HAND (COG)
  OPERA DSWx-S1 (GeoTIFF)                              │
        │ search: asf_search / earthaccess (no login)   │
        │ download: Earthdata Login (.netrc)            │
        ▼                                               ▼
  data/raw/  (never committed; IDs + URLs recorded in config/cases/*.yaml)
        │
        ▼  python -m changelens.pipeline --case config/cases/<case>.yaml
  outputs/<case_id>/
     change_classes.tif  confidence.tif  delta_db.tif      (native CRS, analysis)
     previews/*.png + map_bounds in summary.json           (EPSG:3857 overlays for the web)
     detections.geojson                                    (EPSG:4326 polygons + evidence)
     summary.json  report.md  provenance.json              (what the dashboard reads)
        │
        ▼  streamlit run app/dashboard.py
  Browser: map, layers, legend, confidence, uncertainty, provenance, explanations, downloads
```

### 7.3 API flow

1. **Discovery.** `scripts/find_pairs.py` queries ASF (NISAR) or CMR via earthaccess (OPERA).
   It groups granules by viewing geometry (track, frame, direction for NISAR; burst ID or MGRS
   tile for OPERA) and proposes before/after pairs around the event date.
2. **Authentication.** Earthdata Login via `earthaccess.login(persist=True)` or `~/.netrc`.
   Credentials never go into the repository.
3. **Download.** `--download N` fetches the chosen pair to `data/raw/`. The ASF Vertex and
   Earthdata Search web UIs are the manual fallback.
4. **Processing.** Local CLI. No server.
5. **Serving.** Streamlit reads local files.

### 7.4 Frontend–backend interaction

There is no separate backend. The **pipeline is the backend**: a batch job that writes
files. The **dashboard is read-only**: it never computes detections, which keeps the demo fast
and reproducible and makes the offline mode trivial.

A "live re-run" (time permitting) would call `analyze_pair()` on a cached AOI with a
different threshold and show the new result next to the saved one, never silently replacing it.

### 7.5 Provenance handling

- Every dataset carries a `source` dict (dataset, short name, granule ID, URL, maturity,
  **label**). `Provenance.add_dataset()` refuses unknown labels.
- Every processing step is recorded with its parameters.
- Software versions (Python, NumPy, rasterio, GDAL, ...) are recorded.
- Every output file is recorded with its SHA-256 checksum.
- NISAR `identification` metadata (start time, track, frame, pass direction, granule ID) is
  copied from the HDF5 into provenance automatically.

### 7.6 Model-output format

**Change classes** (`change_classes.tif`, uint8, no-data 0; descriptions stored in the GeoTIFF tags):

| Code | Name | Meaning | Map colour group |
|---|---|---|---|
| 0 | nodata | No data / outside swath / masked input | transparent |
| 1 | no_change | No significant change | transparent |
| 2 | new_water | **Possible flood-related change** | blue `#2a78d6` |
| 3 | persistent_water | Water-like in both images | aqua `#1baf7a` |
| 4 | receded_water | Water-like before, not after | orange `#eb6834` (other) |
| 5 | other_decrease | Strong darkening, not reported as flood | orange (other) |
| 6 | other_increase | Strong brightening (unclassified) | orange (other) |
| 7 | masked_terrain | HAND above threshold | grey `#898781` |

The three map colours were checked with a colour-vision-deficiency validator. Aqua has low
contrast on white, so the legend and a table view always accompany the map.

**`detections.geojson`** (RFC 7946, lon/lat).

- Top-level fields: `disclaimer`, `case_id`, `hazard`, `mode`.
- Each feature has these properties: `id`, `hazard`, `label`, `area_km2`, `pixels`,
  `centroid_lat/lon`, `mean_delta_db`, `mean_pre_db`, `mean_post_db`, `mean_hand_m`,
  `confidence`, `confidence_label`, `explanation`.

**`summary.json`** has these keys:

- `status` (`ok` / `degraded` / `failed`), `mode` (real-data or dry-run), `hazard`,
  `hazard_label`, `disclaimer`
- `scene` (dates, polarization, dataset, threshold)
- `datasets`, `water_model`, `parameters`
- `quality` (overall + issues), `confidence_notes`, `stats` (area per class, noise)
- `detections`, `detected_area_km2`
- `validation` (metrics + by-confidence), `explanations`, `legend`, `map_bounds_latlon`

### 7.7 Error-handling strategy

| Situation | Behaviour |
|---|---|
| File missing / unreadable / not GCOV / missing polarization | `DataLoadError` with a specific message (lists available covariance terms) |
| AOI outside the image | `DataLoadError` |
| Grids cannot be aligned, or overlap < 20 % | `AlignmentError` ("pick acquisitions from the same track/frame") |
| Geographic (degree) CRS where metres are needed | `AlignmentError`: refuses to compute areas in degrees |
| Too few valid pixels | `InsufficientDataError` |
| Any of the above during `--case` | `summary.json` gets `status: failed`, **stale maps are deleted**, exit code 2. The dashboard shows the error and **no map** |
| Case file still a template | Refuses to run |
| No water population in the scene | Fallback threshold, confidence capped, warning shown |
| Poor scene quality | `status: degraded`; caveats in the dashboard and report |
| Search / download failure | Clear message pointing to the offline fallback |
| LLM error / refusal / ungrounded text | Template explanation (default anyway) |
| Unsupported hazard selected | Dashboard explains why; no map |

### 7.8 Offline / demo-mode strategy

- **Before judging:**
  1. Process the demo cases.
  2. `git add -f` their small web assets (PNG previews, GeoJSON, JSON, report).
  3. Do a fresh-clone test with Wi-Fi off.
- **Dashboard "Demo mode"** lists only successfully processed cases.
- **"Offline image" map mode** draws the layers over the radar image with no internet. Leaflet
  and basemap tiles need the web, so this is the Wi-Fi-failure fallback.
- **Dry-run mode** (`--dry-run`) is for development only. The data is synthetic, labelled on
  every screen, and placed over open ocean so nobody can mistake it for a real place.

---

## 8. Dashboard design

### 8.1 Primary user

**A disaster-response GIS analyst**, for example at a national or provincial disaster-management
agency or a humanitarian NGO.

- **Their problem.** After days of heavy rain, optical satellites see only cloud. Raw radar
  images are hard to read without SAR training. Official flood products may not be activated
  for their area, or may arrive later.
- **What ChangeLens gives them.** A quick, *explained* first look at where open water newly
  appeared, with the evidence and uncertainty on screen and downloadable GeoJSON for their
  own GIS.
- **The decision it supports.** **Where to look first**: which areas to prioritise for
  checking against official products, field reports or higher-resolution imagery. It does
  **not** support evacuation or damage decisions. The tool says so.

Students and educators benefit too, but the design choices (evidence, GeoJSON export,
caveats) are made for the analyst.

### 8.2 Layout (implemented)

```text
┌─────────────── sidebar ───────────────┐┌──────────────────────── main ─────────────────────────────┐
│ Study area                            ││ ChangeLens: radar surface-change screening                │
│ [x] Demo mode (prepared cases only)   ││ ⚠ Screening prototype - not an official warning (always)  │
│ Case  [ Assam, Aug 2026 (NISAR) ▾]    ││ ⛔ SYNTHETIC DATA banner (dry-run cases only)             │
│ Hazard view                           ││ Before 2026-07-01 | After 2026-07-13 | Flagged 2 | 2.9 km2│
│ (•) Flood (MVP)                       ││ NISAR L2 GCOV | HH | scene quality: good                   │
│ ( ) Forest disturbance (Experimental) ││┌──────────── map ─────────────┐ ┌ Why was this flagged? ┐│
│ ( ) Landslide (Future extension)      │││ basemap + layer + outlines   │ │ [FL-001 ▾]            ││
│ ( ) Ice / glacier (Future extension)  │││ tooltips: id, label, conf.   │ │ evidence text ...     ││
│ Map layer                             ││└──────────────────────────────┘ │ Uncertainty & limits ▸││
│ (•) Change classes  ( ) Confidence    ││ legend: ■ possible flood-related │ Data provenance ▸     ││
│ ( ) Change intensity (dB)             ││ ■ other change ■ water both ■ masked│ Agreement w/ ref ▸  ││
│ ( ) Before image ( ) After image      ││                                  └───────────────────────┘│
│ [x] Outline flagged areas             ││ Flagged areas (table view)                                │
│ Map display: (•) Interactive ( ) Offline││ [Download report.md] [detections.geojson] [summary.json] │
└───────────────────────────────────────┘└───────────────────────────────────────────────────────────┘
```

### 8.3 Features and priority

| # | Feature | Priority | Status in repo |
|---|---|---|---|
| 1 | Map view | Essential | Done (folium/Leaflet + offline image) |
| 2 | Study-area selector | Essential (prepared cases) | Done as case selector. *Free-form AOI drawing: future* |
| 3 | Before/after date selector | Essential (one pair per case) | Done (dates shown per case). *Several pairs per AOI: useful if time permits* |
| 4 | SAR layer display | Essential | Done (before/after backscatter) |
| 5 | Change-intensity layer | Essential | Done (diverging Δ dB, blue = darker after) |
| 6 | Detected-change polygons | Essential | Done (outlines, tooltips, table view) |
| 7 | Hazard category label | Essential | Done ("Possible flood-related change"; unsupported hazards explained) |
| 8 | Confidence score | Essential | Done (layer + per object + "not a probability") |
| 9 | Uncertainty and limitations panel | Essential | Done |
| 10 | Data provenance panel | Essential | Done (datasets + labels + provenance.json) |
| 11 | "Why was this area flagged?" | Essential | Done (template). *LLM rewrite: useful if time permits* |
| 12 | Downloadable report / summary | Useful if time permits | Done (report.md, GeoJSON, JSON). *PDF: future* |
| 13 | Demo mode with prepared cases | Essential | Done (toggle + offline map) |
| 14 | "Not an official warning" notice | Essential | Done (banner, GeoJSON, report, explanations) |
| – | Agreement-with-reference panel | Useful if time permits | Done when a reference is configured |
| – | Live threshold slider with re-run | Useful if time permits | Not started |
| – | Time-series view, alerts, user feedback | Future | Not started |

### 8.4 Visual rules

- One categorical palette of three colours for the class map, validated for colour-vision
  deficiency; low-contrast colours are always backed by a legend and a table.
- Confidence uses a single-hue light-to-dark ramp. Change intensity uses a blue↔red diverging
  map with a transparent neutral midpoint.
- Text never uses the series colour.
- The legend is always present.
- The table view repeats everything the map shows.
- Status banners use words and icons, not colour alone.

---

## 9. Data and processing pipeline

| Step | What happens | Where | Key safeguards |
|---|---|---|---|
| 1. Configure | The case YAML names the granules, AOI, HAND tile, reference and parameters; every dataset carries a label | `config/cases/` | Template cases refuse to run |
| 2. Read | NISAR: `/science/LSAR/GCOV/grids/frequencyA/HHHH`, `xCoordinates`, `yCoordinates`, `projection@epsg_code`; only the AOI window is read. GeoTIFF: windowed read | [io_readers.py](../changelens/io_readers.py) | CRS from the file, never guessed. Pixel-centre coordinates → corner transform (one documented function). Fill / non-finite / ≤ 0 power → NaN |
| 3. Align | Same grid? Keep. Otherwise reproject onto the "before" grid | [align.py](../changelens/align.py) | Bilinear on *linear* power; nearest for categories; ≥ 20 % overlap required |
| 4. Speckle filter | Lee filter, window 5, ENL 4 (estimate with `estimate_enl`) | [speckle.py](../changelens/speckle.py) | No-data-aware (normalised convolution) |
| 5. dB and log-ratio | 10·log₁₀; Δ = post − pre | [detect.py](../changelens/detect.py) | Invalid in either date → invalid in both |
| 6. Water model | Pooled 2-component GMM → P(water-like) | `fit_water_model` | Monotonic posterior; plausibility; Otsu cross-check; fallback |
| 7. Classify | Codes 0–7 (see 7.6) | `classify_flood_change` | HAND mask; 2 ha minimum mapping unit |
| 8. Confidence | Evidence score + scene quality | [confidence.py](../changelens/confidence.py) | Caps when the scene is ambiguous |
| 9. Objects | Connected components → polygons in EPSG:4326 with per-object stats | [vectorize.py](../changelens/vectorize.py) | Areas from the projected grid (metres), never degrees |
| 10. Validate | DSWx-S1 new water on our grid → precision / recall / F1 / IoU + by-confidence | [validate.py](../changelens/validate.py) | Only pixels valid in both; DSWx masks (250, 251, 255) excluded |
| 11. Explain | Template (or grounded LLM) per object | [explain.py](../changelens/explain.py) | Numbers checked against the evidence |
| 12. Render + write | COG-style GeoTIFFs (native CRS), EPSG:3857 PNG overlays, GeoJSON, summary, report, provenance | [render.py](../changelens/render.py), [pipeline.py](../changelens/pipeline.py) | Checksums; disclaimers in every file |

**Day-one checks on the first real NISAR granule** (the NISAR reader has only been tested on a
mock file built from the documented layout):

1. List the file's structure with `h5ls -r granule.h5 | head -100`, or open it in HDFView or
   Panoply. Confirm the group path, dataset names and `epsg_code`.
2. Confirm that `xCoordinates` / `yCoordinates` are **pixel centres**. Overlay the "before"
   image on OpenStreetMap and check a river bank or road. A half-pixel shift is 5–10 m; a
   whole-image offset means the convention is wrong.
3. Confirm the fill value: NaN, or a `_FillValue` attribute.
4. Confirm the units are linear power (γ⁰): typical land values are about 0.01–0.3, not
   −30 to 0.

---

## 10. Validation plan

### 10.1 Principles

- **Compare, don't claim.** DSWx-S1 is a validated NASA *product*, not ground truth. We report
  **agreement with DSWx-S1**, never "accuracy".
- **Report the denominator.** Every number comes with the reference name and dates, the date
  gap to our images, the masks applied, and the number of pixels compared.
- **Precision, recall, F1 and IoU**, not overall accuracy.
- **At least one negative (unchanged) case** and **one known false-positive trap**.
- **Visual inspection** where no reference exists. Say so.
- **No threshold tuning on the case used to report agreement.** If you tune, tune on a
  different case or AOI.

### 10.2 Metric definitions (for the slide)

- **Precision:** of the area we flagged, the share DSWx-S1 also marks as new water.
- **Recall:** of DSWx-S1's new water, the share we also flagged.
- **F1:** the balance of the two.
- **IoU:** overlap divided by union.
- **By confidence:** agreement for High, Medium and Low objects separately. High should agree
  more often.

### 10.3 Test cases

| Test case | Expected change | Data used | Detection result | Validation method | Known limitations |
|---|---|---|---|---|---|
| **V1: NISAR flood (primary)** | New open water along the floodplain | NISAR L2 GCOV PROVISIONAL HH, same track/frame; GLO-30 HAND | *Fill in at hour 28:* flagged km², objects, confidence mix | Pixel agreement vs. OPERA DSWx-S1 new water (nearest dates) + by-confidence table + visual check against NASA Disasters layers if they exist | C-band vs. L-band; date gaps; DSWx-S1 has its own errors; inundated vegetation handled differently |
| **V2: OPERA fallback flood (North America)** | New open water | OPERA RTC-S1 VV pre/post; HAND | *Fill in* | vs. DSWx-S1 (**consistency check, not independent**: DSWx-S1 is derived from RTC-S1); vs. ARIA flood proxy map if one exists | Same-sensor reference inflates agreement |
| **V3: Negative / unchanged** | Nothing | NISAR pair over a stable period or area | *Fill in*; expect ~0 km² | False-alarm area per 1,000 km² | Seasonal drift can still produce a few objects |
| **V4: Permanent-water control** | River / lake stays "persistent water", not "new water" | Same as V1 | *Fill in* | Share of the river core classified as persistent | Rivers narrower than about 2 pixels are lost |
| **V5: Known false-positive trap** | Should be masked or low confidence | Steep terrain (shadow), wet snow, or paddies in V1's scene | *Fill in* | Visual + class statistics inside the trap area | HAND does not catch paddies. Shown as an honest limitation |
| **V6: Forest disturbance (experimental)** | HV decrease inside a 2026 burn scar | NISAR GCOV HV pre/post | *Fill in* | Overlap with NASA FIRMS detections / official perimeter; DIST-ALERT-S1 if available | Single pair; moisture confounding; fixed low confidence |
| **V7: Synthetic dry run (software test only)** | Injected floodplain recovered; trap masked; river persistent; unchanged pair → 0 | `changelens.synthetic` | **Passes in CI:** IoU > 0.9 with the injected flood; 0 detections on the unchanged pair; trap > 90 % masked | Automated tests (`pytest`) | Tests the code, **not** the science |

### 10.4 Where false positives and false negatives will come from

- **False positives:** shadow and wet snow (mostly HAND-masked); flooded paddies and irrigation
  (not masked; shown as a limitation); smooth new surfaces; misregistration edges.
- **False negatives:**
  - flooding under forest or in cities (double bounce makes it *brighter*);
  - wind-roughened water;
  - water that rose and fell between the two acquisitions;
  - floods narrower than the 2 ha minimum mapping unit;
  - high-HAND flooding (for example backwater in steep valleys);
  - areas lost to layover.

---

## 11. Project folder structure

```text
sar_change_lens/
├── README.md                     quick start
├── requirements.txt              pinned-minimum dependencies (rasterio wheels include GDAL)
├── requirements-dev.txt          + pytest
├── pytest.ini
├── .gitignore                    raw data and outputs are never committed by accident
├── docs/
│   └── PROJECT_PLAN.md           this document
├── config/cases/
│   ├── TEMPLATE_nisar_gcov_flood.yaml   primary pathway (fill in TODOs)
│   └── TEMPLATE_opera_rtc_flood.yaml    fallback pathway
├── changelens/                   the pipeline (a Python package)
│   ├── scene.py                  SarScene: pixels + CRS + transform + provenance; error types
│   ├── io_readers.py             NISAR GCOV (HDF5) and GeoTIFF/COG readers
│   ├── align.py                  grid checks and reprojection
│   ├── speckle.py                no-data-aware Lee filter, ENL estimate
│   ├── detect.py                 GMM water model, change classes, forest experiment
│   ├── confidence.py             evidence score, scene quality
│   ├── vectorize.py              polygons + per-object evidence (GeoJSON)
│   ├── explain.py                template + grounded optional LLM explanations
│   ├── validate.py               DSWx-S1 codes, agreement metrics
│   ├── render.py                 Web-Mercator PNG overlays, validated colours
│   ├── provenance.py             datasets, steps, versions, checksums; data labels
│   ├── synthetic.py              SYNTHETIC test scenes (over open ocean)
│   └── pipeline.py               CLI: --dry-run | --case; writes outputs/<case>/
├── app/
│   └── dashboard.py              Streamlit dashboard (read-only)
├── scripts/
│   └── find_pairs.py             search NISAR / OPERA pairs; optional download
├── tests/                        51 tests: science, georeferencing, pipeline, dashboard, configs
├── data/raw/                     downloaded granules (git-ignored)
└── outputs/                      pipeline results (git-ignored; force-add demo cases)
```

---

## 12. Implementation roadmap

| # | Step | Commands / actions | Done when | Status |
|---|---|---|---|---|
| 1 | Environment setup | `python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt && pytest` | 51 tests pass (2 skip without the optional anthropic package) | Ready |
| 2 | Data acquisition | Earthdata Login for all; `python scripts/find_pairs.py nisar-gcov --bbox ... --event ...`; `--download N`; HAND tile from AWS; DSWx-S1 tiles | Two NISAR granules + HAND + reference on disk; IDs in YAML | **Day one** |
| 3 | Data inspection | `h5ls -r`; the four checks in [section 9](#9-data-and-processing-pipeline); view the "before" image | CRS, transform, fill value and units confirmed | **Day one** |
| 4 | Preprocessing | Reader with AOI bbox; alignment; Lee filter; check histograms in dB | Clean dB images on one grid | Code ready |
| 5 | Core change detection | `python -m changelens.pipeline --case config/cases/<case>.yaml` | First real change map | Code ready |
| 6 | AI component | GMM water model (built in); inspect `water_model` in `summary.json` (D > 2? plausible means?) | Plausible model, no fallback | Code ready |
| 7 | Confidence | Check High/Medium/Low proportions; read scene issues | Scores make visual sense | Code ready |
| 8 | Map visualisation | Previews render; overlay aligns with OpenStreetMap features | River bank lines up | Code ready |
| 9 | Dashboard | `streamlit run app/dashboard.py`; polish texts for the real case | Real case browsable online and offline | Code ready |
| 10 | Validation | Add the DSWx-S1 reference to the YAML; fill in the table in [section 10](#10-validation-plan) | V1, V3, V4, V5 filled in | Day two |
| 11 | Documentation | README results section; provenance slide; limitations slide | A stranger can reproduce the result | Day two |
| 12 | Presentation | Slides, 2-min demo video, rehearsals | 3 timed rehearsals done | Day two |

Pseudocode of the whole run (the real code is in `pipeline.analyze_pair` and `run_case`):

```python
pre  = read_nisar_gcov_scene(pre_path,  "HH", bbox=aoi)       # linear power, NaN = no-data
post = read_nisar_gcov_scene(post_path, "HH", bbox=aoi)
post = align_to_reference(pre, post)                          # same grid or refuse
pre_db, post_db = to_db(lee_filter(pre.data)), to_db(lee_filter(post.data))
model  = fit_water_model(pre_db, post_db, params)             # unsupervised, per scene
result = classify_flood_change(pre_db, post_db, model, params, pixel_area, hand)
conf, notes = new_water_confidence(result, model, params, hand)
objects = change_objects(result.classes == NEW_WATER, pre.transform, pre.crs, ...)
for obj in objects: obj.explanation = template_explanation(evidence_for(obj, scene))
write_outputs(...)  # GeoTIFFs, PNG overlays, GeoJSON, summary, report, provenance (+ checksums)
```

---

## 13. 48-hour schedule

Two changes from the suggested schedule:

- **The dashboard starts at hour 6, not hour 28.** It is built against the dry-run outputs,
  which share the real output format.
- **There are explicit go/no-go gates.** The critical path is data, not code.

| Hours | SAR / data (P1) | Geo-processing (P2) | AI / detection (P3) | Dashboard (P4) | Science comms / pitch (P5) | Gate |
|---|---|---|---|---|---|---|
| **0–2** | Read the NISAR challenge page and judging criteria. Check local rules on prepared data. Earthdata Logins for all | Environment: `pip install`, `pytest`, dry run | Read `detect.py`; agree thresholds | `streamlit run` on dry-run outputs | Draft the user story and one-sentence pitch | **G0: scope frozen** (flood primary, forest experimental); everyone has run the dry-run dashboard |
| **2–6** | `find_pairs.py` for 2–3 candidate events; download the best NISAR pair + DSWx-S1 + HAND. **In parallel**, one OPERA RTC-S1 fallback pair | Inspect granule structure (`h5ls`); first read of the AOI | Fill in the case YAML; plan the negative case | Layout polish on synthetic outputs | SAR explainer text; data-provenance slide | **G1 (hour 6): is the NISAR pair on disk and opening?** Yes → NISAR primary. No → OPERA fallback primary, and keep trying NISAR until hour 12 |
| **6–12** | Metadata into YAML/provenance; the negative-case pair | Georeferencing checks (centre vs. corner, overlay on OSM); alignment | Histograms, ENL estimate, first GMM fit | Wire real `summary.json` fields | Storyboard the 2-min demo | **G2 (hour 12): first real change map exists** (even if ugly) |
| **12–20** | Forest case data (**only if G2 passed on time**) | Performance: AOI size, windowed reads | Detector on real data; HAND mask; minimum mapping unit; negative case | Map layers, legend, table with real data | Validation table skeleton; limitations slide | *Sleep shifts: P1+P4 at hours 14–18, P2+P3 at 18–22, P5 at 16–20* |
| **20–28** | Reference-date bookkeeping | DSWx-S1 reprojection check | Confidence review; validation metrics; forest experiment | Validation panel, provenance panel, downloads | Fill in V1/V3/V4/V5 results; screenshots | **G3 (hour 28): all demo cases processed; validation numbers recorded** |
| **28–36** | Authenticity checklist evidence (IDs, URLs, metadata) | Offline assets: `git add -f outputs/<case>/…` | Optional LLM rewrite (**first to cut**) | Text polish, offline mode, accessibility pass | Slides v1; demo script v1 | **Feature freeze at hour 36** |
| **36–40** | Fresh-clone test | Wi-Fi-off test | Re-run all cases from clean | Bug fixes only | Slides v2 | All tests pass; clean reproduction |
| **40–44** | Data section of README | Architecture diagram | Method slide | Record a 2-min demo video (backup) | Rehearsal 1 (timed) | **Code freeze at hour 44** |
| **44–48** | Q&A drill (data) | Q&A drill (processing) | Q&A drill (AI, uncertainty) | Submission page, repo link, video | Rehearsals 2–3; submit by **hour 47** | Submitted with 1 h buffer |

**Critical path.** Earthdata Login → pair found → granules downloaded → reader works on the
real file → first change map → real case in the dashboard → validation numbers → rehearsed
demo.

**Backup if the NASA APIs fail.** Try these in order:

1. ASF Vertex web UI (manual search and download).
2. Earthdata Search web UI.
3. Stream only the AOI window (`asf_search` + `fsspec`, per the NISAR guide).
4. Pre-downloaded granules on a USB stick or shared drive (if the rules allowed scouting).
5. The OPERA RTC-S1 fallback case.
6. As a last resort, show the last successfully processed outputs, *with their processing
   timestamp visible*.

**Backup if the AI component fails.**

- The GMM is implausible → fixed fallback threshold with capped confidence (automatic). Or
  pick an AOI that includes a river, so the water class exists.
- The LLM fails → template explanations (already the default).

**Features to cut, in this order:**

1. LLM rewrite
2. Forest-disturbance demo
3. Second flood case
4. Interactive-map polish (keep the offline mode)
5. Report formatting

**Never cut:** disclaimers, provenance, the negative case, the uncertainty panel.

**Rehearsal plan.**

- **Hour 44:** full timed run (2-min demo + 5-min pitch).
- **Hour 45:** Q&A drill with the four standard questions (15.10–15.12) plus "what does dB
  mean?", "why L-band?" and "what if it's a rice field?". One teammate plays a sceptical judge.
- **Hour 46:** full run in offline mode.
- **Hour 47:** final run, then submit.

---

## 14. Team roles

**Five people:**

| Role | Owns | Also backs up |
|---|---|---|
| **P1: SAR / data specialist** | Dataset scouting, Earthdata, `find_pairs.py`, case YAMLs, provenance, the physical explanation of the data, the authenticity checklist | Validation reference data |
| **P2: Geospatial-processing developer** | Readers, CRS / transform checks, alignment, HAND and DSWx reprojection, rendering, performance, offline assets | Dashboard map |
| **P3: AI / change-detection developer** | GMM water model, change rules, confidence, validation metrics, forest experiment, explanation grounding (and the optional LLM) | Geo-processing |
| **P4: Dashboard / frontend developer** | Streamlit app, map layers, legend, tables, downloads, offline mode, UX copy with P5 | Demo video |
| **P5: Science communicator, UX, validator, pitch lead** | User story, validation table, false-positive gallery, slides, demo script, rehearsals, judge Q&A, the wording rules ("possible", "candidate", "not official") | Visual QA of maps |

**Four people:** merge P2 and P3. **Three people:** A = P1 + P5-validation; B = P2 + P3;
C = P4 + P5-pitch.

---

## 15. Judge-facing pitch and demo

**1. One-sentence pitch.**
"ChangeLens turns two NASA NISAR radar images, which see through clouds day and night, into an
explained map of where new open water appeared, so a disaster-response analyst can decide
where to look first, with the evidence and uncertainty on screen."

**2. 30-second explanation.**
"When floods happen, clouds block optical satellites. NASA and ISRO's NISAR radar sees through
them. But raw radar is hard to read. ChangeLens takes a NISAR image from before and one from
after, learns what water looks like in that specific scene, and flags land that turned into
open water. Every flagged area shows how much the signal changed, how confident we are and
why, and what else could look the same. It is a screening tool: it tells you where to look,
not what to do."

**3. Two-minute demo script.**

| Time | On screen | Say |
|---|---|---|
| 0:00–0:15 | Title and warning banner | "This is a screening prototype, not an official warning: that banner never goes away." |
| 0:15–0:35 | Before image, then after image | "Two NISAR L-band images, [dates], same orbit. Dark means smooth, like calm water. Notice what's new." |
| 0:35–0:55 | Change classes layer + outlines | "Blue is possible flood-related change: water-like now, not before, at least 3 dB darker. Grey is masked, because it is too high above the river to flood; those are hillside shadows." |
| 0:55–1:15 | Click FL-001, "Why was this flagged?" | "Here is the evidence: it dropped [N] dB, it is now darker than the water boundary our model learned from this scene, and it is on low-lying ground. [High] confidence, and it tells you this isn't a probability." *(Read the real numbers off the screen; never script them.)* |
| 1:15–1:35 | Uncertainty panel + Agreement panel | "Compared with NASA's own OPERA water product from Sentinel-1, a different radar, we agree on X % of flagged area. High-confidence areas agree more than low ones." |
| 1:35–1:50 | Data provenance | "These are the exact NISAR granules and checksums. Remove the NASA data and there is nothing to show." |
| 1:50–2:00 | Switch to the unchanged case, then to Landslide | "Unchanged pair: nothing flagged. Landslides: we explain why we *don't* map them yet." |

**4. Five-minute presentation outline.**

1. The problem (0:30): floods plus clouds; the analyst's triage question.
2. Why radar and why NISAR (0:45): active microwaves; L-band; calibrated data public since July 2026.
3. What we built (0:45): pipeline diagram, where the AI sits and where it does not.
4. Live demo (2:00): as above.
5. How much to trust it (0:30): validation table, negative case, false-positive gallery.
6. What's next (0:30): DSWx-NI, time-series baselines, inundated vegetation, landslides via coherence.

**5. Why SAR is valuable.** It works through cloud and at night, when floods happen. It
measures a physical property (surface roughness and moisture) consistently, so "before" and
"after" can be compared number to number, not just by eye.

**6. Why NASA data.**

- The core input is NASA-ISRO NISAR L-band data, calibrated and terrain-corrected by NASA JPL
  and distributed by NASA's ASF DAAC.
- The independent check is NASA JPL's OPERA DSWx-S1 product.
- The fallback is NASA's OPERA RTC-S1.
- Every output traces back to specific granule IDs.

**7. What the AI does.** Two things:

- An unsupervised statistical model learns, *for each scene*, what water-like and land-like
  radar backscatter look like. Water's brightness varies with sensor, angle and wind, so fixed
  thresholds transfer badly. The model turns every pixel into a probability, with no training
  labels.
- An explanation generator turns the measured evidence into plain language. If a language
  model is used, a checker rejects any sentence containing a number that was not measured.

The AI never decides alone: physics rules and terrain checks do.

**8. Uncertainty.** Every flagged area has an evidence score built from three checks: model
certainty, strength of change against the scene's noise, and terrain. Every scene also has
plain-language warnings. We show how often high-confidence areas agree with NASA's reference
product, and we show where the method fails: shadows, wet snow, rice paddies, and floods under
trees.

**9. "Why is this better than just viewing satellite images?"** "Looking at two radar images
tells you something changed if you are a radar expert. ChangeLens tells a non-expert *where*,
*how much*, *how sure*, and *why*, as polygons they can load into their own GIS, with the
look-alikes listed. It also processes the whole scene consistently, instead of relying on
where someone happened to look."

**10. "Can this be trusted during a real disaster?"** "Not as a decision-maker, and we say so.
It is a screening layer: it helps an analyst decide where to look first. Our agreement with
NASA's validated water product is [numbers, with dates], and we show the failure cases.
Operational use would need validation across many events and official review. That is what
programmes like NASA OPERA and the Disasters Program do."

**11. "Why did you choose floods first?"** "Because it is the hazard where radar physics is
strongest and simplest: calm water is dark. NASA also publishes an independent reference we
can check against. Landslides and ice need interferometry or offset tracking to be done
honestly, and we would rather ship one reliable layer than four unreliable ones."

**12. "What would you build next?"**

1. Switch the reference to NASA's NISAR-based water product (OPERA DSWx-NI) when it is released.
2. Multi-date baselines as NISAR's record grows, so seasonal change is not mistaken for floods.
3. Flooded-vegetation detection using L-band's double-bounce signal.
4. Landslide screening with NISAR interferometric coherence, checked against NASA's landslide
   catalog.

---

## 16. Risks and fallback plans

| Risk | Likelihood | Impact | Mitigation / fallback |
|---|---|---|---|
| No clean NISAR pair (the event predates 2026-06-17, or has coverage gaps) | Medium | High | Gate G1 at hour 6; OPERA RTC-S1 fallback case; strong seasonal inundation as the NISAR case |
| NISAR HDF5 layout differs from the documentation | Low–medium | Medium | The reader lists available datasets on error; path constants sit in one place; `h5ls` first |
| Pixel-centre vs. corner convention wrong | Low | Low–medium (10 m shift) | One-line change in `transform_from_center_coords`; OSM overlay check |
| Granule too large / slow download | Medium | Medium | AOI window reads; streaming per the NISAR guide; start downloads by hour 3 |
| Earthdata Login problems | Medium | High | Accounts at hour 0; `.netrc`; one member's account as backup |
| GMM implausible (little water in the AOI) | Medium | Medium | Automatic fallback + capped confidence; include a river in the AOI |
| False positives dominate (paddies, shadow) | Medium | Medium | HAND, minimum mapping unit, confidence; choose the AOI; show one as a limitation |
| DSWx-S1 dates far from NISAR dates | Medium | Low–medium | Nearest dates; report the gap; call it agreement |
| Venue Wi-Fi fails | Medium | High | Offline image mode; committed outputs; recorded demo video |
| Dependency or version issues | Medium | Medium | `requirements.txt`; fresh-clone test at hour 36; Python 3.10–3.12 |
| Scope creep (landslide / ice) | High | High | Frozen at G0; the dashboard explains why they are excluded |
| Overclaiming in the pitch | Medium | High | Wording rules; P5 reviews all slides |
| LLM hallucination | Low (checked) | Medium | Template default; number/disclaimer check; the LLM is optional |
| Burnout | High | Medium | Sleep shifts; feature freeze at hour 36 |

---

## 17. NASA-data authenticity checklist

| # | Question | Our answer | How to demonstrate it to judges |
|---|---|---|---|
| 1 | Does the main output change when the NASA data changes? | Yes | Live: switch from the flood pair to the unchanged NISAR pair; detections drop to ~0. Or re-run with a different "after" granule |
| 2 | Is NASA data used in the core processing pipeline? | Yes: NISAR GCOV γ⁰ is the detector's direct input | Show `read_nisar_gcov_scene` reading `/science/LSAR/GCOV/grids/frequencyA/HHHH`; show the pipeline diagram |
| 3 | Can the team show the original source and metadata? | Yes | Provenance panel: granule IDs, ASF URLs, acquisition time, track/frame/direction pulled from the HDF5 `identification` group; checksums in `provenance.json` |
| 4 | Is an appropriate NASA product used, not just a logo? | Yes: RTC-calibrated L-band backscatter for water detection; OPERA DSWx-S1 as reference | Explain why GCOV (terrain-corrected γ⁰) rather than raw data, and why DSWx-S1 is the right comparison |
| 5 | Can the team explain the physical meaning of the data? | Yes | "γ⁰ is the radar energy returned per unit area; calm water mirrors it away, so it is dark; L-band sees through canopy, which makes flooded forests *brighter*" |
| 6 | Are limitations and uncertainty visible? | Yes | Uncertainty panel, confidence layer, scene caveats, provisional-maturity label, negative case |
| 7 | Would the project lose its main function without NASA data? | Yes | With no granule the pipeline refuses to run. The only other data is labelled SYNTHETIC and placed over the ocean |

---

## 18. Future extensions

1. **OPERA DSWx-NI** (NISAR-native water product), when released, as the primary reference.
2. **Time-series baselines.** As NISAR's calibrated record grows, move from a single pair to
   per-pixel statistics over many dates (anomaly vs. seasonal normal). This removes many
   agricultural and seasonal false positives.
3. **Flooded vegetation / urban flooding.** L-band HH double bounce, HH/HV ratios, land-cover
   context.
4. **Forest disturbance, properly.** Multi-date HV baselines; compare with OPERA DIST-ALERT-S1
   and NASA FIRMS.
5. **Landslide screening.** NISAR GUNW coherence loss + slope/DEM masks; checked against NASA
   COOLR inventories; OPERA DISP-S1 for slow-moving slopes in North America.
6. **Ice.** NISAR pixel offsets or ITS_LIVE velocities; calving-front change.
7. **Free-form AOIs** with on-demand processing near the data (NASA Earthdata Cloud, AWS
   us-west-2).
8. **Calibrated confidence** from many validated events (reliability curves).
9. **Accessibility and languages.** Screen-reader friendly tables; translated explanations
   (grounded the same way).
10. **Analyst feedback loop.** Let users mark true or false detections to build honest,
    documented labels for future supervised models.

---

## 19. Sources and assumptions

### 19.1 Official sources consulted (verified via search on 2026-09-26; re-open before the event)

**Space Apps**

- 2026 NASA Space Apps Challenge: <https://www.spaceappschallenge.org/2026/>
- Challenges list: <https://www.spaceappschallenge.org/2026/challenges/>
  (the NISAR surface-change challenge text is quoted from a search extract)
- 2025 SAR challenge, for context: <https://www.spaceappschallenge.org/2025/challenges/through-the-radar-looking-glass-revealing-earth-processes-with-sar/>

**NISAR**

- Available data: <https://nisar-docs.asf.alaska.edu/availability-overview/>
- Products overview: <https://nisar-docs.asf.alaska.edu/products-overview/>
- GCOV: <https://nisar-docs.asf.alaska.edu/gcov/>
- Provisional known issues: <https://nisar-docs.asf.alaska.edu/provisional-known-issues/>
- earthaccess: <https://nisar-docs.asf.alaska.edu/earthaccess/>
- asf_search: <https://nisar-docs.asf.alaska.edu/asf-search/>
- Data format (HDF5): <https://nisar-docs.asf.alaska.edu/data-format/>
- NISAR L-band data release news: <https://www.earthdata.nasa.gov/news/nisar-l-band-data-released-expanding-record-surface-changes>
- NISAR L-band data public alert: <https://www.earthdata.nasa.gov/data/alerts-outages/nisar-l-band-data-now-publicly-available>
- GCOV provisional catalog: <https://www.earthdata.nasa.gov/data/catalog/asf-nisar-l2-gcov-provisional-v1-1>
- GCOV beta catalog: <https://www.earthdata.nasa.gov/data/catalog/asf-nisar-l2-gcov-beta-v1-1>
- NISAR observation strategy: <https://science.nasa.gov/mission/nisar/observation-strategy/>
- NISAR Mozambique flooding visualisation (Jan 2026): <https://svs.gsfc.nasa.gov/5661>
- ARSET 2026 NISAR training: <https://www.earthdata.nasa.gov/learn/trainings/harnessing-nisar-next-generation-radar-observations-earth-applications>
- Work with NISAR sample data: <https://www.earthdata.nasa.gov/learn/tutorials/work-nisar-sample-data>

**OPERA**

- RTC-S1 catalog: <https://www.earthdata.nasa.gov/data/catalog/asf-opera-l2-rtc-s1-v1-1>
- RTC-S1 product guide: <https://hyp3-docs.asf.alaska.edu/guides/opera_rtc_product_guide/>
- DSWx-S1: <https://podaac.jpl.nasa.gov/dataset/OPERA_L3_DSWX-S1_V1>
- DSWx-S1 product specification: <https://d2pn8kiwq2w21t.cloudfront.net/documents/OPERA_DSWx-S1_Product_Spec_v1.0.0_D-108761_RevA_2024-08-16_.pdf>
- DSWx-S1 release notice: <https://podaac.jpl.nasa.gov/announcements/2024-10-25-OPERA-Dynamic-Surface-Water-Extent-from-Sentinel-1-v1-Dataset-Release>
- DSWx suite: <https://www.jpl.nasa.gov/go/opera/products/dswx-product-suite/>
- DSWx-NI status slide: <https://d2pn8kiwq2w21t.cloudfront.net/documents/5th_OPERA_workshop_DSWx-NI_update.pdf>
- DIST-ALERT-S1: <https://www.earthdata.nasa.gov/data/catalog/asf-opera-l3-dist-alert-s1-v1-1>
- DISP-S1: <https://www.earthdata.nasa.gov/data/catalog/asf-opera-l3-disp-s1-v1-1>

**ARIA, HyP3 and Disasters**

- ARIA-S1-GUNW: <https://hyp3-docs.asf.alaska.edu/guides/gunw_product_guide/>
- ARIA damage proxy maps: <https://appliedsciences.nasa.gov/our-impact/news/aria-damage-proxy-map-using-esa-sentinel-1>
- ARIA flood extent example: <https://disasters.nasa.gov/our-impact/news/aria-flood-extent-map-harvey-sentinel-1-sar-data>
- HyP3: <https://hyp3-docs.asf.alaska.edu/>
- HyP3 credits: <https://hyp3-docs.asf.alaska.edu/using/credits/>
- NASA Disasters activations:
  - <https://disasters.nasa.gov/what-we-do/disasters/disasters-activations/alaska-flooding-october-2025>
  - <https://disasters.nasa.gov/what-we-do/disasters/disasters-activations/washington-atmospheric-rivers-2025>
  - <https://appliedsciences.nasa.gov/what-we-do/disasters/disasters-activations/hawaii-floods-march-2026>

**Sentinel-1 (partner)**

- Sentinel-1D at ASF: <https://www.earthdata.nasa.gov/data/alerts-outages/sentinel-1d-data-available-download-asf-daac>
- Sentinel-1D user data opening: <https://sentinels.copernicus.eu/-/sentinel-1d-user-data-opening-from-17-april-2026>

**Supporting data**

- GLO-30 HAND: <https://registry.opendata.aws/glo-30-hand/>
- NASA landslide data: <https://gpm.nasa.gov/landslides/data.html>
- ITS_LIVE: <https://its-live.jpl.nasa.gov/>

### 19.2 Assumptions (check each one)

1. The event dates are **14–15 November 2026** (per several announcements; confirm your local event).
2. The NISAR challenge wording above matches the official 2026 challenge page. The title was not visible to us.
3. The NISAR GCOV HDF5 layout is as documented: `/science/LSAR/GCOV/grids/frequencyA/{HHHH,HVHV,xCoordinates,yCoordinates,projection}`, with `epsg_code` on `projection` and **pixel-centre** coordinates. The reader is tested on a mock built from this description only.
4. GCOV values are **linear power γ⁰** with NaN (or `_FillValue`) for no-data.
5. The DSWx-S1 WTR codes (0/1/3/250/251/255) follow product spec v1.0.0 (D-108761 Rev A, 2024-08-16).
6. Short names: `NISAR_L2_GCOV_PROVISIONAL_V1`, `OPERA_L2_RTC-S1_V1`, `OPERA_L3_DSWX-S1_V1`. Confirm in Earthdata Search.
7. The default thresholds (Δ ≤ −3 dB, HAND ≤ 15 m, 2 ha, water mean ≤ −15 dB, fallback −18 dB) are common-practice starting values, **not** tuned for NISAR L-band. Check them against your scene's histograms.
8. The South Asian monsoon floodplain is only a *candidate* NISAR case (based on NISAR's global land coverage). Actual acquisitions must be checked.
9. OPERA DIST-ALERT-S1 and DSWx-NI availability for specific dates is **unconfirmed**.
10. The Claude API is used only for the optional explanation rewrite (model `claude-opus-5`, low effort). The MVP works without it.

---

## 20. Closing statement: what this project can and cannot claim

- **It can genuinely detect:**
  - areas where calibrated radar backscatter dropped sharply *and* became water-like between
    two NISAR (or Sentinel-1/OPERA) acquisitions, in low-lying terrain larger than 2 ha;
  - in other words, *possible flood-related (open-water) change*, with an evidence score and
    explanation.
- **It cannot reliably detect:**
  - flooding under forest canopy or in dense cities;
  - water that came and went between acquisitions;
  - small or narrow floods;
  - flood depth or damage;
  - landslides, ground deformation, glacier change or ice hazards.

  It also cannot tell a flood from a newly flooded rice field or wet snow without context.
- **Scientifically established (not validated by us):** the physics (specular reflection of
  calm water) and the method family (log-ratio + per-scene statistical thresholding with
  terrain masking) are standard in SAR flood mapping. The input data (NISAR GCOV provisional,
  OPERA RTC-S1 and DSWx-S1) are calibrated or validated by NASA, within their stated maturity.
- **Validated by us:**
  - So far, only the *software*, on synthetic data (51 automated tests).
  - Case-level agreement with OPERA DSWx-S1 is measured during the hackathon and must be
    reported with its dates, masks and pixel counts.
- **Experimental:**
  - the confidence score (a heuristic until checked per case);
  - the forest-disturbance mode (fixed low confidence);
  - all default thresholds on L-band data;
  - the optional LLM explanation rewrite.
- **Truly NASA data:**
  - NISAR L-band products (NASA-ISRO mission; L-band radar and processing by NASA JPL;
    distributed by NASA ASF DAAC);
  - OPERA RTC-S1, DSWx-S1 and DIST products (produced by NASA JPL);
  - ARIA products; NASA FIRMS; NASA landslide catalog; ITS_LIVE.
- **Partner or supporting data:**
  - Copernicus Sentinel-1 (ESA/EU), which OPERA products are derived from;
  - the Copernicus GLO-30 DEM and the ASF HAND dataset derived from it;
  - ISRO's S-band (not used).
- **Is the recommended MVP buildable in 48 hours?** **Yes.**
  - The pipeline, dashboard and tests already run end to end on synthetic data.
  - The remaining work is data work: finding, downloading and checking one NISAR pair and one
    reference, then running, validating and presenting.
  - The main risk (no usable NISAR pair) has a decision gate at hour 6 and a tested fallback
    path (OPERA RTC-S1 through the same code).
