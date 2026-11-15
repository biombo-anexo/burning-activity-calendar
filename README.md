# Burning Activity Calendar (MODIS + VIIRS)

A web app that turns NASA FIRMS active-fire hotspots into a **burning calendar** (months × years) for any area you draw
on a map. MODIS (2000→) and VIIRS (2012→) are harmonized into **one consistent series**, so 26 years can be compared
without a jump when the sensor changes. It shows when the fire season is, how unusual each month is against the
historical median, the most active months and years, and lets you download the numbers.

**Data loaded:** South America plus North and Central America (everything north of 10° N), 2000-11-01 → 2026-11-02.
That is 92 million FIRMS detections, which become 26 million fire cell-days.
**Default view:** Colombia (pilot country). Draw any rectangle or polygon inside the dashed outline on the map, or pick a quick area.

## Run it

```bash
pip install -r requirements.txt
BURNCAL_DATA=data/processed python -m uvicorn app.main:app --port 8000   # then open http://127.0.0.1:8000
pip install pytest && python -m pytest -q                                  # 52 tests
```
On Windows PowerShell: `$env:BURNCAL_DATA="data\processed"; python -m uvicorn app.main:app --port 8000`.
Open the page through the server, not by double-clicking `index.html`.

## Method

| Step | What | Why |
|---|---|---|
| Clean | Keep FIRMS `type = 0` (presumed vegetation fire). MODIS confidence must be ≥ 30; VIIRS must be `nominal`/`high`. Detections within ±0.01° of known static sources are dropped, also in near-real-time files. | Removes gas flares, volcanoes, offshore and low-confidence detections. |
| Unit | A **fire cell-day** is a 0.1° (~11 km) cell that has at least 1 detection on a given day. | VIIRS pixels are 375 m and MODIS pixels 1 km, so raw pixel counts are not comparable. |
| Calibrate | On the overlap period (2012-02 →), the number of MODIS cell-days in each 1° block-month is related to its VIIRS cell-days. Three models are compared (ratio, power law, seasonal power law). The winner is chosen on **held-out years**: train ≤ 2018, test 2019–2026. | The accuracy claim is measured on data the model never saw. |
| Regional correction | The MODIS→VIIRS ratio differs by region. In boreal forests VIIRS adds about 1.3–1.8× over MODIS; in tropical farmland about 2.2–2.5×. Each 1° block gets a factor, which is shrunk towards the factor of its 5° parent region, which in turn is shrunk towards 1. The correction is used only if it lowers the held-out error by ≥ 5%. | A single curve over- or under-converts whole regions. With the parent level, sparse blocks borrow their region's factor. |
| Terra-only fix | Before July 2002 only Terra flew. Those counts are multiplied by the Terra+Aqua / Terra ratio for the same calendar month, measured 2002-07 → 2011 (2.2–2.7). | Without it the first 20 months would be far too low. |
| Apply | MODIS before 2012-02 is converted to VIIRS-equivalent weights; VIIRS is used as-is from 2012-02. | One continuous series. |
| Quality flags | Every month is flagged: `ok`, `terra_only`, `gaps` (suspected satellite outage), `provisional` (near-real-time data) or `partial`. | Shown in the calendar; flagged months are not used for anomalies. |

## Validation (real data, held-out years 2019–2026)

| | Monthly error over the whole area (nMAE) | Yearly error per 1° block |
|---|---|---|
| Raw MODIS, no harmonization | 59.8% (bias −60%) | – |
| Power-law model (selected) | 8.3% | 27.6% |
| + regional correction (single level) | – | 16.6% |
| **+ regional correction (with 5° parent level, used)** | **7.7%** (bias +4%) | **16.2%** |

**Independent check.** Raw MODIS covers both eras with the same sensor, so it shows the true change in activity between
2003–2011 and 2013–2025. The harmonized series should show the same change.

| Region (quick area) | Change, raw MODIS | Change, harmonized series |
|---|---|---|
| Colombia | 0.95 | 0.93 |
| Whole South America | 0.79 | 0.78 |
| Brazilian Amazon | 0.78 | 0.76 |
| Cerrado | 0.75 | 0.73 |
| Gran Chaco | 0.78 | 0.75 |
| Mexico | 0.87 | 0.87 |
| Central America | 0.88 | 0.86 |
| Southeastern US | 0.93 | 0.91 |
| Alaska | 0.60 | 0.60 |
| Bolivian lowlands | 0.67 | 0.62 (−7%) |
| Canadian boreal forest | 1.60 | 1.49 (−7%) |
| Western United States | 1.00 | 1.16 (+16%) |

Agreement is within ±3% for 13 of 16 regions tested. The three outliers are listed under limits.

**Consistency check.** The same Colombia border polygon was run on an earlier Colombia-only download and on this
continental dataset. It gives 1,150,816 and 1,149,492 cell-days: a 0.1% difference.

## Limits

- **Years 2000–2011 are estimates** (converted MODIS), not observations. The app marks them with a dashed border and "(est.)".
- **The record ends 2026-11-02.** VIIRS S-NPP stopped delivering data that day. Continuing the series needs NOAA-20/21
  calibrated against S-NPP (not implemented).
- **Remaining regional bias:** Western US +16%, Bolivian lowlands −7%, Canadian boreal −7% in the independent check. The
  regional factors are constant in time and work at 1° blocks, so much smaller areas can still be biased.
- **Cell-days are not burned area** or emissions. They count days with detected fire per ~11 km cell.
- **Coverage:** South America and the Americas north of 10° N only. The Colombian quick areas follow the national border
  (Natural Earth 1:50m); the other quick areas are approximate rectangles.

## Data and sources

- NASA FIRMS active fire archive and near-real-time data: MODIS Collection 6.1 (Terra and Aqua) and VIIRS S-NPP
  Collection 2 (375 m), from <https://firms.modaps.eosdis.nasa.gov/download/>.
- Country border: Natural Earth 1:50m (public domain), through the `world-atlas` package.
- Map tiles: OpenStreetMap contributors and Esri. Map library: Leaflet and Leaflet.draw (bundled in `app/static/vendor`).

## Use of AI

In accordance with NASA Space Apps Challenge guidelines, our team utilized **Claude (Anthropic)** as a supporting tool to accelerate specific implementation tasks, while all core engineering, design, and scientific decisions were driven by the team:

* **Team-Led (Core Development & Direction):** Challenge formulation, pilot region scoping, dataset curation and preprocessing logic, system architecture, harmonization methodology design, UI/UX decisions, integration, debugging, and final validation of all code and scientific outputs.
* **AI-Assisted (Supporting Tasks):** Drafting boilerplate code modules (initial data ingestion templates, web interface components, and unit test structures), assisting with validation script execution, and refining the structure of this README.

## Layout

```
app/main.py                 FastAPI backend (API + static files)
app/static/index.html       the whole interface (map, calendar, charts, rankings, downloads)
burncal/ingest.py           read and clean FIRMS files (pandas, small areas)
burncal/fastingest.py       same filters with DuckDB, out-of-core, for continental downloads
burncal/harmonize.py        MODIS→VIIRS calibration, regional factors, outage detection
burncal/pipeline.py         CLI: raw FIRMS files -> data/processed
burncal/analytics.py        queries behind the API (calendar, anomalies, rankings, map density)
burncal/shard.py            split the data into parts < 25 MB for GitHub
data/processed/             calibration.json + harmonized_parts/*.parquet (what the app reads)
tests/                      pytest suite (synthetic data with known truth + regression tests)
```

Rebuild from FIRMS CSV files (MODIS archive + NRT, VIIRS S-NPP archive + NRT, same area):
```bash
python -m burncal.pipeline --engine duckdb --modis <modis csv files> --viirs <viirs csv files> --out data/processed
python -m burncal.shard data/processed --remove-original
```
