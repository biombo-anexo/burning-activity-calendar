# Burning Activity Calendar (MODIS + VIIRS)

**Live demo (stable):** https://burning-activity-calendar.onrender.com

**Live demo with NOAA-20 extension (after S-NPP):** https://burning-activity-calendar-noaa20.onrender.com

A web app that turns NASA FIRMS active-fire hotspots into a **burning calendar** (months × years) for any area you draw
on a map. MODIS (2000→) and VIIRS (2012→) are harmonized into **one consistent series**, so 26 years can be compared
without a jump when the sensor changes. It shows when the fire season is, how unusual each month is against the
historical median, the most active months and years, and lets you download the numbers.

**Data loaded:** South America plus North and Central America (everything north of 10° N), 2000-11-01 → 2026-11-15. VIIRS S-NPP stopped on 2026-11-02; the days from 3 to 15 Nov 2026 come from VIIRS NOAA-20, calibrated against S-NPP with a per-calendar-month ratio (held-out test 2023 → 2026-10: 4.4% monthly error over the whole area, bias +0.9%).
That is 92 million FIRMS detections, which become 26 million fire cell-days.
**Default view:** Colombia (pilot country). Draw any rectangle or polygon inside the dashed outline on the map, or pick a quick area.

## Install and run (step by step)

The app runs on your own computer: a small Python server reads the data and your web browser shows the interface.
You need about **10 minutes** and **400 MB** of free space the first time. After that, opening it takes seconds.

### 1. What you need

| Requirement | How to check | Where to get it |
|---|---|---|
| **Python 3.11 or newer** | `python --version` (Windows) or `python3 --version` (macOS/Linux) | <https://www.python.org/downloads/>. On Windows, tick **"Add python.exe to PATH"** in the installer. |
| **A web browser** | Chrome, Edge, Firefox or Safari | |
| **Internet connection** | Only for the first install and for the map background | |

Git is optional: you can download the project as a ZIP instead.

### 2. Get the project

**Option A, without Git:** on this page click the green **Code** button, then **Download ZIP**, and extract it.
You get a folder called `burning-activity-calendar-main`.

**Option B, with Git:**
```bash
git clone https://github.com/biombo-anexo/burning-activity-calendar.git
```
You get a folder called `burning-activity-calendar`.

### 3. Install (first time only)

Open a terminal **inside the project folder** (the one that contains `README.md` and `requirements.txt`):

- **Windows:** open the folder in File Explorer, click the address bar, type `powershell` and press Enter.
- **macOS:** right-click the folder in Finder, then **New Terminal at Folder**.
- **Linux:** right-click inside the folder, then **Open in Terminal**.

Then create a private Python environment for the project and install the libraries into it:

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The install downloads pandas, DuckDB, FastAPI and a few other libraries (about 2–5 minutes).
It ends with a line starting with `Successfully installed`.

### 4. Start the app (every time)

In a terminal inside the project folder:

**Windows (PowerShell)**
```powershell
.venv\Scripts\python -m uvicorn app.main:app --port 8000
```

**macOS / Linux**
```bash
.venv/bin/python -m uvicorn app.main:app --port 8000
```

Wait until it prints `Uvicorn running on http://127.0.0.1:8000`, then open **<http://127.0.0.1:8000>** in your browser.
The default view is Colombia; pick a **Quick area** or draw a rectangle or polygon on the map, then press **Analyze**.

**Keep the terminal open while you use the app**: it is the server. To stop it, press **Ctrl+C** in the terminal (or close it).

### 5. Run the tests (optional)

```powershell
.venv\Scripts\python -m pip install pytest httpx
.venv\Scripts\python -m pytest -q
```
(macOS/Linux: replace `.venv\Scripts\python` with `.venv/bin/python`.) All 52 tests should pass.

### Troubleshooting

| Problem | Cause and fix |
|---|---|
| `python` is not recognized, or opens the Microsoft Store | Python is not installed or not on PATH. Reinstall it from python.org with **"Add python.exe to PATH"** ticked, then open a new terminal. |
| `Could not open requirements file` | The terminal is not inside the project folder. Use `cd` to go to the folder that contains `requirements.txt`. |
| `Failed to fetch` or an empty page | The page was opened by double-clicking `index.html`. Start the server (step 4) and open <http://127.0.0.1:8000> instead. |
| `address already in use` / port 8000 busy | The app is already running in another terminal: just open the link. Or start it on another port with `--port 8001` and open <http://127.0.0.1:8001>. |
| The map background is grey | The map tiles come from the internet. Check your connection; the analysis itself works offline. |
| Changes do not appear after an update | Refresh the page with **Ctrl+F5** (macOS: Cmd+Shift+R). |

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
- **S-NPP ended 2026-11-02; 3–15 Nov 2026 use calibrated NOAA-20.** NOAA-20 is converted to S-NPP-equivalent values with a per-calendar-month ratio measured on the 2018-04 → 2026-10 overlap (train 2018–2022, test 2023 → 2026-10): 4.4% monthly error over the whole area (bias +0.9%), versus 58% for raw MODIS. Per 1° block the error is about 23% per month, so small areas after 2 Nov 2026 are less certain.
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
