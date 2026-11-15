"""MODIS -> VIIRS harmonization.

Idea
----
MODIS (1 km) and VIIRS (375 m) cannot be compared by raw pixel counts. We (1) collapse both to
*fire cell-days* (a 0.1 deg cell counts once per day if it has >=1 detection), (2) learn on the
period where both sensors fly (2012-02 onward) how many VIIRS cell-days correspond to a given
number of MODIS cell-days in a 1 deg block-month, and (3) apply that mapping to MODIS-only
history (2000-11 .. 2012-01) so the whole series is expressed in VIIRS-equivalent units.

Three candidate mappings are fitted (ratio, power law, power law with a smooth seasonal term).
The one with the lowest error on *held-out* overlap years is kept; the seasonal model is only
chosen if it beats the best simple model by a clear margin (anti-overfitting rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG, Config

MIN_GAP_DAYS_FLAG = 2     # a month with this many suspected outage days is flagged 'gaps'


# ----------------------------------------------------------------------------- helpers
def month_index(dates: pd.Series) -> np.ndarray:
    return (dates.dt.year.values * 12 + dates.dt.month.values - 1).astype("int32")


def ym_to_year(ym):
    return np.asarray(ym) // 12


def ym_to_month(ym):
    return np.asarray(ym) % 12 + 1


def block_monthly(raw: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """Cell-day counts per (1 deg block, month) and sensor -> columns bi, bj, ym, MODIS, VIIRS."""
    bs = cfg.block_cells
    d = pd.DataFrame(
        {
            "bi": raw["cell_i"].values // bs,
            "bj": raw["cell_j"].values // bs,
            "ym": month_index(raw["date"]),
            "sensor": raw["sensor"].values,
        }
    )
    t = d.groupby(["bi", "bj", "ym", "sensor"]).size().unstack("sensor", fill_value=0)
    for s in ("MODIS", "VIIRS"):
        if s not in t.columns:
            t[s] = 0
    return t[["MODIS", "VIIRS"]].reset_index()


# ----------------------------------------------------------------------------- outage detection
def suspected_gap_days(raw: pd.DataFrame, sensor: str, start, end, window: int = 30) -> pd.DatetimeIndex:
    """Days with zero detections that look like a satellite/data outage rather than a fire-free day.

    A run of zero days is suspicious when the surrounding +/-`window` days (excluding the run) are
    clearly active: median >= 8 cell-days/day for runs of 2+ days, >= 30 for a single day. Heuristic:
    genuine fire-free days happen in the wet season, which is why quiet neighbours do not qualify.
    """
    sub = raw.loc[raw["sensor"] == sensor, "date"]
    daily = sub.groupby(sub.dt.normalize()).size().reindex(pd.date_range(start, end), fill_value=0)
    zero = daily.index[daily.values == 0]
    if len(zero) == 0:
        return pd.DatetimeIndex([])
    runs = pd.Series(zero).diff().dt.days.ne(1).cumsum().values
    out = []
    for r in np.unique(runs):
        days = zero[runs == r]
        lo, hi = days[0] - pd.Timedelta(days=window), days[-1] + pd.Timedelta(days=window)
        w = daily[(daily.index >= lo) & (daily.index <= hi) & ~daily.index.isin(days)]
        med = float(w.median()) if len(w) else 0.0
        if (len(days) >= 2 and med >= 8) or (len(days) == 1 and med >= 30):
            out.extend(days)
    return pd.DatetimeIndex(out)


def gap_month_counts(days: pd.DatetimeIndex) -> dict:
    if len(days) == 0:
        return {}
    s = pd.Series(1, index=days).groupby([days.year, days.month]).sum()
    return {f"{y}-{m:02d}": int(v) for (y, m), v in s.items()}


# ----------------------------------------------------------------------------- models
# All models share: fit(M, V, month) and predict(M, month). `month` is the calendar month (1-12)
# of each block-month; non-seasonal models ignore it.
class RatioModel:
    name = "ratio"
    n_params = 1

    def fit(self, M, V, month=None):
        # Uses ALL block-months, including those where only VIIRS saw fire, so the total is preserved.
        self.r = float(V.sum() / max(M.sum(), 1))
        return self

    def predict(self, M, month=None):
        return self.r * np.asarray(M, dtype="float64")

    def params(self):
        return {"r": self.r}


class PowerModel:
    """log(1+V) = a + b*log(1+M), with smearing correction and a total-preserving rescale."""

    name = "power"
    n_params = 3

    def fit(self, M, V, month=None):
        pos = M > 0
        x, y = np.log1p(M[pos]), np.log1p(V[pos])
        self.b, self.a = np.polyfit(x, y, 1)
        resid = y - (self.a + self.b * x)
        self.smear = float(np.mean(np.exp(resid)))
        # VIIRS-only block-months cannot be recovered from MODIS; rescale so totals match on train.
        self.scale = 1.0
        self.scale = float(V.sum() / max(self._raw(M, month).sum(), 1e-9))
        return self

    def _raw(self, M, month=None):
        M = np.asarray(M, dtype="float64")
        out = np.exp(self.a + self.b * np.log1p(M)) * self.smear - 1.0
        return np.where(M > 0, np.maximum(out, 0.0), 0.0)

    def predict(self, M, month=None):
        return self.scale * self._raw(M, month)

    def params(self):
        return {"a": float(self.a), "b": float(self.b), "smear": self.smear, "scale": self.scale}


class SeasonalPowerModel(PowerModel):
    """Power law whose intercept AND slope vary smoothly with calendar month (one annual harmonic).

    Why: MODIS misses small agricultural burns that VIIRS sees, and the fire-size mix changes with
    the season, so a single V-vs-M curve is biased in some months and too high in others.
    """

    name = "seasonal_power"
    n_params = 9

    @staticmethod
    def _design(x, month):
        th = 2 * np.pi * (np.asarray(month, dtype="float64") - 0.5) / 12.0
        c, s = np.cos(th), np.sin(th)
        return np.column_stack([np.ones_like(x), x, c, s, x * c, x * s])

    def fit(self, M, V, month=None):
        if month is None:
            raise ValueError("SeasonalPowerModel needs the calendar month")
        pos = M > 0
        x, y = np.log1p(M[pos]), np.log1p(V[pos])
        X = self._design(x, np.asarray(month)[pos])
        self.coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ self.coef
        self.smear = float(np.mean(np.exp(resid)))
        self.scale = 1.0
        self.scale = float(V.sum() / max(self._raw(M, month).sum(), 1e-9))
        return self

    def _raw(self, M, month=None):
        M = np.asarray(M, dtype="float64")
        X = self._design(np.log1p(M), month)
        out = np.exp(X @ self.coef) * self.smear - 1.0
        return np.where(M > 0, np.maximum(out, 0.0), 0.0)

    def params(self):
        return {"coef": [float(c) for c in self.coef], "smear": self.smear, "scale": self.scale}


MODELS = {"ratio": RatioModel, "power": PowerModel, "seasonal_power": SeasonalPowerModel}
SIMPLE = ("ratio", "power")
SEASONAL_MIN_GAIN = 0.10   # seasonal model must cut held-out nMAE by >= 10% vs best simple model


def model_from_params(name: str, params: dict):
    m = MODELS[name]()
    for k, v in params.items():
        setattr(m, k, np.asarray(v) if k == "coef" else v)
    return m


# ----------------------------------------------------------------------------- validation
def _metrics(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    mae = float(np.mean(np.abs(actual - pred)))
    mean_a = float(np.mean(actual)) if len(actual) else float("nan")
    r = float(np.corrcoef(actual, pred)[0, 1]) if len(actual) > 2 and pred.std() > 0 else float("nan")
    return {
        "n": int(len(actual)),
        "mae": mae,
        "nmae": mae / mean_a if mean_a else float("nan"),
        "r": r,
        "bias_pct": float((pred.sum() - actual.sum()) / actual.sum() * 100) if actual.sum() else float("nan"),
    }


def _region_month(bm: pd.DataFrame, pred_V: np.ndarray):
    t = pd.DataFrame({"ym": bm["ym"].values, "V": bm["VIIRS"].values, "P": pred_V})
    return t.groupby("ym").sum()


def overlap_table(bm: pd.DataFrame, last_complete_ym: int, cfg: Config = CFG, exclude_ym=()) -> pd.DataFrame:
    first = pd.Timestamp(cfg.viirs_first_full_month)
    first_ym = first.year * 12 + first.month - 1
    keep = (bm["ym"] >= first_ym) & (bm["ym"] <= last_complete_ym) & ~bm["ym"].isin(list(exclude_ym))
    return bm[keep].reset_index(drop=True)


REGIONAL_SHRINK = 300.0     # cell-days of "prior" pulling a block's factor towards 1 (small blocks stay near 1)
REGIONAL_CLIP = (0.4, 2.5)
REGIONAL_MIN_GAIN = 0.05    # regional factors are used only if they cut held-out block-year error by >= 5%


REGIONAL_PARENT = 5         # parent block size in calibration blocks (5 x 1 deg): sparse blocks borrow its factor


def regional_factors(ov: pd.DataFrame, pred: np.ndarray, k: float = REGIONAL_SHRINK,
                     parent: int | None = REGIONAL_PARENT) -> dict:
    """Per-block correction to the overall model: observed VIIRS / predicted, with shrinkage.

    MODIS and VIIRS do not relate the same way everywhere (large boreal crown fires vs. small tropical farm
    burns), so a single curve over- or under-converts whole regions. Two levels:
      parent (parent x parent blocks):  f_p = (V_p + k) / (P_p + k)            (shrunk towards 1)
      block:                            f   = (V + k * f_p) / (P + k)          (shrunk towards its parent)
    so a block with little data takes its region's factor instead of the global one. parent=None gives the
    single-level version (shrunk towards 1). The factors are rescaled to leave the overall total unchanged.
    """
    t = pd.DataFrame({"bi": ov["bi"].values, "bj": ov["bj"].values, "V": ov["VIIRS"].values.astype(float), "P": pred})
    g = t.groupby(["bi", "bj"]).sum()
    if parent:
        bi = g.index.get_level_values("bi").values
        bj = g.index.get_level_values("bj").values
        pk = pd.MultiIndex.from_arrays([bi // parent, bj // parent])
        gp = g.groupby(pk).sum()
        fp = ((gp["V"] + k) / (gp["P"] + k)).clip(*REGIONAL_CLIP)
        prior = fp.reindex(pk).values
    else:
        prior = 1.0
    f = ((g["V"] + k * prior) / (g["P"] + k)).clip(*REGIONAL_CLIP)
    scale = g["P"].sum() / max((g["P"] * f).sum(), 1e-9)
    return {(int(i), int(j)): float(v * scale) for (i, j), v in f.items()}


def apply_regional(bi, bj, factors: dict) -> np.ndarray:
    bi, bj = np.asarray(bi, dtype="int64"), np.asarray(bj, dtype="int64")
    if not factors:
        return np.ones(len(bi))
    keys = np.array([(int(i) + 100_000) * 1_000_000 + (int(j) + 100_000) for i, j in factors], dtype="int64")
    vals = np.array(list(factors.values()), dtype="float64")
    pos = pd.Index(keys).get_indexer((bi + 100_000) * 1_000_000 + (bj + 100_000))
    return np.where(pos >= 0, vals[np.maximum(pos, 0)], 1.0)


def _block_year(ov: pd.DataFrame, pred: np.ndarray):
    t = pd.DataFrame({"bi": ov["bi"].values, "bj": ov["bj"].values, "y": ym_to_year(ov["ym"].values),
                      "V": ov["VIIRS"].values.astype(float), "P": pred})
    return t.groupby(["bi", "bj", "y"]).sum()


def validate_and_fit(bm: pd.DataFrame, last_complete_ym: int, cfg: Config = CFG, exclude_ym=()):
    """Compare models on held-out years, pick the winner, refit it on the full overlap.

    exclude_ym: months with suspected satellite outages (either sensor); not used for fitting or testing.
    """
    ov = overlap_table(bm, last_complete_ym, cfg, exclude_ym)
    if ov.empty or ov["VIIRS"].sum() == 0 or ov["MODIS"].sum() == 0:
        raise ValueError("Need MODIS and VIIRS data over the overlap period (>= 2012-02) to calibrate.")
    years = ym_to_year(ov["ym"].values)
    train = ov[years <= cfg.train_end_year]
    test = ov[years > cfg.train_end_year]
    report = {"train_years": [int(years.min()), cfg.train_end_year], "models": {},
              "excluded_gap_months": int(len(set(exclude_ym)))}

    if len(train) == 0 or len(test) == 0:
        report["warning"] = "No held-out years available; validation skipped, ratio model used."
        chosen = "ratio"
        report["selected"] = chosen
    else:
        report["test_years"] = [int(ym_to_year(test["ym"].values).min()), int(ym_to_year(test["ym"].values).max())]
        Mtr, Vtr = train["MODIS"].values.astype(float), train["VIIRS"].values.astype(float)
        mtr = ym_to_month(train["ym"].values)
        Mte, mte = test["MODIS"].values.astype(float), ym_to_month(test["ym"].values)

        raw_rm = _region_month(test, Mte)
        report["no_harmonization"] = _metrics(raw_rm["V"], raw_rm["P"])  # MODIS used as-is
        for name, cls in MODELS.items():
            m = cls().fit(Mtr, Vtr, mtr)
            p = m.predict(Mte, mte)
            rm = _region_month(test, p)
            met = {"region_month": _metrics(rm["V"], rm["P"]), "params": m.params(), "n_params": cls.n_params}
            pos = test["MODIS"].values > 0
            met["block_month_log"] = _metrics(np.log1p(test["VIIRS"].values[pos]), np.log1p(p[pos]))
            cal = pd.DataFrame({"m": ym_to_month(rm.index), "V": rm["V"].values, "P": rm["P"].values}).groupby("m").sum()
            bias = [float((cal.loc[k, "P"] - cal.loc[k, "V"]) / cal.loc[k, "V"] * 100) if k in cal.index and cal.loc[k, "V"] else None
                    for k in range(1, 13)]
            met["bias_pct_by_calendar_month"] = bias
            met["max_abs_calendar_bias_pct"] = max((abs(b) for b in bias if b is not None), default=None)
            report["models"][name] = met

        nm = {k: v["region_month"]["nmae"] for k, v in report["models"].items()}
        best_simple = min(SIMPLE, key=lambda k: nm[k])
        chosen = best_simple
        if nm["seasonal_power"] < nm[best_simple] * (1 - SEASONAL_MIN_GAIN):
            chosen = "seasonal_power"
        report["selected"] = chosen
        report["selection_rule"] = (f"lowest held-out region-month nMAE; the seasonal model is chosen only if it is "
                                    f">= {int(SEASONAL_MIN_GAIN * 100)}% better than the best simple model")

    final = MODELS[chosen]().fit(ov["MODIS"].values.astype(float), ov["VIIRS"].values.astype(float),
                                 ym_to_month(ov["ym"].values))
    report["final_fit_months"] = int(len(ov["ym"].unique()))

    # Regional step: does a per-block factor learned on the training years help on the held-out years?
    report["regional"] = {"used": False}
    if len(train) and len(test):
        mt = MODELS[chosen]().fit(Mtr, Vtr, mtr)
        f_tr = regional_factors(train, mt.predict(Mtr, mtr))
        p0 = mt.predict(Mte, mte)
        p1 = p0 * apply_regional(test["bi"].values, test["bj"].values, f_tr)
        by0, by1 = _block_year(test, p0), _block_year(test, p1)
        e0, e1 = _metrics(by0["V"], by0["P"]), _metrics(by1["V"], by1["P"])
        rm1 = _region_month(test, p1)
        use = e1["nmae"] < e0["nmae"] * (1 - REGIONAL_MIN_GAIN)
        p_flat = p0 * apply_regional(test["bi"].values, test["bj"].values,
                                     regional_factors(train, mt.predict(Mtr, mtr), parent=None))
        by_flat = _block_year(test, p_flat)
        report["regional"] = {"used": bool(use), "shrink": REGIONAL_SHRINK, "parent_blocks": REGIONAL_PARENT,
                              "block_year_nmae_single_level": _metrics(by_flat["V"], by_flat["P"])["nmae"],
                              "block_year_nmae_without": e0["nmae"], "block_year_nmae_with": e1["nmae"],
                              "region_month_with": _metrics(rm1["V"], rm1["P"]),
                              "rule": f"used only if held-out block-year nMAE improves by >= {int(REGIONAL_MIN_GAIN * 100)}%"}
        if use:
            f = regional_factors(ov, final.predict(ov["MODIS"].values.astype(float), ym_to_month(ov["ym"].values)))
            report["regional"]["factors"] = {f"{i},{j}": v for (i, j), v in f.items()}
            vals = np.array(list(f.values()))
            report["regional"]["factor_range"] = [float(vals.min()), float(np.median(vals)), float(vals.max())]
    return final, report


# ----------------------------------------------------------------------------- apply
def harmonize(raw: pd.DataFrame, model, series_end: pd.Timestamp, cfg: Config = CFG,
              terra_factors: dict | float = 1.0, provisional_start=None, gap_months: dict | None = None,
              regional: dict | None = None) -> pd.DataFrame:
    """Return the harmonized cell-day table (VIIRS-equivalent weight `w` per row).

    terra_factors: {calendar month: Terra+Aqua / Terra cell-day ratio} (or one number), measured on
    2002-07..2011. Applied to block-month counts before Aqua existed, so the one-satellite months
    are brought to the two-satellite level the calibration was learned on.
    provisional_start: first date of near-real-time data (flagged 'provisional').
    gap_months: {'YYYY-MM': n suspected outage days} for the sensor that supplies each month.
    """
    first = pd.Timestamp(cfg.viirs_first_full_month)
    terra_cut = pd.Timestamp(cfg.terra_only_until)
    if provisional_start is None:
        provisional_start = (series_end.to_period("M") - (cfg.provisional_months - 1)).to_timestamp()
    provisional_start = pd.Timestamp(provisional_start)
    gap_months = gap_months or {}

    modis = raw[(raw["sensor"] == "MODIS") & (raw["date"] < first)].copy()
    viirs = raw[(raw["sensor"] == "VIIRS") & (raw["date"] >= first)].copy()

    bs = cfg.block_cells
    modis["bi"] = modis["cell_i"] // bs
    modis["bj"] = modis["cell_j"] // bs
    modis["ym"] = month_index(modis["date"])
    counts = modis.groupby(["bi", "bj", "ym"])["date"].transform("size").values.astype(float)
    mon = modis["date"].dt.month.values
    if isinstance(terra_factors, dict):
        default = float(np.mean(list(terra_factors.values()))) if terra_factors else 1.0
        lut = np.array([terra_factors.get(m, default) for m in range(13)], dtype="float64")
        fac = lut[mon]
    else:
        fac = np.full(len(modis), float(terra_factors))
    equiv = np.where((modis["date"] < terra_cut).values, counts * fac, counts)
    pred = model.predict(equiv, mon)
    if regional:                                   # {"bi,bj": factor} from validate_and_fit
        fmap = {tuple(int(x) for x in k.split(",")): v for k, v in regional.items()}
        pred = pred * apply_regional(modis["bi"].values, modis["bj"].values, fmap)
    modis["w"] = np.where(counts > 0, pred / np.maximum(counts, 1), 0.0)
    viirs["w"] = 1.0

    cols = ["date", "cell_i", "cell_j", "sensor", "n_det", "frp_sum", "w"]
    out = pd.concat([modis[cols], viirs[cols]], ignore_index=True)
    flag = np.full(len(out), "ok", dtype=object)
    flag[(out["sensor"] == "MODIS").values & (out["date"] < terra_cut).values] = "terra_only"
    flag[(out["date"] >= provisional_start).values] = "provisional"
    if gap_months:
        bad = [int(k[:4]) * 12 + int(k[5:7]) - 1 for k, v in gap_months.items() if v >= MIN_GAP_DAYS_FLAG]
        flag[np.isin(month_index(out["date"]), bad)] = "gaps"
    out["flag"] = flag
    out["w"] = out["w"].astype("float32")
    return out.sort_values(["date", "cell_i", "cell_j"]).reset_index(drop=True)
