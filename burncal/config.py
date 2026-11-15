"""Central configuration. Every tunable lives here so the method is auditable."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Grid used to turn raw pixels into "fire cell-days".
    cell_deg: float = 0.1        # ~11 km cells: one cell counts once per day if >=1 detection
    block_deg: float = 1.0       # calibration blocks (must be a multiple of cell_deg)

    # Sensor timeline (FIRMS archive availability).
    viirs_first_full_month: str = "2012-02-01"   # S-NPP starts 2012-01-20, so Jan 2012 is partial
    terra_only_until: str = "2002-07-04"          # Aqua starts; before this MODIS = Terra only

    # Quality filters.
    modis_min_conf: int = 30          # MODIS confidence 0-100; <30 = low
    viirs_ok_conf: tuple = ("n", "nominal", "h", "high")
    keep_types: tuple = (0,)          # 0 = presumed vegetation fire (drops volcanoes, flares, offshore)

    # Validation: calibrate on overlap years <= train_end_year, test on later years.
    train_end_year: int = 2018

    # Months at the end of the record that may still be near-real-time data.
    provisional_months: int = 5

    # Minimum number of baseline years needed to report an anomaly.
    min_baseline_years: int = 5

    @property
    def block_cells(self) -> int:
        n = round(self.block_deg / self.cell_deg)
        assert abs(n * self.cell_deg - self.block_deg) < 1e-9, "block_deg must be a multiple of cell_deg"
        return n


CFG = Config()
