"""
Build the IDL chi-square CSVs from the IDL .sav outputs.

Reads:
  data/IDL_synthetic_dem_data/idl_synthetic_dem_idx{N}_allfilters_base.sav
      -> chisq is a scalar
  data/IDL_monte_carlo_synthetic_dem_data/idl_synthetic_dem_idx{N}_allfilters_MC1000.sav
      -> chisq has shape (1001,): index 0 = un-perturbed fit, 1..1000 = MC runs

Writes:
  data/idl_base_chisq.csv      columns: dem_id, chisq
  data/idl_mc_chisq_long.csv   columns: dem_id, mc_run, chisq   (all runs, 0..1000)

All CSV logic lives in Python on purpose, even though the data comes from IDL.
"""

import csv
import re
from pathlib import Path

from scipy.io import readsav

BASE_DIR = Path(__file__).parent
IDL_BASE_DIR = BASE_DIR / "data" / "IDL_synthetic_dem_data"
IDL_MC_DIR = BASE_DIR / "data" / "IDL_monte_carlo_synthetic_dem_data"
OUT_BASE_CSV = BASE_DIR / "data" / "idl_base_chisq.csv"
OUT_MC_CSV = BASE_DIR / "data" / "idl_mc_chisq_long.csv"

IDX_RE = re.compile(r"_idx(\d+)_")


def dem_id_from_name(path):
    match = IDX_RE.search(path.name)
    if match is None:
        raise ValueError(f"No DEM index found in file name: {path.name}")
    return int(match.group(1))


def sorted_by_dem_id(folder, pattern):
    return sorted(folder.glob(pattern), key=dem_id_from_name)


def write_base_csv():
    files = sorted_by_dem_id(IDL_BASE_DIR, "*_base.sav")
    with OUT_BASE_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dem_id", "chisq"])
        for path in files:
            chisq = float(readsav(path)["chisq"])
            writer.writerow([dem_id_from_name(path), repr(chisq)])
    print(f"Wrote {len(files)} rows -> {OUT_BASE_CSV.relative_to(BASE_DIR)}")


def write_mc_csv():
    files = sorted_by_dem_id(IDL_MC_DIR, "*_MC1000.sav")
    n_rows = 0
    with OUT_MC_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dem_id", "mc_run", "chisq"])
        for path in files:
            dem_id = dem_id_from_name(path)
            for mc_run, chisq in enumerate(readsav(path)["chisq"]):
                writer.writerow([dem_id, mc_run, repr(float(chisq))])
                n_rows += 1
    print(
        f"Wrote {n_rows} rows from {len(files)} files "
        f"-> {OUT_MC_CSV.relative_to(BASE_DIR)}"
    )


if __name__ == "__main__":
    write_base_csv()
    write_mc_csv()
