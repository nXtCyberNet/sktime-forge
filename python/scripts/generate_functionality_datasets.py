from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_datasets"
RNG = random.Random(42)


def _date_sequence(start: str, periods: int, freq: str) -> list[str]:
    base = datetime.strptime(start, "%Y-%m-%d")
    values: list[str] = []

    if freq == "D":
        step = timedelta(days=1)
        for i in range(periods):
            values.append((base + i * step).strftime("%Y-%m-%d %H:%M:%S"))
        return values

    if freq == "H":
        step = timedelta(hours=1)
        for i in range(periods):
            values.append((base + i * step).strftime("%Y-%m-%d %H:%M:%S"))
        return values

    if freq == "MS":
        year = base.year
        month = base.month
        for _ in range(periods):
            values.append(f"{year:04d}-{month:02d}-01 00:00:00")
            month += 1
            if month > 12:
                month = 1
                year += 1
        return values

    raise ValueError(f"Unsupported frequency: {freq}")


def _write_series(name: str, values: list[float], freq: str, start: str) -> None:
    timestamps = _date_sequence(start=start, periods=len(values), freq=freq)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / name
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "y"])
        for ts, val in zip(timestamps, values):
            writer.writerow([ts, f"{val:.4f}"])


def _m4_monthly_subset_like() -> list[float]:
    values: list[float] = []
    for t in range(72):
        seasonal = 12.0 * math.sin(2.0 * math.pi * t / 12.0)
        trend = 220.0 + 0.9 * t
        noise = RNG.gauss(0.0, 1.5)
        values.append(trend + seasonal + noise)
    return values


def _m5_covid_period_like() -> list[float]:
    values: list[float] = []
    for t in range(120):
        baseline = 180.0 + 0.25 * t
        weekly = 8.0 * math.sin(2.0 * math.pi * t / 7.0)
        noise = RNG.gauss(0.0, 2.2)
        values.append(baseline + weekly + noise)
    # Simulate sharp pandemic disruption and gradual recovery.
    for idx in range(60, 85):
        values[idx] -= 60.0
    for idx in range(85, 110):
        values[idx] -= 25.0
    return values


def _etth1_like_hourly() -> list[float]:
    values: list[float] = []
    for t in range(24 * 21):
        daily = 10.0 * math.sin(2.0 * math.pi * t / 24.0)
        weekly = 4.0 * math.sin(2.0 * math.pi * t / (24.0 * 7.0))
        trend = 95.0 + 0.04 * t
        noise = RNG.gauss(0.0, 1.1)
        values.append(trend + daily + weekly + noise)
    return values


def _yahoo_s5_like_drift() -> list[float]:
    values: list[float] = []
    for t in range(180):
        base = 75.0 + 0.05 * t
        periodic = 5.0 * math.sin(2.0 * math.pi * t / 18.0)
        noise = RNG.gauss(0.0, 1.0)
        values.append(base + periodic + noise)
    for idx in range(110, 180):
        values[idx] += 18.0
    anomaly_adjustments = {35: 14.0, 88: -12.0, 142: 16.0, 167: -10.0}
    for idx, delta in anomaly_adjustments.items():
        values[idx] += delta
    return values


def _m5_promotional_events_like() -> list[float]:
    values: list[float] = []
    for t in range(160):
        base = 140.0 + 0.18 * t
        weekly = 6.0 * math.sin(2.0 * math.pi * t / 7.0)
        noise = RNG.gauss(0.0, 1.6)
        values.append(base + weekly + noise)
    promo_days = [20, 41, 62, 83, 104, 125, 146]
    for idx in promo_days:
        values[idx] += 35.0
    for idx in range(95, 160):
        values[idx] += 12.0
    return values


def main() -> None:
    _write_series("m4_monthly_subset_like.csv", _m4_monthly_subset_like(), freq="MS", start="2017-01-01")
    _write_series("m5_covid_period_like.csv", _m5_covid_period_like(), freq="D", start="2019-01-01")
    _write_series("etth1_like_hourly.csv", _etth1_like_hourly(), freq="H", start="2022-01-01")
    _write_series("yahoo_s5_like_drift.csv", _yahoo_s5_like_drift(), freq="H", start="2022-04-01")
    _write_series("m5_promotional_events_like.csv", _m5_promotional_events_like(), freq="D", start="2020-01-01")
    print(f"Generated scenario fixtures in {OUT_DIR}")


if __name__ == "__main__":
    main()
