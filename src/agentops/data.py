"""Daily weather for ten French farming towns (Open-Meteo historical archive, ERA5-based).

The snapshot (2024-01-01 .. 2024-12-31) is committed in ``data/weather_2024.csv`` so tests,
CI and every experiment run on identical numbers. ``agentops fetch`` re-downloads it.
Open-Meteo data is licensed CC BY 4.0.
"""

from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

CITIES = {
    "Beauvais": (49.43, 2.08),
    "Amiens": (49.89, 2.30),
    "Reims": (49.26, 4.03),
    "Chartres": (48.45, 1.49),
    "Rennes": (48.11, -1.68),
    "Poitiers": (46.58, 0.34),
    "Dijon": (47.32, 5.04),
    "Valence": (44.93, 4.89),
    "Toulouse": (43.60, 1.44),
    "Montpellier": (43.61, 3.88),
}
FIELDS = ("tmax", "tmin", "precip", "et0")
API = "https://archive-api.open-meteo.com/v1/archive"


def fetch(out: Path, start: str = "2024-01-01", end: str = "2024-12-31") -> Path:
    rows = []
    for city, (lat, lon) in CITIES.items():
        q = urllib.parse.urlencode(
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": start,
                "end_date": end,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                "et0_fao_evapotranspiration",
                "timezone": "Europe/Paris",
            }
        )
        req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": "agentops/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)["daily"]
        for i, day in enumerate(d["time"]):
            rows.append(
                [
                    city,
                    day,
                    d["temperature_2m_max"][i],
                    d["temperature_2m_min"][i],
                    d["precipitation_sum"][i],
                    d["et0_fao_evapotranspiration"][i],
                ]
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["city", "date", *FIELDS])
        w.writerows(rows)
    return out


def load(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """``{city: {date: {tmax, tmin, precip, et0}}}``"""
    out: dict[str, dict[str, dict[str, float]]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["city"], {})[row["date"]] = {k: float(row[k]) for k in FIELDS}
    return out
