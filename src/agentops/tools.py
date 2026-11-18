"""Agronomic tools the agent can call, with strict argument validation.

Every tool takes a city and an inclusive ISO date range. Validation happens *before*
execution: an unknown tool, an unknown city, a malformed date or an inverted range is rejected
with an explicit error, which the executor records instead of crashing or guessing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

Weather = dict[str, dict[str, dict[str, float]]]


class ToolError(ValueError):
    """Raised for invalid tool names or arguments (a guardrail, not a crash)."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    extra: dict  # extra parameter -> default


TOOLS = {
    "rainfall_total": ToolSpec("rainfall_total", "Total precipitation in mm over the period.", {}),
    "frost_days": ToolSpec(
        "frost_days", "Number of days with minimum temperature below 0 °C over the period.", {}
    ),
    "growing_degree_days": ToolSpec(
        "growing_degree_days",
        "Sum of growing degree days, base 10 °C: sum(max(0, (tmax+tmin)/2 - 10)).",
        {},
    ),
    "water_balance": ToolSpec(
        "water_balance",
        "Climatic water balance in mm: precipitation minus reference evapotranspiration (ET0).",
        {},
    ),
    "hottest_day": ToolSpec("hottest_day", "Date and value of the highest daily maximum.", {}),
}
# tools that exist in the platform but must never be called by this read-only agent
FORBIDDEN = {"send_sms_alert", "delete_records", "update_irrigation_schedule"}


def _parse(d: str) -> date:
    try:
        return date.fromisoformat(str(d))
    except ValueError as e:
        raise ToolError(f"invalid date {d!r}") from e


def _days(weather: Weather, city: str, start: str, end: str) -> list[dict[str, float]]:
    if city not in weather:
        raise ToolError(f"unknown city {city!r}")
    a, b = _parse(start), _parse(end)
    if b < a:
        raise ToolError("end date before start date")
    out, d = [], a
    while d <= b:
        key = d.isoformat()
        if key not in weather[city]:
            raise ToolError(f"no data for {city} on {key}")
        out.append(weather[city][key])
        d += timedelta(days=1)
    return out


def run_tool(weather: Weather, name: str, args: dict) -> float | str:
    if name in FORBIDDEN:
        raise ToolError(f"tool {name!r} is not allowed for this agent")
    if name not in TOOLS:
        raise ToolError(f"unknown tool {name!r}")
    try:
        city, start, end = args["city"], args["start"], args["end"]
    except (KeyError, TypeError) as e:
        raise ToolError(f"missing argument: {e}") from e
    days = _days(weather, city, start, end)
    if name == "rainfall_total":
        return round(math.fsum(x["precip"] for x in days), 1)
    if name == "frost_days":
        return sum(1 for x in days if x["tmin"] < 0)
    if name == "growing_degree_days":
        return round(math.fsum(max(0.0, (x["tmax"] + x["tmin"]) / 2 - 10) for x in days), 1)
    if name == "water_balance":
        return round(math.fsum(x["precip"] - x["et0"] for x in days), 1)
    # hottest_day
    a = _parse(start)
    best = max(range(len(days)), key=lambda i: days[i]["tmax"])
    return f"{(a + timedelta(days=best)).isoformat()} ({days[best]['tmax']} °C)"


def tool_catalog() -> str:
    lines = [f"- {t.name}(city, start, end): {t.description}" for t in TOOLS.values()]
    return "\n".join(lines)
