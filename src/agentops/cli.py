"""``agentops`` CLI: evalset | run | gate | continuity | report."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from . import data as data_mod
from . import evalset
from .agent import HFBackend, RuleBackend, Trace, execute
from .ops import continuity_table, regression_gate, summary
from .report import write_report


def load_traces(path: Path) -> list[Trace]:
    return [Trace(**json.loads(x)) for x in path.read_text(encoding="utf-8").splitlines() if x]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="agentops", description=__doc__)
    p.add_argument("--weather", type=Path, default=Path("data/weather_2024.csv"))
    p.add_argument("--cases", type=Path, default=Path("data/evalset.jsonl"))
    p.add_argument("--reports", type=Path, default=Path("reports"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="re-download the Open-Meteo snapshot")
    sub.add_parser("evalset", help="(re)build the golden test set from the snapshot")
    r = sub.add_parser("run", help="run one backend on every case, write traces + summary")
    r.add_argument("backend", help="'rules' or a Hugging Face chat model id")
    g = sub.add_parser("gate", help="non-regression gate between two recorded runs")
    g.add_argument("baseline")
    g.add_argument("candidate")
    c = sub.add_parser("continuity", help="fault-injection replay of recorded runs")
    c.add_argument("--primary", required=True)
    c.add_argument("--fallback", required=True)
    c.add_argument("--degraded", default="rules")
    sub.add_parser("report", help="rebuild reports/results.md")
    a = p.parse_args(argv)
    a.reports.mkdir(parents=True, exist_ok=True)

    if a.cmd == "fetch":
        print(data_mod.fetch(a.weather))
        return
    if a.cmd == "report":
        print(write_report(a.reports))
        return
    weather = data_mod.load(a.weather)
    if a.cmd == "evalset":
        cases = evalset.build(weather)
        evalset.save(cases, a.cases)
        print(f"{len(cases)} cases -> {a.cases}")
        return
    cases = evalset.load(a.cases)
    if a.cmd == "run":
        backend = RuleBackend() if a.backend == "rules" else HFBackend(a.backend)
        traces = []
        for case in cases:
            tr = execute(weather, case.cid, backend.name, backend(case.question))
            traces.append(tr)
            print(f"{case.cid:24s} {tr.status:12s} {tr.tool} {tr.latency_s:.2f}s")
        path = a.reports / f"traces_{backend.name}.jsonl"
        path.write_text(
            "\n".join(json.dumps(asdict(t), ensure_ascii=False) for t in traces), encoding="utf-8"
        )
        res = summary(cases, traces)
        (a.reports / f"summary_{backend.name}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
    elif a.cmd == "gate":
        base = load_traces(a.reports / f"traces_{a.baseline}.jsonl")
        cand = load_traces(a.reports / f"traces_{a.candidate}.jsonl")
        res = regression_gate(cases, base, cand)
        res.update(baseline_name=a.baseline, candidate_name=a.candidate)
        (a.reports / f"gate_{a.baseline}_to_{a.candidate}.json").write_text(
            json.dumps(res, indent=1)
        )
        print(json.dumps(res, indent=1))
    elif a.cmd == "continuity":
        runs = {
            n: load_traces(a.reports / f"traces_{n}.jsonl")
            for n in {a.primary, a.fallback, a.degraded}
        }
        rows = continuity_table(cases, runs, a.primary, a.fallback, a.degraded)
        out = {"primary": a.primary, "fallback": a.fallback, "degraded": a.degraded, "rows": rows}
        (a.reports / "continuity.json").write_text(json.dumps(out, indent=1))
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
