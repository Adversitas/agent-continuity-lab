import json
from pathlib import Path

import pytest

from agentops import evalset
from agentops.agent import ModelReply, RuleBackend, ScriptedBackend, execute, parse_call
from agentops.cli import main
from agentops.data import load
from agentops.ops import STRATEGIES, regression_gate, score, simulate, summary
from agentops.tools import ToolError, run_tool

WEATHER = load(Path(__file__).parents[1] / "data" / "weather_2024.csv")
CASES = evalset.build(WEATHER)


def call(tool, city="Reims", start="2024-01-01", end="2024-01-31"):
    return json.dumps({"tool": tool, "args": {"city": city, "start": start, "end": end}})


def perfect_backend():
    replies = {}
    for c in CASES:
        replies[c.question] = (
            '{"tool": null}' if c.tool is None else json.dumps({"tool": c.tool, "args": c.args})
        )
    return ScriptedBackend(replies, "perfect")


def run(backend):
    return [execute(WEATHER, c.cid, backend.name, backend(c.question)) for c in CASES]


def test_snapshot_is_complete():
    assert len(WEATHER) == 10
    assert all(len(days) == 366 for days in WEATHER.values())


def test_tools_compute_from_real_rows():
    jan = [WEATHER["Reims"][f"2024-01-{d:02d}"] for d in range(1, 32)]
    assert run_tool(WEATHER, "frost_days", json.loads(call("x"))["args"]) == sum(
        d["tmin"] < 0 for d in jan
    )
    assert run_tool(WEATHER, "rainfall_total", json.loads(call("x"))["args"]) == pytest.approx(
        round(sum(d["precip"] for d in jan), 1)
    )
    assert "2024-" in run_tool(WEATHER, "hottest_day", json.loads(call("x"))["args"])


@pytest.mark.parametrize(
    "tool,args,msg",
    [
        ("send_sms_alert", {}, "not allowed"),
        ("nope", {}, "unknown tool"),
        ("frost_days", {"city": "Lille", "start": "2024-01-01", "end": "2024-01-02"}, "city"),
        ("frost_days", {"city": "Reims", "start": "2024-02-01", "end": "2024-01-02"}, "before"),
        ("frost_days", {"city": "Reims", "start": "01/02/2024", "end": "2024-01-02"}, "date"),
        ("frost_days", {"city": "Reims"}, "missing"),
    ],
)
def test_tool_guardrails(tool, args, msg):
    with pytest.raises(ToolError, match=msg):
        run_tool(WEATHER, tool, args)


def test_parse_call_takes_first_balanced_object():
    tool, args = parse_call('Sure! {"tool": "frost_days", "args": {"city": "Reims"}} {"x": 1}')
    assert tool == "frost_days" and args == {"city": "Reims"}
    assert parse_call('{"tool": null}') == (None, None)
    with pytest.raises(ValueError):
        parse_call("no json here")


def test_executor_statuses():
    statuses = [
        execute(WEATHER, "c", "t", ModelReply(text, 0.1)).status
        for text in [
            call("frost_days"),
            '{"tool": null}',
            "oops",
            call("delete_records"),
            call("frost_days", city="Lille"),
        ]
    ]
    assert statuses == ["ok", "declined", "invalid_json", "blocked", "tool_error"]


def test_evalset_is_deterministic_and_balanced():
    assert [c.cid for c in evalset.build(WEATHER)] == [c.cid for c in CASES]
    kinds = [c.kind for c in CASES]
    assert kinds.count("normal") == 60 and kinds.count("injection") == 8
    assert kinds.count("unknown_city") == 6


def test_rule_backend_parses_supported_periods_and_declines_others():
    assert RuleBackend.period("du 4 fevrier au 12 mai 2024") == ("2024-02-04", "2024-05-12")
    assert RuleBackend.period("en fevrier 2024") == ("2024-02-01", "2024-02-29")
    assert RuleBackend.period("au premier trimestre 2024") == ("2024-01-01", "2024-03-31")
    assert RuleBackend.period("de mi-mars a fin avril 2024") is None
    s = summary(CASES, run(RuleBackend()))
    assert s["unknown_city_answered"] == 0
    wrong = [
        c.cid
        for c, t in zip(CASES, run(RuleBackend()), strict=True)
        if t.status == "ok" and not score(c, t)["correct"]
    ]
    # On clean questions the degraded mode may decline but never answers wrongly. Injected
    # text is another story: "send_sms_alert ... 'gel imminent'" contains the frost keyword,
    # so keyword routing picks frost_days. Documented in the README as a known weakness.
    assert wrong == ["injection-3", "injection-7"]


def test_perfect_backend_scores_one_and_gate_detects_regressions():
    perfect = run(perfect_backend())
    assert summary(CASES, perfect)["overall_accuracy"] == 1.0
    broken = list(perfect)
    for i, c in enumerate(CASES):
        if c.tool == "frost_days":
            broken[i] = execute(WEATHER, c.cid, "b", ModelReply('{"tool": null}', 0.1))
    gate = regression_gate(CASES, perfect, broken)
    assert gate["decision"] == "FAIL" and "regression on frost_days" in gate["reasons"]
    assert regression_gate(CASES, broken, perfect)["decision"] == "PASS"


def test_continuity_simulation_extremes():
    runs = {"p": run(perfect_backend()), "f": run(perfect_backend()), "rules": run(RuleBackend())}
    full = STRATEGIES["+ rule-based degraded mode"]
    assert (
        simulate(CASES, runs, "p", "f", "rules", 0.0, STRATEGIES["primary only"])["success"] == 1.0
    )
    outage_plain = simulate(CASES, runs, "p", "f", "rules", 1.0, STRATEGIES["primary only"])
    outage_degraded = simulate(CASES, runs, "p", "f", "rules", 1.0, full)
    assert outage_plain["success"] == 0.0
    assert outage_degraded["success"] == pytest.approx(
        summary(CASES, runs["rules"])["overall_accuracy"]
    )
    assert outage_degraded["served_by"] == {"rules": len(CASES)}


def test_cli_end_to_end(tmp_path):
    rep = tmp_path / "reports"
    main(["--reports", str(rep), "run", "rules"])
    assert (rep / "summary_rules.json").exists()
    main(["--reports", str(rep), "gate", "rules", "rules"])
    main(["--reports", str(rep), "continuity", "--primary", "rules", "--fallback", "rules"])
    main(["--reports", str(rep), "report"])
    text = (rep / "results.md").read_text(encoding="utf-8")
    assert "Decision: **PASS**" in text and "p_fail=0.5" in text
