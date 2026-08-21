"""Daily monitor: refresh data, replay the strategy, publish the dashboard feed.

    python3 scripts/monitor.py             # one cycle
    python3 scripts/monitor.py --no-fetch  # reuse local data
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btcbot import data as D
from btcbot.config import CHAMPION, DEFAULT_PROFILE, RISK_PROFILES, RISK_WARNING
from btcbot.live import PAPER_START, run_paper

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
RESEARCH = ROOT / "research"


def json_safe(obj):
    """Strip non-finite floats before writing.

    Python's json writes Infinity/NaN as bare literals, which the browser's
    JSON.parse rejects outright — so a profile with zero losing trades (its
    profit factor is +inf) does not degrade the dashboard, it blanks the whole
    page with a parse error. Python round-trips the file happily, which is why
    this only shows up if someone actually opens the page.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def refresh_data() -> dict:
    """Pull any bars that closed since the last run."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fetch_data.py")],
        capture_output=True, text=True, timeout=1800,
    )
    return {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()

    payload: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_start": PAPER_START,
        "config": {k: v for k, v in CHAMPION.items() if k != "daily"},
        "risk_profiles": RISK_PROFILES,
        "risk_warning": RISK_WARNING,
        "errors": [],
    }

    if not args.no_fetch:
        fetch = refresh_data()
        payload["data_refresh"] = {"ok": fetch["ok"]}
        if not fetch["ok"]:
            payload["errors"].append(f"data refresh failed: {fetch['stderr'][-400:]}")
            print("WARNING: data refresh failed, continuing with local data",
                  file=sys.stderr)

    df = D.load_ohlcv(CHAMPION["tf"])
    payload["data"] = {
        "symbol": "BTCUSDT",
        "timeframe": CHAMPION["tf"],
        "bars": int(len(df)),
        "last_bar": df.index[-1].isoformat(),
        "last_price": float(df["close"].iloc[-1]),
    }

    # A stale feed is the failure mode that makes a dashboard lie quietly, so
    # it gets surfaced rather than swallowed.
    age_h = (datetime.now(timezone.utc) - df.index[-1].to_pydatetime()).total_seconds() / 3600
    payload["data"]["age_hours"] = round(age_h, 1)
    payload["data"]["stale"] = age_h > 12

    # Which exchange actually served this run. Running on a backup is fine;
    # running on a backup without knowing it is how you end up explaining a
    # basis as a strategy edge.
    prov = ROOT / "data" / "fetch_provenance.json"
    if prov.exists():
        payload["data_sources"] = json.loads(prov.read_text())
        used = {v.get("source") for v in payload["data_sources"]
                .get("timeframes", {}).values()}
        used.add(payload["data_sources"].get("funding", {}).get("source"))
        fallback = sorted(s for s in used
                          if s and not s.startswith(("binance", "archive", "none")))
        payload["data"]["on_backup_source"] = fallback
        if fallback:
            payload["errors"].append(f"running on backup data source: {', '.join(fallback)}")

    payload["profiles"] = {}
    for profile in RISK_PROFILES:
        try:
            payload["profiles"][profile] = run_paper(profile)
        except Exception as exc:  # noqa: BLE001 - one bad profile must not kill the run
            payload["errors"].append(f"{profile}: {exc}")
            traceback.print_exc()

    if payload["profiles"]:
        payload["default_profile"] = DEFAULT_PROFILE \
            if DEFAULT_PROFILE in payload["profiles"] else next(iter(payload["profiles"]))

    report = RESEARCH / "final_report.json"
    if report.exists():
        full = json.loads(report.read_text())
        payload["backtest"] = {
            k: full.get(k) for k in
            ("comparison", "walk_forward", "monte_carlo", "per_year",
             "concentration", "risk_profiles", "equity_curve", "data")
        }

    SITE.mkdir(exist_ok=True)
    payload = json_safe(payload)
    text = json.dumps(payload, indent=2, allow_nan=False)  # raises rather than emit Infinity
    (SITE / "data.json").write_text(text)
    (SITE / ".nojekyll").write_text("")

    default = payload.get("default_profile")
    if default:
        state = payload["profiles"][default]
        pos = state["position"]
        print(f"[{payload['generated_at']}] BTC {payload['data']['last_price']:,.0f} "
              f"| {default}: equity ${state['equity']:,.0f} "
              f"({state['return_pct']:+.1%}) | "
              f"{'IN ' + pos['direction'] + ' @ ' + format(pos['entry_price'], ',.0f') if pos else 'FLAT'}"
              f" | signal {'READY' if state['signal']['ready'] else 'waiting'}")
    if payload["errors"]:
        print("errors:", payload["errors"], file=sys.stderr)
    print(f"wrote {SITE / 'data.json'}")


if __name__ == "__main__":
    main()
