#!/usr/bin/env python3
"""QuantBot command-line entry point.

    python main.py config-check                 validate the configuration
    python main.py fetch                        download and cache candles
    python main.py scan                         one analysis pass, printed
    python main.py run                          the live signal loop
    python main.py backtest --symbol BTC/USDT   replay history
    python main.py optimize                     walk-forward parameter search
    python main.py telegram-test                verify notifications
    python main.py selftest                     exercise the pipeline offline

Every command accepts ``--config PATH`` and repeatable ``--set section.key=value``
overrides, so nothing needs editing to try a variation:

    python main.py backtest --set risk.risk_per_trade_pct=0.5 --set strategy.min_score_to_emit=80
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import ConfigError, Settings, load_settings
from utils.logger import get_logger, setup_logging

_log = get_logger("main")

#: Exit codes, so a systemd unit or a CI job can react meaningfully.
EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_DATA = 3
EXIT_RUNTIME = 4


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------
def _parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Turn ``["a.b=1", "c.d=x"]`` into a mapping."""
    overrides: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ConfigError(f"--set expects section.key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        overrides[key.strip()] = value.strip()
    return overrides


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="quantbot",
        description="Modular, risk-first trading-signal bot (crypto / forex / stocks).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--set", dest="overrides", action="append", metavar="KEY=VALUE",
        help="override a config value (repeatable), e.g. --set risk.risk_per_trade_pct=0.5",
    )
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--no-env", action="store_true", help="do not read the .env file")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("config-check", help="validate the configuration and print it masked")
    subparsers.add_parser("run", help="start the live signal loop")
    subparsers.add_parser("scan", help="analyse every symbol once and print the verdicts")
    subparsers.add_parser("telegram-test", help="verify the Telegram configuration")
    subparsers.add_parser("selftest", help="exercise the full pipeline on synthetic data, offline")

    fetch = subparsers.add_parser("fetch", help="download and cache candles")
    fetch.add_argument("--symbol", action="append", help="symbol (repeatable); default: all")

    backtest = subparsers.add_parser("backtest", help="replay history")
    backtest.add_argument("--symbol", default=None)
    backtest.add_argument("--timeframe", default=None)
    backtest.add_argument("--start", default=None, help="YYYY-MM-DD")
    backtest.add_argument("--end", default=None)
    backtest.add_argument("--offline", action="store_true", help="use cached data only")
    backtest.add_argument("--report-dir", default="reports", help="where to write CSV/HTML")
    backtest.add_argument("--no-export", action="store_true", help="print only")

    optimize = subparsers.add_parser("optimize", help="walk-forward parameter search")
    optimize.add_argument("--symbol", default=None)
    optimize.add_argument("--offline", action="store_true")
    optimize.add_argument("--max-candidates", type=int, default=None)
    optimize.add_argument("--out", default="reports/optimisation.json")

    return parser


def _load(args: argparse.Namespace) -> Settings:
    """Load settings with CLI overrides applied."""
    overrides = _parse_overrides(args.overrides)
    if args.log_level:
        overrides["logging.level"] = args.log_level
    settings = load_settings(
        args.config, overrides=overrides, env_file=None if args.no_env else ".env"
    )
    setup_logging(settings.logging)
    return settings


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
async def _load_frames(
    settings: Settings, symbol: str, *, offline: bool
) -> dict[str, pd.DataFrame]:
    """Load every timeframe the strategy needs for one symbol."""
    from data.collector import DataCollector
    from data.exchange import CCXTProvider, ExchangeUnavailable

    provider = None
    if not offline:
        try:
            provider = CCXTProvider(settings.exchange)
            await provider.load_markets()
        except ExchangeUnavailable as exc:
            _log.warning("%s - falling back to cached data", exc)
            provider = None
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "cannot reach %s (%s); falling back to cached data",
                settings.exchange.id, exc,
            )
            if provider is not None:
                await provider.close()
            provider = None

    collector = DataCollector(settings, provider)
    try:
        frames = await collector.repository.get_many(
            symbol, collector.timeframes, refresh=provider is not None
        )
    finally:
        await collector.close()
        if provider is not None:
            await provider.close()
    return frames


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def command_config_check(settings: Settings) -> int:
    """Print the validated configuration with credentials masked."""
    print(json.dumps(settings.to_dict(), indent=2, default=str))
    print("\nConfiguration is valid.")
    warnings: list[str] = []
    if not settings.telegram.enabled:
        warnings.append("telegram.enabled is false - signals go to the log files only")
    if not settings.exchange.has_credentials:
        warnings.append(
            "no exchange API keys - public endpoints only (fine for data; rate limits are lower)"
        )
    if settings.app.mode == "signal":
        warnings.append("app.mode=signal - no positions will be tracked; use 'paper' to track them")
    if warnings:
        print("\nNotes:")
        for note in warnings:
            print(f"  - {note}")
    return EXIT_OK


async def command_run(settings: Settings) -> int:
    """Start the live loop."""
    from live.trader import LiveTrader

    return await LiveTrader(settings).run()


async def command_scan(settings: Settings) -> int:
    """One analysis pass over every symbol."""
    from live.trader import LiveTrader
    from strategies.base import Signal

    verdicts = await LiveTrader(settings).scan_once()
    signals = [v for v in verdicts if isinstance(v, Signal)]

    print(f"\nScanned {len(verdicts)} symbol(s) on {settings.data.primary_timeframe}\n")
    for verdict in verdicts:
        marker = ">>" if isinstance(verdict, Signal) else "  "
        print(f"{marker} {verdict.summary()}")
    print(f"\n{len(signals)} signal(s), {len(verdicts) - len(signals)} rejection(s).")
    if not signals:
        print(
            "No signal is the normal outcome: the gates require multi-timeframe agreement, a\n"
            "trigger on this bar, volume confirmation and a net RR above the minimum."
        )
    return EXIT_OK


async def command_fetch(settings: Settings, symbols: list[str] | None) -> int:
    """Download and cache candles."""
    from data.collector import DataCollector
    from data.exchange import CCXTProvider, ExchangeUnavailable

    targets = symbols or settings.universe.symbols
    try:
        provider = CCXTProvider(settings.exchange)
        await provider.load_markets()
    except ExchangeUnavailable as exc:
        _log.error("%s", exc)
        return EXIT_DATA
    except Exception as exc:  # noqa: BLE001
        _log.error("cannot reach %s: %s", settings.exchange.id, exc)
        return EXIT_DATA

    collector = DataCollector(settings, provider)
    try:
        for symbol in targets:
            frames = await collector.repository.get_many(symbol, collector.timeframes)
            for timeframe, frame in frames.items():
                span = (
                    f"{frame.index[0]:%Y-%m-%d} -> {frame.index[-1]:%Y-%m-%d %H:%M}"
                    if not frame.empty
                    else "empty"
                )
                print(f"  {symbol:<12} {timeframe:<5} {len(frame):>6} bars  {span}")
    finally:
        await collector.close()
        await provider.close()
    print(f"\nCached under {settings.data.cache_dir}/")
    return EXIT_OK


async def command_backtest(settings: Settings, args: argparse.Namespace) -> int:
    """Replay history and print a report."""
    from backtest.engine import Backtester
    from backtest.report import export_csv, render_html, text_report
    from strategies import build_strategy

    overrides: dict[str, Any] = {}
    if args.symbol:
        overrides["backtest.symbol"] = args.symbol
    if args.timeframe:
        overrides["backtest.timeframe"] = args.timeframe
    if args.start:
        overrides["backtest.start"] = args.start
    if args.end:
        overrides["backtest.end"] = args.end
    if overrides:
        settings = settings.with_overrides(overrides)

    symbol = settings.backtest.symbol
    frames = await _load_frames(settings, symbol, offline=args.offline)
    if not frames or frames.get(settings.backtest.timeframe) is None:
        _log.error(
            "no data for %s %s. Run 'python main.py fetch' first.",
            symbol, settings.backtest.timeframe,
        )
        return EXIT_DATA

    try:
        result = Backtester(settings, build_strategy(settings)).run(frames, symbol=symbol)
    except ValueError as exc:
        _log.error("backtest cannot run: %s", exc)
        return EXIT_DATA

    print(text_report(result))

    if not args.no_export:
        export_csv(result, args.report_dir)
        html = Path(args.report_dir) / f"{symbol.replace('/', '')}_{result.timeframe}.html"
        if render_html(result, html) is not None:
            print(f"\nHTML report: {html}")
    return EXIT_OK


async def command_optimize(settings: Settings, args: argparse.Namespace) -> int:
    """Run the walk-forward parameter search."""
    from backtest.optimizer import Optimiser

    if args.symbol:
        settings = settings.with_overrides({"backtest.symbol": args.symbol})
    symbol = settings.backtest.symbol

    frames = await _load_frames(settings, symbol, offline=args.offline)
    if not frames.get(settings.backtest.timeframe, pd.DataFrame()).shape[0]:
        _log.error("no data for %s. Run 'python main.py fetch' first.", symbol)
        return EXIT_DATA

    try:
        optimiser = Optimiser(settings, frames, symbol)
    except ConfigError as exc:
        _log.error("%s", exc)
        return EXIT_CONFIG

    print(
        f"Searching {optimiser.grid_size} combination(s) with method="
        f"{settings.optimization.method}, objective={settings.optimization.objective}.\n"
        f"Each candidate is fitted on the first {settings.optimization.train_fraction:.0%} of the\n"
        f"data and scored on the remainder, which it never saw.\n"
    )
    report = optimiser.run(max_candidates=args.max_candidates)
    print(report.summary())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    print(f"\nFull results: {out}")
    return EXIT_OK


async def command_telegram_test(settings: Settings) -> int:
    """Verify the Telegram configuration end to end."""
    from notify.notifier import Notifier

    if not settings.telegram.enabled:
        print(
            "telegram.enabled is false in the configuration.\n"
            "Set it to true (and fill TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env) first."
        )
        return EXIT_CONFIG

    notifier = Notifier(settings)
    try:
        ok, detail = await notifier.test()
    finally:
        await notifier.close()
    print(("OK: " if ok else "FAILED: ") + detail)
    return EXIT_OK if ok else EXIT_RUNTIME


def command_selftest(settings: Settings) -> int:
    """Exercise the pipeline end to end on generated data, offline.

    **This proves the code runs; it says nothing about whether the strategy
    works.** The series is a seeded random walk with injected regime shifts — it
    has no market microstructure, no real volume behaviour and no news. Judging
    a strategy on it would be meaningless, which is why real backtests require
    real cached candles and this command never reports a verdict on edge.
    """
    import numpy as np

    from backtest.engine import Backtester
    from backtest.report import text_report
    from strategies import build_strategy

    print(
        "SELF TEST - synthetic data.\n"
        "This verifies that every module runs and agrees on its interfaces.\n"
        "It is NOT evidence about the strategy: the series is a seeded random walk.\n"
    )

    rng = np.random.default_rng(20240101)
    count = 6000
    index = pd.date_range("2024-01-01", periods=count, freq="15min", tz="UTC")
    regime = np.repeat(rng.choice([0.0011, -0.0009, 0.0], size=count // 80 + 1), 80)[:count]
    returns = rng.normal(0, 0.0035, count) + regime
    close = 30_000 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0018, count)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0018, count)))
    volume = rng.lognormal(5, 0.85, count)
    base = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )

    aggregation = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    frames = {"15m": base}
    for timeframe, rule in (("1h", "1h"), ("4h", "4h")):
        frames[timeframe] = base.resample(rule).agg(aggregation).dropna()

    # 4h needs ~240 bars to seed EMA-200, which 6000 15m bars provide; 1d would
    # need 240 days, so the self test uses a 4h/1h/15m ladder.
    settings = settings.with_overrides(
        {
            "backtest.timeframe": "15m",
            "backtest.start": "2024-02-01",
            "strategy.mtf.context": ["4h"],
            "strategy.mtf.confirm": ["1h"],
            "strategy.mtf.entry": ["15m"],
            "data.primary_timeframe": "15m",
            "indicators.adx_trend_min": 18,
            "strategy.min_score_to_emit": 60,
        }
    )
    result = Backtester(settings, build_strategy(settings)).run(frames, symbol="SYNTH/USDT")
    print(text_report(result))

    checks = {
        "bars processed": result.metrics.bars > 1000,
        "strategy evaluated every bar": sum(result.rejections.values()) + result.signals_emitted > 0,
        "equity curve recorded": not result.equity_curve.empty,
        "metrics computed": result.metrics.initial_equity > 0,
    }
    print("\nInterface checks:")
    for name, passed in checks.items():
        print(f"  [{'ok' if passed else 'FAIL'}] {name}")
    if not all(checks.values()):
        return EXIT_RUNTIME
    print("\nAll modules ran and agreed. Now fetch real data and backtest on it.")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch."""
    args = build_parser().parse_args(argv)

    try:
        settings = _load(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    try:
        if args.command == "config-check":
            return command_config_check(settings)
        if args.command == "selftest":
            return command_selftest(settings)
        if args.command == "run":
            return asyncio.run(command_run(settings))
        if args.command == "scan":
            return asyncio.run(command_scan(settings))
        if args.command == "fetch":
            return asyncio.run(command_fetch(settings, args.symbol))
        if args.command == "backtest":
            return asyncio.run(command_backtest(settings, args))
        if args.command == "optimize":
            return asyncio.run(command_optimize(settings, args))
        if args.command == "telegram-test":
            return asyncio.run(command_telegram_test(settings))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_OK
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
