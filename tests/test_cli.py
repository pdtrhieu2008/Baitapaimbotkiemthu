"""Command-line surface.

The override-precedence tests exist because this was a real bug: with ordinary
argparse ``parents=``, the sub-parser's default silently overwrote a ``--set``
given *before* the sub-command, so a run that looked configured at 0.5% risk
actually used 1%. Silent is the dangerous part.
"""

from __future__ import annotations

import pytest

from config.settings import ConfigError
from main import _load, _parse_overrides, build_parser, command_config_check, main


def load(argv: list[str]):
    """Parse ``argv`` and build the resulting Settings."""
    return _load(build_parser().parse_args(argv))


# ---------------------------------------------------------------------------
# Override parsing
# ---------------------------------------------------------------------------
def test_parse_overrides_splits_on_the_first_equals() -> None:
    parsed = _parse_overrides(["a.b=1", "c.d=x=y"])
    assert parsed == {"a.b": "1", "c.d": "x=y"}


def test_parse_overrides_rejects_a_missing_value() -> None:
    with pytest.raises(ConfigError, match="section.key=value"):
        _parse_overrides(["nonsense"])


def test_parse_overrides_handles_none() -> None:
    assert _parse_overrides(None) == {}


# ---------------------------------------------------------------------------
# Override precedence - the regression this file exists for
# ---------------------------------------------------------------------------
def test_override_before_the_subcommand() -> None:
    settings = load(["--no-env", "--set", "risk.risk_per_trade_pct=0.5", "config-check"])
    assert settings.risk.risk_per_trade_pct == 0.5


def test_override_after_the_subcommand() -> None:
    settings = load(["--no-env", "config-check", "--set", "risk.risk_per_trade_pct=0.5"])
    assert settings.risk.risk_per_trade_pct == 0.5


def test_no_override_keeps_the_configured_value() -> None:
    assert load(["--no-env", "config-check"]).risk.risk_per_trade_pct == 1.0


def test_override_on_both_sides_prefers_the_later_one() -> None:
    settings = load(
        [
            "--no-env", "--set", "risk.risk_per_trade_pct=2.0",
            "config-check", "--set", "risk.risk_per_trade_pct=0.5",
        ]
    )
    assert settings.risk.risk_per_trade_pct == 0.5


def test_repeatable_overrides_all_apply() -> None:
    settings = load(
        [
            "--no-env", "config-check",
            "--set", "risk.risk_per_trade_pct=0.5",
            "--set", "strategy.min_score_to_emit=80",
            "--set", "indicators.ema_fast=13",
        ]
    )
    assert settings.risk.risk_per_trade_pct == 0.5
    assert settings.strategy.min_score_to_emit == 80
    assert settings.indicators.ema_fast == 13


def test_log_level_flag_reaches_the_settings() -> None:
    assert load(["--no-env", "config-check", "--log-level", "DEBUG"]).logging.level == "DEBUG"


def test_readme_backtest_example_parses() -> None:
    """The exact invocation printed in the README must actually work."""
    args = build_parser().parse_args(
        [
            "backtest", "--symbol", "ETH/USDT", "--timeframe", "1h", "--start", "2024-01-01",
            "--no-env",
            "--set", "risk.risk_per_trade_pct=0.5",
            "--set", "strategy.min_score_to_emit=80",
        ]
    )
    assert args.command == "backtest"
    assert args.symbol == "ETH/USDT"
    settings = _load(args)
    assert settings.risk.risk_per_trade_pct == 0.5
    assert settings.strategy.min_score_to_emit == 80


# ---------------------------------------------------------------------------
# Validation still applies through the CLI
# ---------------------------------------------------------------------------
def test_invalid_override_is_rejected_with_exit_code() -> None:
    code = main(["--no-env", "config-check", "--set", "risk.risk_per_trade_pct=9"])
    assert code == 2  # EXIT_CONFIG


def test_unknown_override_path_is_rejected() -> None:
    assert main(["--no-env", "config-check", "--set", "risk.nope=1"]) == 2


def test_malformed_override_is_rejected() -> None:
    assert main(["--no-env", "config-check", "--set", "garbage"]) == 2


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def test_config_check_succeeds_and_masks(capsys) -> None:
    settings = load(["--no-env", "config-check"])
    assert command_config_check(settings) == 0
    output = capsys.readouterr().out
    assert "Configuration is valid." in output
    assert '"bot_token"' in output


def test_every_subcommand_is_reachable() -> None:
    parser = build_parser()
    for command in (
        "config-check", "run", "scan", "telegram-test", "selftest",
        "fetch", "backtest", "optimize",
    ):
        assert parser.parse_args([command, "--no-env"]).command == command


def test_a_subcommand_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
