"""Configuration validation.

The validator is the project's first line of defence: a mis-parameterised bot
that starts is far more dangerous than one that refuses to. These tests pin the
rejections that matter, so a future "convenience" relaxation has to be
deliberate.
"""

from __future__ import annotations

import copy

import pytest

from config.settings import ConfigError, Settings, load_settings


@pytest.fixture
def raw() -> dict:
    """The shipped configuration as a plain mapping."""
    return copy.deepcopy(load_settings("config/config.yaml", env_file=None).raw)


def test_shipped_config_is_valid() -> None:
    settings = load_settings("config/config.yaml", env_file=None)
    assert settings.strategy.name == "confluence"
    assert settings.data.primary_timeframe in settings.strategy.mtf.entry


def test_unknown_key_is_rejected(raw: dict) -> None:
    """A typo must fail loudly, not silently leave the default in place."""
    raw["risk"]["risk_pct"] = 1.0
    with pytest.raises(ConfigError, match="Unknown key"):
        Settings.from_mapping(raw)


def test_unknown_section_is_rejected(raw: dict) -> None:
    raw["stratergy"] = {}
    with pytest.raises(ConfigError, match="Unknown top-level section"):
        Settings.from_mapping(raw)


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("risk", "risk_per_trade_pct", 5.0, "not in allowed_risk_levels"),
        ("risk", "min_rr", 0.5, "min_rr < 1"),
        ("risk", "tp_split", [0.7, 0.7], "must sum to 1.0"),
        ("risk", "daily_loss_limit_pct", 20.0, "should be < max_drawdown_pct"),
        ("risk", "leverage", 0.5, "must be >= 1"),
        ("indicators", "atr_min_pct", 9.0, "must be < atr_max_pct"),
        ("indicators", "macd_fast", 30, "macd_fast must be < macd_slow"),
        ("indicators", "rsi_overbought", 20.0, "rsi_oversold < rsi_overbought"),
        ("backtest", "execution", "close", "lookahead"),
        ("optimization", "min_trades", 5, "statistical noise"),
        ("app", "mode", "live", "not supported"),
        ("app", "only_closed_bars", False, "forming bar"),
    ],
)
def test_dangerous_values_are_rejected(
    raw: dict, section: str, key: str, value: object, match: str
) -> None:
    raw[section][key] = value
    with pytest.raises(ConfigError, match=match):
        Settings.from_mapping(raw)


def test_ema_ordering_is_enforced(raw: dict) -> None:
    raw["indicators"]["ema_fast"] = 100
    with pytest.raises(ConfigError, match="ema_fast < ema_slow < ema_trend"):
        Settings.from_mapping(raw)


def test_furthest_target_must_reach_min_rr(raw: dict) -> None:
    """A configuration where no trade could ever pass the RR gate is rejected."""
    raw["risk"]["tp_rr_targets"] = [1.0, 1.5]
    raw["risk"]["min_rr"] = 2.0
    with pytest.raises(ConfigError, match="below risk.min_rr"):
        Settings.from_mapping(raw)


def test_mtf_timeframe_must_be_downloaded(raw: dict) -> None:
    raw["strategy"]["mtf"]["confirm"] = ["8h"]
    with pytest.raises(ConfigError, match="not downloaded"):
        Settings.from_mapping(raw)


def test_warmup_must_cover_longest_lookback(raw: dict) -> None:
    raw["data"]["warmup_bars"] = 50
    with pytest.raises(ConfigError, match="longest indicator"):
        Settings.from_mapping(raw)


def test_funding_requires_futures(raw: dict) -> None:
    raw["exchange"]["market_type"] = "spot"
    raw["data"]["funding_rate"] = True
    with pytest.raises(ConfigError, match="requires exchange.market_type=future"):
        Settings.from_mapping(raw)


def test_telegram_enabled_without_credentials_is_rejected(raw: dict) -> None:
    raw["telegram"]["enabled"] = True
    raw["telegram"]["bot_token"] = ""
    with pytest.raises(ConfigError, match="TELEGRAM_BOT_TOKEN"):
        Settings.from_mapping(raw)


def test_weights_need_not_sum_to_100(raw: dict) -> None:
    """Weights are relative; the scorer normalises over the available weight."""
    raw["strategy"]["weights"] = dict.fromkeys(raw["strategy"]["weights"], 5.0)
    settings = Settings.from_mapping(raw)
    assert sum(settings.strategy.weights.values()) == pytest.approx(40.0)


def test_missing_weight_component_is_rejected(raw: dict) -> None:
    raw["strategy"]["weights"].pop("volume")
    with pytest.raises(ConfigError, match="missing component"):
        Settings.from_mapping(raw)


def test_overrides_coerce_types(settings: Settings) -> None:
    """``--set`` passes strings; they must land as the right Python types."""
    updated = settings.with_overrides(
        {
            "risk.risk_per_trade_pct": "0.5",
            "indicators.ema_fast": "13",
            "strategy.allow_short": "false",
            "backtest.start": "2024-01-01",
        }
    )
    assert updated.risk.risk_per_trade_pct == 0.5
    assert updated.indicators.ema_fast == 13
    assert updated.strategy.allow_short is False
    assert updated.backtest.start == "2024-01-01"
    # The original must be untouched: the optimiser relies on this.
    assert settings.risk.risk_per_trade_pct == 1.0


def test_override_of_unknown_path_is_rejected(settings: Settings) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        settings.with_overrides({"risk.nope": 1})


def test_overrides_are_revalidated(settings: Settings) -> None:
    with pytest.raises(ConfigError, match="not in allowed_risk_levels"):
        settings.with_overrides({"risk.risk_per_trade_pct": "9"})


def test_to_dict_masks_secrets(raw: dict) -> None:
    # Distinctive sentinels: "secret" itself appears in the *key* name
    # (api_secret), so a substring check on it would be vacuous.
    raw["exchange"]["api_key"] = "AKIA-LEAKME-KEY"
    raw["exchange"]["api_secret"] = "LEAKME-SECRETVALUE"
    raw["telegram"]["bot_token"] = "123:LEAKME-TOKEN"
    raw["telegram"]["chat_id"] = "-100123"
    raw["telegram"]["enabled"] = False
    raw["data"]["news_api_key"] = "LEAKME-NEWSKEY"

    masked = Settings.from_mapping(raw).to_dict()
    assert masked["exchange"]["api_key"] == "***"
    assert masked["exchange"]["api_secret"] == "***"
    assert masked["telegram"]["bot_token"] == "***"
    assert masked["telegram"]["chat_id"] == "***"
    assert masked["data"]["news_api_key"] == "***"
    assert "LEAKME" not in str(masked)


def test_round_trip_cost_is_both_sides(settings: Settings) -> None:
    risk = settings.risk
    assert risk.round_trip_cost_pct == pytest.approx(2 * (risk.fee_pct + risk.slippage_pct))
