"""Typed configuration objects and the YAML/env loader.

Design rules enforced here:

* **No hard-coded parameters anywhere else.** Every module receives the piece
  of configuration it needs; nothing reads ``os.environ`` on its own except
  this file.
* **Fail fast and loudly.** A malformed config raises :class:`ConfigError` at
  start-up instead of producing a silently mis-parameterised bot.
* **Reproducible mutation.** :meth:`Settings.with_overrides` returns a *new*
  ``Settings`` built from the raw mapping, which is what the optimiser uses to
  sweep parameters without mutating shared state.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

__all__ = [
    "AppSection",
    "BacktestConfig",
    "ConfigError",
    "DataConfig",
    "ExchangeConfig",
    "IndicatorConfig",
    "LoggingConfig",
    "MTFConfig",
    "OptimizationConfig",
    "RiskConfig",
    "Settings",
    "StrategyConfig",
    "StructureConfig",
    "TelegramConfig",
    "UniverseConfig",
    "load_settings",
]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_T = TypeVar("_T")


class ConfigError(RuntimeError):
    """Raised when the configuration is missing, malformed or inconsistent."""


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------
def _load_dotenv(path: Path) -> None:
    """Populate ``os.environ`` from a ``.env`` file without extra dependencies.

    Real environment variables always win, so a container's ``-e`` flags or a
    systemd ``Environment=`` line override the file. Lines may be blank,
    comments, or ``KEY=value`` with optional ``export`` prefix and quotes.
    """
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ").strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _expand_env(node: Any) -> Any:
    """Recursively replace ``${VAR}`` placeholders with environment values.

    An unset variable becomes an empty string rather than an error: many
    credentials are genuinely optional (the bot runs against public endpoints
    without API keys). The components that *require* a value validate it
    themselves and report a precise message.
    """
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), node)
    return node


def _deep_update(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``other`` into ``base`` (returns a new mapping)."""
    merged = dict(base)
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce(value: Any, target: Any) -> Any:
    """Best-effort scalar coercion so ``--set x=3`` yields an int, not a str."""
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null", ""}:
        return None
    if isinstance(target, bool):
        return bool(value)
    try:
        if isinstance(target, int) and not isinstance(target, bool):
            return int(value)
        if isinstance(target, float):
            return float(value)
    except ValueError:
        pass
    # No type hint to lean on: try int then float then leave as string.
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _section(cls: type[_T], data: Any, name: str) -> _T:
    """Build a dataclass from a mapping, rejecting unknown keys.

    Unknown keys are an error, not a warning: a typo such as ``risk_pct``
    instead of ``risk_per_trade_pct`` would otherwise leave the default
    silently in place, which is exactly the class of bug that costs money.
    """
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"Section '{name}' must be a mapping, got {type(data).__name__}")
    valid = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - valid
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in section '{name}': {sorted(unknown)}. Valid keys: {sorted(valid)}"
        )
    return cls(**data)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class AppSection:
    """Top-level runtime behaviour."""

    name: str = "quantbot"
    mode: str = "signal"
    loop_interval_seconds: float = 30.0
    only_closed_bars: bool = True
    state_file: str = "logs/state.json"
    persist_state: bool = True

    def validate(self) -> None:
        if self.mode not in {"signal", "paper", "live"}:
            raise ConfigError(f"app.mode must be signal|paper|live, got {self.mode!r}")
        if self.mode == "live":
            raise ConfigError(
                "app.mode='live' is not supported: this project ships no order-execution "
                "module by design. See README section 'Why there is no execution module'."
            )
        if not self.only_closed_bars:
            raise ConfigError(
                "app.only_closed_bars=false would evaluate signals on a forming bar, whose "
                "high/low/close still change. That makes live results non-reproducible and "
                "inconsistent with the backtester. Refusing to start."
            )
        if self.loop_interval_seconds < 1:
            raise ConfigError("app.loop_interval_seconds must be >= 1")


@dataclass(slots=True)
class ExchangeConfig:
    """ccxt connection parameters."""

    id: str = "binance"
    market_type: str = "future"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    enable_rate_limit: bool = True
    timeout_ms: int = 20_000
    max_retries: int = 4
    retry_backoff_seconds: float = 2.0

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret)

    @property
    def is_futures(self) -> bool:
        return self.market_type in {"future", "futures", "swap"}

    def validate(self) -> None:
        if self.market_type not in {"spot", "future", "futures", "swap"}:
            raise ConfigError(f"exchange.market_type invalid: {self.market_type!r}")
        if self.max_retries < 0:
            raise ConfigError("exchange.max_retries must be >= 0")


@dataclass(slots=True)
class UniverseConfig:
    """Which instruments get scanned."""

    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT"])
    quote_currency: str = "USDT"
    min_quote_volume_24h: float = 0.0

    def validate(self) -> None:
        if not self.symbols:
            raise ConfigError("universe.symbols must not be empty")


@dataclass(slots=True)
class DataConfig:
    """Market-data collection settings."""

    primary_timeframe: str = "15m"
    timeframes: list[str] = field(default_factory=lambda: ["15m", "1h", "4h"])
    ohlcv_limit: int = 600
    warmup_bars: int = 250
    cache_dir: str = "data/cache"
    persist: bool = True
    funding_rate: bool = True
    open_interest: bool = True
    liquidations: bool = False
    fear_greed: bool = True
    btc_dominance: bool = True
    news_sentiment: bool = False
    news_provider: str = "cryptopanic"
    news_api_key: str = ""
    context_ttl_seconds: float = 300.0

    def validate(self) -> None:
        if self.primary_timeframe not in self.timeframes:
            raise ConfigError(
                f"data.primary_timeframe {self.primary_timeframe!r} must be listed in "
                f"data.timeframes {self.timeframes}"
            )
        if self.ohlcv_limit <= self.warmup_bars:
            raise ConfigError(
                f"data.ohlcv_limit ({self.ohlcv_limit}) must exceed data.warmup_bars "
                f"({self.warmup_bars}), otherwise no bar is ever tradable"
            )
        if self.news_sentiment and not self.news_api_key:
            raise ConfigError(
                "data.news_sentiment=true but NEWS_API_KEY is empty. Either set the key or "
                "set news_sentiment=false (its scoring weight is then redistributed)."
            )


@dataclass(slots=True)
class IndicatorConfig:
    """Every indicator period and threshold."""

    ema_fast: int = 21
    ema_slow: int = 50
    ema_trend: int = 200
    sma_period: int = 100
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_bull_mid: float = 50.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14
    adx_trend_min: float = 25.0
    adx_strong: float = 40.0
    atr_period: int = 14
    atr_min_pct: float = 0.15
    atr_max_pct: float = 5.0
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    bb_period: int = 20
    bb_std: float = 2.0
    keltner_period: int = 20
    keltner_multiplier: float = 1.5
    donchian_period: int = 20
    stoch_rsi_period: int = 14
    stoch_rsi_k: int = 3
    stoch_rsi_d: int = 3
    mfi_period: int = 14
    cmf_period: int = 20
    obv_ema_period: int = 21
    volume_sma_period: int = 20
    volume_spike_multiplier: float = 1.8
    volume_dry_multiplier: float = 0.6
    volume_profile_bins: int = 24
    volume_profile_lookback: int = 120
    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_senkou_b: int = 52
    ichimoku_displacement: int = 26
    pivot_method: str = "classic"
    pivot_timeframe: str = "1d"

    @property
    def max_period(self) -> int:
        """Longest lookback, used to size the warm-up window."""
        return max(
            self.ema_trend,
            self.sma_period,
            self.ichimoku_senkou_b + self.ichimoku_displacement,
            self.volume_profile_lookback,
            self.macd_slow + self.macd_signal,
        )

    def validate(self) -> None:
        periods = {
            name: getattr(self, name)
            for name in (
                "ema_fast", "ema_slow", "ema_trend", "sma_period", "rsi_period",
                "macd_fast", "macd_slow", "macd_signal", "adx_period", "atr_period",
                "supertrend_period", "bb_period", "keltner_period", "donchian_period",
                "stoch_rsi_period", "mfi_period", "cmf_period", "volume_sma_period",
                "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_b",
            )
        }
        for name, value in periods.items():
            if int(value) < 1:
                raise ConfigError(f"indicators.{name} must be >= 1, got {value}")
        if not self.ema_fast < self.ema_slow < self.ema_trend:
            raise ConfigError(
                "indicators requires ema_fast < ema_slow < ema_trend, got "
                f"{self.ema_fast}/{self.ema_slow}/{self.ema_trend}"
            )
        if self.macd_fast >= self.macd_slow:
            raise ConfigError("indicators.macd_fast must be < macd_slow")
        if not 0 < self.rsi_oversold < self.rsi_overbought < 100:
            raise ConfigError("indicators requires 0 < rsi_oversold < rsi_overbought < 100")
        if self.atr_min_pct >= self.atr_max_pct:
            raise ConfigError("indicators.atr_min_pct must be < atr_max_pct")
        if self.pivot_method not in {"classic", "fibonacci"}:
            raise ConfigError("indicators.pivot_method must be classic|fibonacci")


@dataclass(slots=True)
class StructureConfig:
    """Swing / support-resistance / zone detection settings."""

    swing_strength: int = 3
    lookback: int = 150
    sr_cluster_pct: float = 0.25
    sr_min_touches: int = 2
    zone_atr_multiplier: float = 0.5
    room_to_level_atr: float = 1.0
    sweep_wick_ratio: float = 0.55
    false_breakout_bars: int = 3

    def validate(self) -> None:
        if self.swing_strength < 1:
            raise ConfigError("structure.swing_strength must be >= 1")
        if self.lookback < 4 * self.swing_strength:
            raise ConfigError("structure.lookback is too small for the chosen swing_strength")
        if not 0 < self.sweep_wick_ratio < 1:
            raise ConfigError("structure.sweep_wick_ratio must be in (0, 1)")


@dataclass(slots=True)
class MTFConfig:
    """Multi-timeframe roles and agreement threshold."""

    context: list[str] = field(default_factory=lambda: ["1d", "4h"])
    confirm: list[str] = field(default_factory=lambda: ["1h"])
    entry: list[str] = field(default_factory=lambda: ["15m", "5m"])
    min_agreement: float = 0.66
    weights: dict[str, float] = field(default_factory=dict)

    @property
    def all_timeframes(self) -> list[str]:
        seen: dict[str, None] = {}
        for tf in [*self.context, *self.confirm, *self.entry]:
            seen[tf] = None
        return list(seen)

    def weight_of(self, timeframe: str) -> float:
        return float(self.weights.get(timeframe, 1.0))

    def validate(self) -> None:
        if not self.confirm:
            raise ConfigError("strategy.mtf.confirm must list at least one timeframe")
        if not self.entry:
            raise ConfigError("strategy.mtf.entry must list at least one timeframe")
        if not 0 < self.min_agreement <= 1:
            raise ConfigError("strategy.mtf.min_agreement must be in (0, 1]")


@dataclass(slots=True)
class StrategyConfig:
    """Signal-generation rules: weights, thresholds and hard gates."""

    name: str = "confluence"
    allow_long: bool = True
    allow_short: bool = True
    min_score_strong: float = 80.0
    min_score_normal: float = 70.0
    min_score_weak: float = 60.0
    min_score_to_emit: float = 70.0
    weights: dict[str, float] = field(default_factory=dict)
    gates: dict[str, bool] = field(default_factory=dict)
    mtf: MTFConfig = field(default_factory=MTFConfig)
    cooldown_bars: int = 3
    max_signals_per_symbol_per_day: int = 6

    #: Component keys the scorer knows about.
    COMPONENTS = ("trend", "macd", "rsi", "volume", "adx", "price_action", "structure", "sentiment")
    #: Gate keys the strategy knows about.
    GATE_KEYS = (
        "mtf_alignment", "adx_min", "volume_confirmation", "atr_in_range",
        "room_to_level", "min_rr", "trigger_required", "no_conflicting_structure",
    )

    def gate_enabled(self, key: str) -> bool:
        return bool(self.gates.get(key, False))

    def validate(self) -> None:
        if not self.allow_long and not self.allow_short:
            raise ConfigError("strategy.allow_long and allow_short are both false: nothing to do")
        unknown_w = set(self.weights) - set(self.COMPONENTS)
        if unknown_w:
            raise ConfigError(f"strategy.weights has unknown component(s): {sorted(unknown_w)}")
        missing_w = set(self.COMPONENTS) - set(self.weights)
        if missing_w:
            raise ConfigError(f"strategy.weights is missing component(s): {sorted(missing_w)}")
        total = sum(float(v) for v in self.weights.values())
        if abs(total - 100.0) > 1e-6:
            raise ConfigError(f"strategy.weights must sum to 100, got {total}")
        if any(float(v) < 0 for v in self.weights.values()):
            raise ConfigError("strategy.weights must all be >= 0")
        unknown_g = set(self.gates) - set(self.GATE_KEYS)
        if unknown_g:
            raise ConfigError(f"strategy.gates has unknown key(s): {sorted(unknown_g)}")
        if not self.min_score_weak <= self.min_score_normal <= self.min_score_strong:
            raise ConfigError(
                "strategy requires min_score_weak <= min_score_normal <= min_score_strong"
            )
        if self.min_score_to_emit < self.min_score_weak:
            raise ConfigError(
                "strategy.min_score_to_emit must be >= min_score_weak, otherwise setups below "
                "the weakest tier would be emitted"
            )
        self.mtf.validate()


@dataclass(slots=True)
class RiskConfig:
    """Position sizing, stop placement and circuit breakers."""

    account_equity: float = 1000.0
    risk_per_trade_pct: float = 1.0
    allowed_risk_levels: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    max_open_positions: int = 3
    max_positions_per_symbol: int = 1
    max_correlated_positions: int = 2
    sl_atr_multiplier: float = 1.5
    sl_structure_buffer_atr: float = 0.25
    sl_max_pct: float = 5.0
    use_structure_stop: bool = True
    tp_rr_targets: list[float] = field(default_factory=lambda: [2.0, 3.0])
    tp_split: list[float] = field(default_factory=lambda: [0.5, 0.5])
    min_rr: float = 2.0
    breakeven_at_rr: float = 1.0
    trailing_enabled: bool = True
    trailing_activate_rr: float = 1.5
    trailing_atr_multiplier: float = 2.0
    daily_loss_limit_pct: float = 3.0
    max_drawdown_pct: float = 15.0
    max_consecutive_losses: int = 4
    max_spread_pct: float = 0.08
    volatility_anomaly_atr_ratio: float = 3.0
    max_funding_rate_abs: float = 0.15
    blocked_hours_utc: list[int] = field(default_factory=list)
    leverage: float = 1.0
    fee_pct: float = 0.04
    slippage_pct: float = 0.02

    @property
    def round_trip_cost_pct(self) -> float:
        """Total cost of a round trip in percent (both sides, fee + slippage)."""
        return 2.0 * (self.fee_pct + self.slippage_pct)

    def validate(self) -> None:
        if self.account_equity <= 0:
            raise ConfigError("risk.account_equity must be > 0")
        if self.risk_per_trade_pct <= 0:
            raise ConfigError("risk.risk_per_trade_pct must be > 0")
        if self.allowed_risk_levels and self.risk_per_trade_pct not in self.allowed_risk_levels:
            raise ConfigError(
                f"risk.risk_per_trade_pct ({self.risk_per_trade_pct}) is not in "
                f"allowed_risk_levels {self.allowed_risk_levels}"
            )
        if self.risk_per_trade_pct > 2.0:
            raise ConfigError(
                "risk.risk_per_trade_pct > 2% is rejected: with a 50% win rate a 4-loss streak "
                "already costs >8% of the account. Raise allowed_risk_levels deliberately if you "
                "really mean it."
            )
        if len(self.tp_rr_targets) != len(self.tp_split):
            raise ConfigError("risk.tp_rr_targets and risk.tp_split must have the same length")
        if not self.tp_rr_targets:
            raise ConfigError("risk.tp_rr_targets must not be empty")
        if abs(sum(self.tp_split) - 1.0) > 1e-6:
            raise ConfigError(f"risk.tp_split must sum to 1.0, got {sum(self.tp_split)}")
        if any(x <= 0 for x in self.tp_split):
            raise ConfigError("risk.tp_split entries must be > 0")
        if list(self.tp_rr_targets) != sorted(self.tp_rr_targets):
            raise ConfigError("risk.tp_rr_targets must be in ascending order")
        if self.min_rr < 1.0:
            raise ConfigError(
                "risk.min_rr < 1 means the average winner is smaller than the average loser; "
                "that needs a >50% win rate just to break even. Rejected."
            )
        if max(self.tp_rr_targets) < self.min_rr:
            raise ConfigError(
                f"the furthest take-profit ({max(self.tp_rr_targets)}R) is below risk.min_rr "
                f"({self.min_rr}R): no trade could ever satisfy the RR gate"
            )
        if self.sl_atr_multiplier <= 0:
            raise ConfigError("risk.sl_atr_multiplier must be > 0")
        if not 0 < self.daily_loss_limit_pct <= 100:
            raise ConfigError("risk.daily_loss_limit_pct must be in (0, 100]")
        if not 0 < self.max_drawdown_pct <= 100:
            raise ConfigError("risk.max_drawdown_pct must be in (0, 100]")
        if self.daily_loss_limit_pct >= self.max_drawdown_pct:
            raise ConfigError(
                "risk.daily_loss_limit_pct should be < max_drawdown_pct, otherwise the daily "
                "brake never fires before the global one"
            )
        if self.max_open_positions < 1:
            raise ConfigError("risk.max_open_positions must be >= 1")
        if self.leverage < 1:
            raise ConfigError("risk.leverage must be >= 1")
        if any(not 0 <= h <= 23 for h in self.blocked_hours_utc):
            raise ConfigError("risk.blocked_hours_utc entries must be in 0..23")


@dataclass(slots=True)
class BacktestConfig:
    """Backtest window and fill assumptions."""

    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    start: str | None = None
    end: str | None = None
    initial_equity: float = 1000.0
    execution: str = "next_open"
    intrabar: str = "pessimistic"
    risk_free_rate: float = 0.0
    periods_per_year: float | None = None

    def validate(self) -> None:
        if self.execution not in {"next_open", "close"}:
            raise ConfigError("backtest.execution must be next_open|close")
        if self.execution == "close":
            raise ConfigError(
                "backtest.execution='close' fills at the same close that produced the signal, "
                "which is lookahead. Use 'next_open'."
            )
        if self.intrabar not in {"pessimistic", "optimistic"}:
            raise ConfigError("backtest.intrabar must be pessimistic|optimistic")
        if self.initial_equity <= 0:
            raise ConfigError("backtest.initial_equity must be > 0")


@dataclass(slots=True)
class OptimizationConfig:
    """Parameter-search settings."""

    method: str = "grid"
    n_trials: int = 60
    objective: str = "expectancy"
    min_trades: int = 30
    train_fraction: float = 0.7
    n_jobs: int = 1
    param_grid: dict[str, list[Any]] = field(default_factory=dict)

    OBJECTIVES = ("expectancy", "profit_factor", "sharpe", "sortino", "net_profit", "calmar")

    def validate(self) -> None:
        if self.method not in {"grid", "optuna"}:
            raise ConfigError("optimization.method must be grid|optuna")
        if self.objective not in self.OBJECTIVES:
            raise ConfigError(f"optimization.objective must be one of {self.OBJECTIVES}")
        if not 0.1 <= self.train_fraction < 1.0:
            raise ConfigError("optimization.train_fraction must be in [0.1, 1.0)")
        if self.min_trades < 10:
            raise ConfigError(
                "optimization.min_trades < 10 makes every metric statistical noise. Rejected."
            )


@dataclass(slots=True)
class TelegramConfig:
    """Telegram Bot API settings."""

    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""
    parse_mode: str = "HTML"
    send_signals: bool = True
    send_trades: bool = True
    send_errors: bool = True
    send_daily_summary: bool = True
    daily_summary_hour_utc: int = 0
    rate_limit_seconds: float = 1.0
    max_retries: int = 3
    timeout_seconds: float = 15.0

    @property
    def is_configured(self) -> bool:
        return bool(self.enabled and self.bot_token and self.chat_id)

    def validate(self) -> None:
        if self.parse_mode not in {"HTML", "Markdown", "MarkdownV2"}:
            raise ConfigError("telegram.parse_mode must be HTML|Markdown|MarkdownV2")
        if not 0 <= self.daily_summary_hour_utc <= 23:
            raise ConfigError("telegram.daily_summary_hour_utc must be in 0..23")


@dataclass(slots=True)
class LoggingConfig:
    """Logging destinations and rotation."""

    level: str = "INFO"
    dir: str = "logs"
    console: bool = True
    json_lines: bool = True
    max_bytes: int = 5_242_880
    backup_count: int = 5

    def validate(self) -> None:
        if self.level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError(f"logging.level invalid: {self.level!r}")


# ---------------------------------------------------------------------------
# Root object
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Settings:
    """Fully-validated configuration tree."""

    app: AppSection = field(default_factory=AppSection)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    #: The merged mapping this object was built from (env already expanded).
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Settings:
        """Build and validate a ``Settings`` tree from a nested mapping."""
        if not isinstance(data, dict):
            raise ConfigError("configuration root must be a mapping")

        known = {
            "app", "exchange", "universe", "data", "indicators", "structure",
            "strategy", "risk", "backtest", "optimization", "telegram", "logging",
        }
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Unknown top-level section(s): {sorted(unknown)}")

        strategy_raw = dict(data.get("strategy") or {})
        mtf_raw = strategy_raw.pop("mtf", {}) or {}
        strategy = _section(StrategyConfig, strategy_raw, "strategy")
        strategy.mtf = _section(MTFConfig, mtf_raw, "strategy.mtf")

        settings = cls(
            app=_section(AppSection, data.get("app"), "app"),
            exchange=_section(ExchangeConfig, data.get("exchange"), "exchange"),
            universe=_section(UniverseConfig, data.get("universe"), "universe"),
            data=_section(DataConfig, data.get("data"), "data"),
            indicators=_section(IndicatorConfig, data.get("indicators"), "indicators"),
            structure=_section(StructureConfig, data.get("structure"), "structure"),
            strategy=strategy,
            risk=_section(RiskConfig, data.get("risk"), "risk"),
            backtest=_section(BacktestConfig, data.get("backtest"), "backtest"),
            optimization=_section(OptimizationConfig, data.get("optimization"), "optimization"),
            telegram=_section(TelegramConfig, data.get("telegram"), "telegram"),
            logging=_section(LoggingConfig, data.get("logging"), "logging"),
            raw=copy.deepcopy(data),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Run every section validator, then the cross-section checks."""
        for section in (
            self.app, self.exchange, self.universe, self.data, self.indicators,
            self.structure, self.strategy, self.risk, self.backtest,
            self.optimization, self.telegram, self.logging,
        ):
            section.validate()

        # --- cross-section consistency ---
        missing = [tf for tf in self.strategy.mtf.all_timeframes if tf not in self.data.timeframes]
        if missing:
            raise ConfigError(
                f"strategy.mtf references timeframe(s) {missing} that are not downloaded; "
                f"add them to data.timeframes {self.data.timeframes}"
            )
        if self.data.primary_timeframe not in self.strategy.mtf.entry:
            raise ConfigError(
                f"data.primary_timeframe {self.data.primary_timeframe!r} must be one of "
                f"strategy.mtf.entry {self.strategy.mtf.entry}"
            )
        if self.data.warmup_bars < self.indicators.max_period:
            raise ConfigError(
                f"data.warmup_bars ({self.data.warmup_bars}) is below the longest indicator "
                f"lookback ({self.indicators.max_period}); early bars would carry NaN-derived "
                f"signals"
            )
        if self.structure.lookback > self.data.ohlcv_limit - self.data.warmup_bars:
            raise ConfigError(
                "structure.lookback exceeds the tradable window "
                f"(ohlcv_limit - warmup_bars = {self.data.ohlcv_limit - self.data.warmup_bars})"
            )
        if self.data.funding_rate and not self.exchange.is_futures:
            raise ConfigError(
                "data.funding_rate=true requires exchange.market_type=future (spot markets have "
                "no funding). Set funding_rate=false for spot."
            )
        if self.data.open_interest and not self.exchange.is_futures:
            raise ConfigError(
                "data.open_interest=true requires exchange.market_type=future. "
                "Set open_interest=false for spot."
            )
        if self.telegram.enabled and not self.telegram.is_configured:
            raise ConfigError(
                "telegram.enabled=true but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are empty. "
                "Fill them in .env or set telegram.enabled=false."
            )

    # -- mutation -----------------------------------------------------------
    def with_overrides(self, overrides: dict[str, Any]) -> Settings:
        """Return a new ``Settings`` with dotted-path keys replaced.

        Used by the CLI (``--set a.b=1``) and by the optimiser, which needs to
        sweep parameters without mutating the shared instance.

        Args:
            overrides: mapping of ``"section.key"`` to value.

        Returns:
            A freshly validated ``Settings``.
        """
        data = copy.deepcopy(self.raw)
        for path, value in overrides.items():
            parts = path.split(".")
            node: Any = data
            for part in parts[:-1]:
                if not isinstance(node, dict) or part not in node:
                    raise ConfigError(f"override path {path!r} does not exist in the config")
                node = node[part]
            leaf = parts[-1]
            if not isinstance(node, dict) or leaf not in node:
                raise ConfigError(f"override path {path!r} does not exist in the config")
            node[leaf] = _coerce(value, node[leaf])
        return Settings.from_mapping(data)

    def to_dict(self) -> dict[str, Any]:
        """Return the raw mapping, with credentials masked (safe to log)."""
        data = copy.deepcopy(self.raw)
        for section, key in (
            ("exchange", "api_key"), ("exchange", "api_secret"),
            ("telegram", "bot_token"), ("telegram", "chat_id"),
            ("data", "news_api_key"),
        ):
            node = data.get(section)
            if isinstance(node, dict) and node.get(key):
                node[key] = "***"
        return data


def load_settings(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    env_file: str | Path | None = ".env",
) -> Settings:
    """Load, env-expand and validate the configuration file.

    Args:
        path: YAML path. Defaults to ``$QUANTBOT_CONFIG`` or
            ``config/config.yaml``.
        overrides: dotted-path overrides applied after the file is read.
        env_file: ``.env`` file to pre-load into the environment.

    Returns:
        A validated :class:`Settings`.

    Raises:
        ConfigError: if the file is missing, unparsable or inconsistent.
    """
    if env_file is not None:
        _load_dotenv(Path(env_file))

    cfg_path = Path(path or os.environ.get("QUANTBOT_CONFIG") or "config/config.yaml")
    if not cfg_path.is_file():
        raise ConfigError(
            f"config file not found: {cfg_path}. Copy config/config.yaml or set QUANTBOT_CONFIG."
        )
    try:
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse {cfg_path}: {exc}") from exc

    data = _expand_env(loaded)
    if log_level := os.environ.get("QUANTBOT_LOG_LEVEL"):
        data = _deep_update(data, {"logging": {"level": log_level}})

    settings = Settings.from_mapping(data)
    if overrides:
        settings = settings.with_overrides(overrides)
    return settings
