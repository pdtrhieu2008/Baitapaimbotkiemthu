"""The indicator engine: one call turns an OHLCV frame into a feature frame.

Everything downstream (price action, structure, scoring, backtest) consumes the
frame produced here, so there is exactly one place where indicator periods are
read from the configuration and exactly one definition of each column name.
"""

from __future__ import annotations

import pandas as pd

from config.settings import IndicatorConfig
from indicators import momentum, trend, volatility, volume
from indicators.base import series_slope, validate_ohlcv
from indicators.levels import pivot_points
from indicators.volume import VolumeProfile
from utils.logger import get_logger

__all__ = ["IndicatorEngine"]

_log = get_logger("indicators.engine")


class IndicatorEngine:
    """Compute the full indicator set for one timeframe.

    The engine is stateless with respect to market data — it takes a frame and
    returns an enriched copy — so the same instance is safely reused across
    symbols, timeframes, the live loop and the backtester. That reuse is what
    guarantees the backtest measures the same features the live bot sees.

    Args:
        cfg: the ``indicators`` configuration section.
    """

    def __init__(self, cfg: IndicatorConfig) -> None:
        self.cfg = cfg

    @property
    def required_history(self) -> int:
        """Minimum number of bars before every column is non-NaN.

        A little headroom is added because Wilder-smoothed series (ADX in
        particular) need roughly two periods to shed their seeding bias.
        """
        return self.cfg.max_period + 2 * self.cfg.adx_period + 10

    def compute(self, df: pd.DataFrame, *, with_pivots: bool = True) -> pd.DataFrame:
        """Return ``df`` plus every indicator column.

        Args:
            df: OHLCV frame indexed by bar-open timestamp, ascending.
            with_pivots: compute higher-timeframe pivot levels. Disabled for
                timeframes at or above the pivot period, where they would be
                degenerate.

        Returns:
            A new frame: the original columns followed by the indicators.

        Raises:
            ValueError: if the frame is malformed (see
                :func:`indicators.base.validate_ohlcv`).
        """
        validate_ohlcv(df, min_rows=2)
        cfg = self.cfg
        out = df.copy()
        close = out["close"]

        if len(out) < self.required_history:
            # Not fatal: the caller may legitimately be warming up. The strategy
            # refuses to act on NaN features, so this only costs a log line.
            _log.debug(
                "only %d bars available, %d recommended for fully-seeded indicators",
                len(out), self.required_history,
            )

        # --- trend -------------------------------------------------------
        out["ema_fast"] = trend.ema(close, cfg.ema_fast)
        out["ema_slow"] = trend.ema(close, cfg.ema_slow)
        out["ema_trend"] = trend.ema(close, cfg.ema_trend)
        out["sma"] = trend.sma(close, cfg.sma_period)
        out["ema_slow_slope"] = series_slope(out["ema_slow"], max(5, cfg.ema_fast // 2))
        out["vwap"] = trend.vwap(out)
        out = out.join(trend.adx(out, cfg.adx_period))
        out = out.join(trend.supertrend(out, cfg.supertrend_period, cfg.supertrend_multiplier))
        out = out.join(
            trend.ichimoku(
                out,
                cfg.ichimoku_tenkan,
                cfg.ichimoku_kijun,
                cfg.ichimoku_senkou_b,
                cfg.ichimoku_displacement,
            )
        )

        # --- momentum ----------------------------------------------------
        out["rsi"] = momentum.rsi(close, cfg.rsi_period)
        out = out.join(momentum.macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal))
        out = out.join(
            momentum.stochastic(out, cfg.stoch_rsi_period, cfg.stoch_rsi_k, cfg.stoch_rsi_d)
        )
        out = out.join(
            momentum.stoch_rsi(
                close, cfg.rsi_period, cfg.stoch_rsi_period, cfg.stoch_rsi_k, cfg.stoch_rsi_d
            )
        )

        # --- volatility --------------------------------------------------
        out["atr"] = volatility.atr(out, cfg.atr_period)
        out["natr"] = volatility.natr(out, cfg.atr_period)
        # Median ATR is the reference the volatility-anomaly guard compares
        # against: "is the market 3x more violent than it normally is?"
        out["atr_median"] = out["atr"].rolling(cfg.atr_period * 5, min_periods=cfg.atr_period).median()
        bb = volatility.bollinger_bands(close, cfg.bb_period, cfg.bb_std)
        kc = volatility.keltner_channel(
            out, cfg.keltner_period, cfg.atr_period, cfg.keltner_multiplier
        )
        out = out.join(bb).join(kc)
        out = out.join(volatility.donchian_channel(out, cfg.donchian_period))
        out["squeeze_on"] = volatility.squeeze_on(bb, kc)
        # A squeeze that just ended: compression released, expansion likely.
        out["squeeze_released"] = out["squeeze_on"].shift(1).fillna(False) & ~out["squeeze_on"]

        # --- volume ------------------------------------------------------
        out["obv"] = volume.obv(out)
        out["obv_ema"] = trend.ema(out["obv"], cfg.obv_ema_period)
        out["cmf"] = volume.cmf(out, cfg.cmf_period)
        out["mfi"] = volume.mfi(out, cfg.mfi_period)
        out = out.join(volume.relative_volume(out, cfg.volume_sma_period))
        out = out.join(volume.effort_split(out))

        # --- horizontal levels -------------------------------------------
        if with_pivots:
            out = out.join(pivot_points(out, cfg.pivot_method, cfg.pivot_timeframe))

        return out

    def profile(self, df: pd.DataFrame) -> VolumeProfile | None:
        """Volume profile for the most recent window.

        Kept out of :meth:`compute` because it collapses a window into a few
        scalars instead of producing per-bar columns.
        """
        return volume.volume_profile(
            df, bins=self.cfg.volume_profile_bins, lookback=self.cfg.volume_profile_lookback
        )
