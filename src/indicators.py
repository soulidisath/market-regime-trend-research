"""Trend-indicator implementations used in the research experiments.

Several functions here are Python implementations/adaptations of public indicator
ideas originally encountered in TradingView scripts. Source links are retained in
the relevant docstrings. The purpose of this module is research and comparison,
not to claim original authorship of each indicator formula.
"""

import pandas as pd
import numpy as np
import talib as ta


def ta_linreg(series, length, offset=0):
    """Calculate linear regression using TA-Lib's LINEARREG function."""
    intercept = ta.LINEARREG_INTERCEPT(series, length)
    slope = ta.LINEARREG_SLOPE(series, length)
    return intercept + slope * (length - 1 - offset)


def percentile_nearest_rank_vectorized(series, length, percentile):
    rank = int(np.ceil(percentile / 100 * length)) - 1
    windows = np.lib.stride_tricks.sliding_window_view(series, length)
    sorted_windows = np.sort(windows, axis=1)
    percentiles = sorted_windows[:, rank]
    return np.concatenate([np.full(length - 1, np.nan), percentiles])


def ta_hma(series, length):
    half_length = length // 2
    sqrt_length = int(np.sqrt(length))
    wma_half = ta.WMA(series, half_length)
    wma_full = ta.WMA(series, length)
    return np.array(ta.WMA(2 * wma_half - wma_full, sqrt_length))


def ta_rma(series, length):
    return ta.EMA(series, 2 * length - 1)


def ta_sum(series, length):
    rsum = np.cumsum(series)
    rsum[length:] = rsum[length:] - rsum[:-length]
    rsum[:length] = np.nan
    return rsum


def ta_lowest(series, length):
    lowest = np.lib.stride_tricks.sliding_window_view(series, length).min(axis=1)
    return np.concatenate([np.full(length - 1, np.nan), lowest])


def ta_highest(series, length):
    highest = np.lib.stride_tricks.sliding_window_view(series, length).max(axis=1)
    return np.concatenate([np.full(length - 1, np.nan), highest])


def ta_rising(series, length):
    windows = np.lib.stride_tricks.sliding_window_view(series, length + 1)
    rising = np.all(np.diff(windows) > 0, axis=1)
    return np.concatenate([np.full(length, False), rising])


def trendilo(close, hl2, lsma_length=25, offset=0, smooth=5, hma_length=50):
    """Trendilo LSMA adaptation.

    Source: https://www.tradingview.com/script/Xb4BQ8GS-Trendilo-LSMA-Band-Example/
    """
    src = ta.EMA(hl2, smooth)
    lsma = ta_linreg(src, lsma_length, offset)
    hma = ta_hma(src, hma_length)
    combine = (lsma + hma) / 2
    diff = np.diff(combine, prepend=0)
    signals = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    return pd.Series(signals, index=close.index)


def dema_dmi(close, high, low, len_dema=15, adx_smoothing_len=3, di_len=18):
    """DEMA/DMI adaptation.

    Source: https://www.tradingview.com/script/rzPRTDQu-Dema-DMI-viResearch/
    """
    demah = ta.DEMA(high, len_dema)
    demal = ta.DEMA(low, len_dema)
    u = np.diff(demah, prepend=np.nan)
    d = -np.diff(demal, prepend=np.nan)
    p = np.where((u > d) & (u > 0), u, 0)
    m = np.where((d > u) & (d > 0), d, 0)
    t = ta_rma(ta.TRANGE(high, low, close), di_len)
    plus = 100 * ta_rma(p, di_len) / t
    minus = 100 * ta_rma(m, di_len) / t
    plus = np.nan_to_num(plus)
    minus = np.nan_to_num(minus)
    sum_dm = plus + minus
    adx = 100 * ta_rma(np.abs(plus - minus) / np.where(sum_dm == 0, 1, sum_dm), adx_smoothing_len)
    adx_rising = adx > np.roll(adx, 1)
    dmil = (plus > minus) & adx_rising
    dmis = minus > plus
    signal = np.where(dmil & ~dmis, 1, np.where(dmis, -1, np.nan))
    return pd.Series(signal, index=close.index).ffill().fillna(0)


def kama(close, fast_period=7, slow_period=19, er_period=8, norm_period=50):
    """Normalized KAMA adaptation.

    Source: https://www.tradingview.com/script/OwtiIzT3-Normalized-KAMA-Oscillator-Ikke-Omar/
    """
    change = np.abs(close - np.roll(close, er_period))
    diff = abs(np.diff(close, prepend=close.to_numpy()[0]))
    volatility = ta_sum(diff, er_period)
    er = change / volatility
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    ema_fast = ta.EMA(close, fast_period)
    kama_values = ema_fast + sc * (close - ema_fast)
    lowest = ta_lowest(kama_values, norm_period)
    highest = ta_highest(kama_values, norm_period)
    normalized = (kama_values - lowest) / (highest - lowest) - 0.5
    signals = np.where(normalized > 0, 1, np.where(normalized < 0, -1, 0))
    return pd.Series(signals, index=close.index)


def stc(close, stc_length=12, fast_length=26, slow_length=50, factor=0.5):
    """Schaff Trend Cycle adaptation.

    Source: https://www.tradingview.com/script/WhRRThMI-STC-Indicator-A-Better-MACD-SHK/
    """
    macd = ta.EMA(close, fast_length) - ta.EMA(close, slow_length)
    lowest_k = ta_lowest(macd, stc_length)
    highest_k = ta_highest(macd, stc_length)
    range_k = highest_k - lowest_k
    percent_k = np.where(range_k > 0, (macd - lowest_k) / range_k * 100, 0)

    def smooth(series):
        result = np.zeros(len(series))
        for i in range(1, len(series)):
            result[i] = result[i - 1] + factor * (series[i] - result[i - 1])
        return result

    smoothed_k = smooth(percent_k)
    lowest_d = ta_lowest(smoothed_k, stc_length)
    highest_d = ta_highest(smoothed_k, stc_length)
    range_d = np.where(highest_d - lowest_d == 0, 1e-10, highest_d - lowest_d)
    percent_d = np.where(range_d > 0, (smoothed_k - lowest_d) / range_d * 100, 0)
    stc_values = np.round(smooth(percent_d), 10)
    return pd.Series(np.where(ta_rising(stc_values, 1), 1, -1), index=close.index)


def vii_stop(close, high, low, length=12, multiplier=2.8):
    """Vii Stop adaptation.

    Source: https://www.tradingview.com/script/cT7O5AAq-vii-Stop/
    """
    close_values = np.asarray(close)
    atr = ta.ATR(high.to_numpy(), low.to_numpy(), close_values, length)
    signals = np.zeros(len(close_values))
    stop = np.zeros(len(close_values))
    max_val = close_values[0]
    min_val = close_values[0]
    uptrend = True

    for i in range(1, len(close_values)):
        if np.isnan(close_values[i]):
            max_val = min_val = 0
            stop[i] = 0
            uptrend = True
            continue
        atr_m = atr[i] * multiplier
        if uptrend:
            max_val = max(max_val, close_values[i])
            stop[i] = max(stop[i - 1], max_val - atr_m)
            if close_values[i] < stop[i]:
                uptrend = False
                min_val = close_values[i]
                stop[i] = min_val + atr_m
        else:
            min_val = min(min_val, close_values[i])
            stop[i] = min(stop[i - 1], min_val + atr_m)
            if close_values[i] > stop[i]:
                uptrend = True
                max_val = close_values[i]
                stop[i] = max_val - atr_m
        signals[i] = 1 if uptrend else -1
    return pd.Series(signals, index=high.index)


def rti(close, trend_data_count=100, trend_sensitivity_percentage=95, signal_length=20, long_threshold=50, short_threshold=50):
    """Relative Trend Index adaptation.

    Source: https://www.tradingview.com/script/VwmUNNwp-Relative-Trend-Index-RTI-by-Zeiierman/
    """
    stdev = ta.STDDEV(close, 2)
    upper_trend = close + stdev
    lower_trend = close - stdev
    upper_windows = np.lib.stride_tricks.sliding_window_view(upper_trend, trend_data_count)
    lower_windows = np.lib.stride_tricks.sliding_window_view(lower_trend, trend_data_count)
    upper_sorted = np.sort(upper_windows, axis=1)
    lower_sorted = np.sort(lower_windows, axis=1)
    upper_index = int(np.round(trend_sensitivity_percentage / 100 * trend_data_count)) - 1
    lower_index = int(np.round((100 - trend_sensitivity_percentage) / 100 * trend_data_count)) - 1
    upper = upper_sorted[:, upper_index]
    lower = lower_sorted[:, lower_index]
    values = ((close[trend_data_count - 1:] - lower) / (upper - lower)) * 100
    full = np.concatenate([np.full(trend_data_count - 1, np.nan), values])
    ma = ta.EMA(full, signal_length)
    return pd.Series(np.where(ma > long_threshold, 1, np.where(ma < short_threshold, -1, np.nan)), index=close.index).ffill()


def lrsi(close, alpha=0.2, long_threshold=0.5, short_threshold=0.5):
    length = len(close)
    l0 = np.zeros(length)
    l1 = np.zeros(length)
    l2 = np.zeros(length)
    l3 = np.zeros(length)
    for i in range(1, length):
        if np.isnan(close.iloc[i]):
            l0[i] = l1[i] = l2[i] = l3[i] = 0
            continue
        l0[i] = alpha * close.iloc[i] + (1 - alpha) * l0[i - 1]
        l1[i] = -(1 - alpha) * l0[i] + l0[i - 1] + (1 - alpha) * l1[i - 1]
        l2[i] = -(1 - alpha) * l1[i] + l1[i - 1] + (1 - alpha) * l2[i - 1]
        l3[i] = -(1 - alpha) * l2[i] + l2[i - 1] + (1 - alpha) * l3[i - 1]
    cu = ((l0 >= l1) * (l0 - l1)) + ((l1 >= l2) * (l1 - l2)) + ((l2 >= l3) * (l2 - l3))
    cd = ((l0 < l1) * (l1 - l0)) + ((l1 < l2) * (l2 - l1)) + ((l2 < l3) * (l3 - l2))
    value = cu / (cu + cd + 1e-10)
    signal = np.where(value > long_threshold, 1, np.where(value < short_threshold, -1, np.nan))
    return pd.Series(signal, index=close.index).ffill()


def lsma_atr(close, high, low, len_lsma=75, len_atr=20, atr_multiplier=1.1):
    lsma = ta_linreg(close, len_lsma)
    atr = ta.ATR(high, low, close, len_atr)
    smoothed_atr = ta.SMA(atr, len_atr)
    slope = np.diff(lsma, prepend=np.nan)
    uptrend = (slope > 0) & (close > lsma + atr_multiplier * smoothed_atr)
    downtrend = (slope < 0) & (close < lsma - atr_multiplier * smoothed_atr)
    result = np.full_like(close, np.nan)
    result[uptrend & ~downtrend] = 1
    result[downtrend] = -1
    return pd.Series(result, index=close.index).ffill()


def hullloop(close, length=5, a=1, b=50):
    hma_values = ta_hma(close, length)
    shift_indices = np.arange(a, b + 1)
    shifted = np.array([np.roll(hma_values, shift) for shift in shift_indices]).T
    score = np.sum(np.where(hma_values[:, np.newaxis] > shifted, 1, -1), axis=1)
    long_condition = score > 40
    short_condition = score < -10
    result = np.where(long_condition & ~short_condition, 1, np.where(short_condition, -1, np.nan))
    return pd.Series(result, index=close.index).ffill()


def aamdmom(close, len_momentum=4, len_percentile=35):
    momentum = ta.ROC(close, len_momentum)
    percentile_75 = percentile_nearest_rank_vectorized(close, len_percentile, 75)
    percentile_25 = percentile_nearest_rank_vectorized(close, len_percentile, 25)
    score = np.where(
        (momentum > 0) & (close > percentile_75),
        1,
        np.where((momentum < 0) & (close < percentile_25), -1, np.nan),
    )
    return pd.Series(score, index=close.index).ffill()
