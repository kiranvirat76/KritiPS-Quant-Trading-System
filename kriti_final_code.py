import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.4f}".format

# =====================================================================
# STRATEGY 1 (B) - MOMENTUM RANKING (FROM sr1.py)
# =====================================================================


class Config_SR1:
    DATA_PATH = "/Users/vokirankumar/kriti quant/training_data.parquet"
    INITIAL_CAPITAL = 5_000_000
    MAX_STOCKS = 50
    MAX_WEIGHT = 0.05
    COST_BASIS = 0.00268

    TOP_K = 45
    EXIT_RANK = 55

    EMA_FAST = 50
    EMA_SLOW = 200
    ENTRY_SCALE = 1

    MIN_HOLD_DAYS = 42
    k = 3
    max_stop = 0.20

    BEAR_REGIME_THRESHOLD = 0.5
    BEAR_REGIME_SCALE = 0.9

    USE_TRAILING_STOP = True
    VOL_ADJUST_SIZING = True

    ENFORCE_MAX_STOCKS = True


def generate_features_and_signals_SR1(df, cfg):
    print("Generating features and signals...")

    df = df.sort_values(["fid", "tradedate"])
    g = df.groupby("fid")["close"]

    df["ret_1m"] = g.pct_change(21)
    df["ret_6m"] = g.pct_change(126)
    df["ret_12m"] = g.pct_change(252)

    df["vol_20d"] = g.transform(lambda x: x.pct_change().rolling(20).std())

    ema_200 = g.transform(lambda x: x.ewm(span=cfg.EMA_SLOW, adjust=False).mean())
    df["trend_strength"] = (df["close"] / ema_200) - 1

    df["is_bull_200"] = (df["close"] > ema_200).astype(float)
    daily_regime = (
        df.groupby("tradedate")["is_bull_200"].mean().to_frame("bull_frac_200")
    )
    daily_regime["bear_regime"] = (
        (1 - daily_regime["bull_frac_200"]).shift(1) > cfg.BEAR_REGIME_THRESHOLD
    ).astype(int)
    df = df.merge(
        daily_regime[["bear_regime"]], left_on="tradedate", right_index=True, how="left"
    )

    ema_50 = g.transform(lambda x: x.ewm(span=cfg.EMA_FAST, adjust=False).mean())
    df["is_healthy"] = (df["close"] > ema_50).astype(float)
    daily_breadth = df.groupby("tradedate")["is_healthy"].mean().to_frame("mkt_breadth")
    daily_breadth["market_exposure"] = daily_breadth["mkt_breadth"].shift(1).clip(0, 1)
    df = df.merge(
        daily_breadth[["market_exposure"]],
        left_on="tradedate",
        right_index=True,
        how="left",
    )
    daily_breadth["crash_regime"] = (
        daily_breadth["mkt_breadth"].shift(1) < 0.30
    ).astype(int)

    df = df.merge(
        daily_breadth[["crash_regime"]],
        left_on="tradedate",
        right_index=True,
        how="left",
    )

    rank_cols = ["ret_1m", "ret_6m", "ret_12m", "trend_strength", "vol_20d"]
    for col in rank_cols:
        df[f"rank_{col}"] = df.groupby("tradedate")[col].rank(pct=True)

    df["score"] = (
        0.40 * df["rank_ret_12m"]
        + 0.30 * df["rank_ret_6m"]
        + 0.10 * df["rank_ret_1m"]
        + 0.10 * df["rank_trend_strength"]
        + 0.10 * (1 - df["rank_vol_20d"])
    )

    df["final_rank"] = df.groupby("tradedate")["score"].rank(
        ascending=False, method="first"
    )

    df["buy_signal"] = (
        (df["final_rank"] <= cfg.TOP_K)
        & (df["trend_strength"] > 0)
        & (df["ret_12m"] > 0)
    ).astype(int)

    df["sell_signal"] = (
        (df["trend_strength"] <= 0)
        | (df["final_rank"] > cfg.EXIT_RANK)
        | (df["ret_12m"] < 0)
    ).astype(int)

    df = df.dropna(subset=["ret_12m", "vol_20d", "trend_strength", "market_exposure"])
    return df


class Backtester_SR1:
    def __init__(self, cfg, initial_capital):
        self.cfg = cfg
        self.initial_capital = initial_capital
        self.turnover = {"buy": 0.0, "sell": 0.0}
        self.position_history = []

    def load_data(self):
        print("Loading data...")
        df = pd.read_parquet(self.cfg.DATA_PATH)
        df.columns = df.columns.str.lower().str.strip()
        df["ohlc_avg"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

        df = df.rename(columns={"symbol": "fid", "date": "tradedate"})
        df["tradedate"] = pd.to_datetime(df["tradedate"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["fid", "tradedate", "close"])
        df = df[df["close"] > 0]
        df = df.sort_values(["fid", "tradedate"]).reset_index(drop=True)
        print(f"Loaded {len(df)} rows.")
        return df

    def run(self):
        df = self.load_data()
        df = generate_features_and_signals_SR1(df, self.cfg)

        all_dates = sorted(df["tradedate"].unique())

        cash = self.initial_capital
        holdings = {}
        pending_exit = set()
        hold_days = {}
        entry_price = {}
        peak_price = {}
        history = []

        week_ends = set(
            pd.Series(all_dates)
            .groupby(pd.Series(all_dates).dt.to_period("W"))
            .max()
            .values
        )

        print(
            f"Running ASYMMETRIC backtest: Daily Sells, Weekly Buys ({len(week_ends)} weeks)..."
        )

        for i in range(len(all_dates) - 1):
            curr_date = all_dates[i]
            exec_date = all_dates[i + 1]

            sig = df[df["tradedate"] == curr_date].set_index("fid")
            exe = df[df["tradedate"] == exec_date].set_index("fid")

            if sig.empty:
                continue

            for fid in list(holdings.keys()):
                if fid not in exe.index:

                    if fid in sig.index:
                        last_price = sig.at[fid, "ohlc_avg"]
                    else:
                        last_price = entry_price.get(fid, 0)

                    shares = holdings[fid]
                    val = shares * last_price

                    cash += val * (1 - self.cfg.COST_BASIS)
                    self.turnover["sell"] += val

                    self.position_history.append(
                        {
                            "date": curr_date,
                            "action": "delist_exit",
                            "fid": fid,
                            "shares": shares,
                            "price": last_price,
                            "value": val,
                            "hold_days": hold_days.get(fid, None),
                        }
                    )

                    del holdings[fid]
                    hold_days.pop(fid, None)
                    entry_price.pop(fid, None)
                    peak_price.pop(fid, None)

            for fid in list(hold_days):
                hold_days[fid] += 1
                if self.cfg.USE_TRAILING_STOP and fid in sig.index:
                    current_px = sig.at[fid, "close"]
                    peak_price[fid] = max(peak_price.get(fid, current_px), current_px)

            for fid in list(holdings.keys()):
                if fid not in sig.index or fid not in exe.index:
                    continue

                try:
                    vol = sig.at[fid, "vol_20d"]
                    stop_pct = min(self.cfg.k * vol, self.cfg.max_stop)
                    signal_price = sig.at[fid, "close"]

                    if self.cfg.USE_TRAILING_STOP:
                        stop_level = peak_price[fid] * (1 - stop_pct)
                    else:
                        stop_level = entry_price[fid] * (1 - stop_pct)

                    stop_hit = signal_price < stop_level

                    if stop_hit:
                        pending_exit.add(fid)
                        continue
                except (KeyError, TypeError):
                    continue

            for pid in list(pending_exit):
                if pid not in holdings:
                    pending_exit.discard(pid)
                    continue
                if pid not in exe.index:
                    continue

                prev_close = sig.at[pid, "close"]
                lower_circuit_price = prev_close * 0.80

                is_lower_circuit = (
                    exe.at[pid, "low"] <= lower_circuit_price
                    and exe.at[pid, "high"] == exe.at[pid, "low"]
                )

                if is_lower_circuit:
                    continue

                px = exe.at[pid, "ohlc_avg"]
                shares = holdings.get(pid, 0)
                if shares <= 0:
                    pending_exit.discard(pid)
                    continue

                val = shares * px
                cash += val * (1 - self.cfg.COST_BASIS)
                self.turnover["sell"] += val

                self.position_history.append(
                    {
                        "date": exec_date,
                        "action": "sell_full",
                        "fid": pid,
                        "shares": shares,
                        "price": px,
                        "value": val,
                        "hold_days": hold_days.get(pid, None),
                    }
                )

                del holdings[pid]
                hold_days.pop(pid, None)
                entry_price.pop(pid, None)
                peak_price.pop(pid, None)
                pending_exit.discard(pid)

            if i % 5 != 0:
                nav = cash
                for h in holdings:
                    if h in exe.index:
                        px = exe.at[h, "ohlc_avg"]
                    elif h in sig.index:
                        px = sig.at[h, "close"]
                    else:
                        px = entry_price.get(h, 0)
                    nav += holdings[h] * px
                history.append(
                    {
                        "date": exec_date,
                        "nav": nav,
                        "cash": cash,
                        "num_holdings": len(holdings),
                    }
                )
                continue

            for fid in list(holdings.keys()):
                if fid not in sig.index or fid not in exe.index:
                    continue
                if hold_days[fid] < self.cfg.MIN_HOLD_DAYS:
                    continue
                try:
                    if (
                        sig.loc[fid, "sell_signal"] == 1
                        and sig.loc[fid, "bear_regime"] == 1
                    ):
                        pending_exit.add(fid)
                except (KeyError, TypeError):
                    continue

            equity = cash + sum(
                holdings[h] * exe.at[h, "ohlc_avg"] for h in holdings if h in exe.index
            )

            selected = (
                sig[sig["buy_signal"] == 1]
                .sort_values("final_rank")
                .head(self.cfg.TOP_K)
            )

            if not sig.empty:
                if sig["crash_regime"].iloc[0] == 1:
                    max_stocks_dynamic = 5
                elif sig["bear_regime"].iloc[0] == 1:
                    max_stocks_dynamic = 15
                else:
                    max_stocks_dynamic = self.cfg.MAX_STOCKS
            else:
                max_stocks_dynamic = self.cfg.MAX_STOCKS

            available_slots = max_stocks_dynamic - len(holdings)

            if available_slots > 0:
                added_this_week = 0
                for fid, row in selected.iterrows():
                    if (
                        self.cfg.ENFORCE_MAX_STOCKS
                        and added_this_week >= available_slots
                    ):
                        break

                    if fid in holdings or fid not in exe.index:
                        continue

                    try:
                        base_alloc = equity * self.cfg.MAX_WEIGHT
                        rank_pct = 1 - (row["final_rank"] / self.cfg.TOP_K)
                        rank_scalar = 0.5 + rank_pct

                        target_alloc = base_alloc * rank_scalar * row["market_exposure"]

                        if row["bear_regime"] == 1:
                            target_alloc *= self.cfg.BEAR_REGIME_SCALE

                        if self.cfg.VOL_ADJUST_SIZING:
                            vol_scalar = 1 / (1 + row["vol_20d"] * 10)
                            vol_scalar = np.clip(vol_scalar, 0.5, 1.5)
                            target_alloc *= vol_scalar

                        target_alloc = min(target_alloc, equity * self.cfg.MAX_WEIGHT)
                        alloc = target_alloc * self.cfg.ENTRY_SCALE

                        px = exe.at[fid, "ohlc_avg"]
                        shares = int(alloc / (px * (1 + self.cfg.COST_BASIS)))
                        if shares <= 0:
                            continue

                        cost = shares * px * (1 + self.cfg.COST_BASIS)
                        if cash >= cost:
                            cash -= cost
                            self.turnover["buy"] += cost
                            holdings[fid] = holdings.get(fid, 0) + shares
                            hold_days.setdefault(fid, 0)
                            added_this_week += 1

                            if fid not in entry_price:
                                entry_price[fid] = px
                                peak_price[fid] = px

                            self.position_history.append(
                                {
                                    "date": exec_date,
                                    "action": "buy",
                                    "fid": fid,
                                    "shares": shares,
                                    "price": px,
                                    "value": cost,
                                    "portfolio_pct": cost / equity,
                                }
                            )
                    except (KeyError, TypeError):
                        continue

            nav = cash
            for h in holdings:
                if h in exe.index:
                    px = exe.at[h, "ohlc_avg"]
                elif h in sig.index:
                    px = sig.at[h, "close"]
                else:
                    px = entry_price.get(h, 0)
                nav += holdings[h] * px

            history.append(
                {
                    "date": exec_date,
                    "nav": nav,
                    "cash": cash,
                    "num_holdings": len(holdings),
                }
            )

        self.results = pd.DataFrame(history).set_index("date")
        print(f"Backtest complete. Final NAV: ₹{self.results['nav'].iloc[-1]:,.0f}")


# =====================================================================
# STRATEGY 2 (A) - KALMAN 50/50 SPLIT (FROM sr2.py)
# =====================================================================


class Config_SR2:
    DATA_PATH = "/Users/vokirankumar/kriti quant/training_data.parquet"
    BENCHMARK_PATH = "/Users/vokirankumar/kriti quant/indexes.xlsx"
    INITIAL_CAPITAL = 5_000_000
    TRANS_COST = 0.00268

    STOP_LOSS = 0.20
    ENABLE_PROFIT_TARGET = False
    PROFIT_TARGET = 0.05

    MOM_LOOKBACK = 21
    SKIP_RECENT = 10
    REBALANCE_EVERY = 5

    TOP_N_BUY = 5
    TOP_N_HOLD = 100
    MAX_POSITIONS = 100

    WEIGHT_MOMENTUM = 0.50
    WEIGHT_QUALITY = 0.20
    WEIGHT_TREND = 0.15
    WEIGHT_CONSISTENCY = 0.15
    WEIGHT_ADX = 0.00
    WEIGHT_MACD = 0.00


cfg_SR2 = Config_SR2()


class KalmanFilter1D:
    @staticmethod
    def smooth(prices, R_val=0.5, Q_val=1e-5):
        n = len(prices)
        xhat = np.zeros(n)
        P = np.zeros(n)
        xhat[0] = prices[0]
        P[0] = 1.0
        for k in range(1, n):
            xhatminus = xhat[k - 1]
            Pminus = P[k - 1] + Q_val
            K = Pminus / (Pminus + R_val)
            xhat[k] = xhatminus + K * (prices[k] - xhatminus)
            P[k] = (1 - K) * Pminus
        return xhat


def calculate_features_per_stock(fid, df):
    df = df.sort_values("tradedate").copy()
    try:
        df["kf_price"] = KalmanFilter1D.smooth(df["close"].values)
    except:
        df["kf_price"] = df["close"]

    lag = cfg_SR2.MOM_LOOKBACK + cfg_SR2.SKIP_RECENT
    df["momentum"] = df["kf_price"].pct_change(lag)
    df["vol"] = df["close"].pct_change().rolling(60).std()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["dist_kf"] = (df["close"] / df["kf_price"]) - 1

    df["ohlc_avg"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0

    df["is_lower_circuit"] = (
        (df["open"] == df["low"]) & (df["low"] == df["close"]) & (df["open"] > 0)
    )
    return df


def calculate_transparent_score(df):
    df = df.copy()
    total_weight = (
        cfg_SR2.WEIGHT_MOMENTUM
        + cfg_SR2.WEIGHT_QUALITY
        + cfg_SR2.WEIGHT_TREND
        + cfg_SR2.WEIGHT_CONSISTENCY
        + cfg_SR2.WEIGHT_ADX
        + cfg_SR2.WEIGHT_MACD
    )
    if total_weight == 0:
        total_weight = 1.0

    w_mom = cfg_SR2.WEIGHT_MOMENTUM / total_weight * 100
    w_qual = cfg_SR2.WEIGHT_QUALITY / total_weight * 100
    w_trend = cfg_SR2.WEIGHT_TREND / total_weight * 100
    w_cons = cfg_SR2.WEIGHT_CONSISTENCY / total_weight * 100

    def truncate_outliers(series, lower_quantile=0.01, upper_quantile=0.99):
        lower_limit = series.quantile(lower_quantile)
        upper_limit = series.quantile(upper_quantile)
        return series.clip(lower=lower_limit, upper=upper_limit)

    df["momentum"] = truncate_outliers(df["momentum"])
    df["mom_percentile"] = df["momentum"].rank(pct=True)
    df["momentum_score"] = df["mom_percentile"] * w_mom

    df["rsi_quality"] = 1 - abs(df["rsi"] - 55) / 55
    df["rsi_quality"] = df["rsi_quality"].clip(0, 1)
    df["quality_score"] = df["rsi_quality"] * w_qual

    df["dist_kf"] = truncate_outliers(df["dist_kf"])
    df["trend_quality"] = 1 / (1 + abs(df["dist_kf"]) * 10)
    df["trend_score"] = df["trend_quality"] * w_trend

    df["vol"] = truncate_outliers(df["vol"])
    df["vol_percentile"] = df["vol"].rank(pct=True)
    df["consistency_score"] = (1 - df["vol_percentile"]) * w_cons

    df["composite_score"] = (
        df["momentum_score"]
        + df["quality_score"]
        + df["trend_score"]
        + df["consistency_score"]
    )
    return df


def load_data_SR2():
    print(" Loading Data...")
    try:
        df = pd.read_parquet(cfg_SR2.DATA_PATH)
    except FileNotFoundError:
        print(f"Error: {cfg_SR2.DATA_PATH} not found.")
        exit()
    df.columns = df.columns.str.lower().str.strip()
    df["tradedate"] = pd.to_datetime(df["tradedate"]).dt.normalize()
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close", "fid"])
    unique_fids = df["fid"].unique()
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(calculate_features_per_stock)(fid, df[df["fid"] == fid])
        for fid in unique_fids
    )
    return pd.concat(results, ignore_index=True)


def get_benchmark_df_SR2():
    if not os.path.exists(cfg_SR2.BENCHMARK_PATH):
        return None
    try:
        df = pd.read_excel(cfg_SR2.BENCHMARK_PATH)
        df.columns = df.columns.str.lower().str.strip()
        if "index_name" in df.columns:
            df = df[df["index_name"] == "NSE500"]
        df["tradedate"] = pd.to_datetime(df["tradedate"]).dt.normalize()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["tradedate", "close"])
        df = df.sort_values("tradedate")
        df = df.drop_duplicates(subset=["tradedate"], keep="last")
        df = df.set_index("tradedate")
        df["ma200"] = df["close"].rolling(200).mean()
        return df
    except:
        return None


def load_benchmark_aligned_SR2(strategy_dates):
    df = get_benchmark_df_SR2()
    if df is None:
        return pd.Series(index=strategy_dates, data=100)
    bench_series = df["close"]
    aligned = bench_series.reindex(strategy_dates)
    aligned = aligned.fillna(method="ffill").fillna(method="bfill")
    return aligned


def run_backtest_SR2(df, initial_capital):
    print("\n" + "=" * 70)
    print(" STRATEGY: 50/50 SPLIT (AGGRESSIVE + DEFENSIVE)")
    print("=" * 70)
    print("   • 50% Capital -> Strategy 1: Equal Weight (Always Invested)")
    print("   • 50% Capital -> Strategy 2: Inv Vol Weight (Regime Filtered)")

    bench_df = get_benchmark_df_SR2()
    if bench_df is not None:
        print("Regime Filter: ACTIVE (Only applies to Strategy 2)")
    else:
        print(" Regime Filter: INACTIVE (Missing Benchmark)")

    print("=" * 70)

    df = df.sort_values("tradedate")
    unique_dates = pd.Series(df["tradedate"].unique()).sort_values().tolist()
    data_by_date = {date: group for date, group in df.groupby("tradedate")}

    positions = {}
    cash = initial_capital
    results = []
    trades_log = {}
    pending_sells = set()
    total_trades = 0
    stop_loss_exits = 0
    circuit_breaker_delays = 0
    rebalance_exits = 0

    start_idx = 0
    pbar = tqdm(range(start_idx, len(unique_dates) - 1), desc="Simulating")

    for i in pbar:
        curr_date = unique_dates[i]
        exec_date = unique_dates[i + 1]

        curr_df = data_by_date.get(curr_date)
        exec_df = data_by_date.get(exec_date)
        if curr_df is None or exec_df is None:
            continue

        curr_close_map = dict(zip(curr_df["fid"], curr_df["close"]))
        exec_low_map = dict(zip(exec_df["fid"], exec_df["low"]))
        exec_high_map = dict(zip(exec_df["fid"], exec_df["high"]))

        def is_lower_circuit(fid):
            prev_close = curr_close_map.get(fid)
            low = exec_low_map.get(fid)
            high = exec_high_map.get(fid)

            if prev_close is None or low is None or high is None:
                return False

            lower_price = prev_close * 0.80

            return (low <= lower_price) and (high == low)

        curr_closing_prices = dict(zip(curr_df["fid"], curr_df["close"]))

        exec_fids = set(exec_df["fid"].unique())

        for fid in list(positions.keys()):

            if fid not in exec_fids:

                pos = positions[fid]

                if fid in curr_df["fid"].values:
                    last_price = curr_df.loc[
                        curr_df["fid"] == fid, "ohlc_avg"
                    ].values[0]
                else:
                    last_price = pos.get("entry_price", 0)

                shares = pos["shares"]

                proceeds = shares * last_price * (1 - cfg_SR2.TRANS_COST)

                cash += proceeds
                positions.pop(fid)

                print(f"️ Delisting Exit (SR2): {fid} at {last_price:.2f}")
        exec_prices = dict(zip(exec_df["fid"], exec_df["ohlc_avg"]))
        marking_prices = dict(zip(exec_df["fid"], exec_df["close"]))

        day_turnover = 0.0


        strat2_weight = 0.50

        if bench_df is not None:
            if curr_date in bench_df.index:
                bench_row = bench_df.loc[curr_date]
                if pd.notna(bench_row["ma200"]):
                    if bench_row["close"] < bench_row["ma200"]:
                        strat2_weight = 0.25
                    else:
                        strat2_weight = 0.50

        for fid, pos in list(positions.items()):
            closing_price = curr_closing_prices.get(fid, pos.get("last_price", 0))

            if "highest_price" not in pos:
                pos["highest_price"] = pos.get("entry_price", closing_price)

            if closing_price > pos["highest_price"]:
                pos["highest_price"] = closing_price

            drawdown = (closing_price - pos["highest_price"]) / pos["highest_price"]

            if drawdown <= -cfg_SR2.STOP_LOSS and fid not in pending_sells:
                pending_sells.add(fid)
                stop_loss_exits += 1

        target_alloc = None
        if i % cfg_SR2.REBALANCE_EVERY == 0:
            candidates = curr_df.dropna(
                subset=["momentum", "rsi", "dist_kf", "vol"]
            ).copy()
            candidates = candidates[candidates["momentum"] > 0]

            if not candidates.empty:
                candidates = calculate_transparent_score(candidates)
                candidates = candidates.sort_values("composite_score", ascending=False)
                candidates["rank"] = range(1, len(candidates) + 1)

                held = list(positions.keys())
                final_portfolio = candidates[
                    (
                        candidates["fid"].isin(held)
                        & (candidates["rank"] <= cfg_SR2.TOP_N_HOLD)
                    )
                    | (candidates["rank"] <= cfg_SR2.TOP_N_BUY)
                ].drop_duplicates(subset=["fid"])

                if len(final_portfolio) > cfg_SR2.MAX_POSITIONS:
                    final_portfolio = final_portfolio.sort_values("rank").head(
                        cfg_SR2.MAX_POSITIONS
                    )

                if not final_portfolio.empty:
                    target_alloc = {}

                    w1 = 0.50 / len(final_portfolio)
                    for _, r in final_portfolio.iterrows():
                        target_alloc[r["fid"]] = target_alloc.get(r["fid"], 0) + w1

                    final_portfolio["inv_vol"] = 1 / (final_portfolio["vol"] + 1e-6)
                    tot_vol = final_portfolio["inv_vol"].sum()

                    for _, r in final_portfolio.iterrows():
                        w2_raw = r["inv_vol"] / tot_vol
                        w2 = w2_raw * strat2_weight
                        target_alloc[r["fid"]] = target_alloc.get(r["fid"], 0) + w2

        for fid in list(pending_sells):
            if fid not in positions:
                pending_sells.remove(fid)
                continue

            if is_lower_circuit(fid):
                circuit_breaker_delays += 1
                continue

            pos = positions[fid]
            ep = exec_prices.get(fid)

            if ep and ep > 0:
                proceeds = pos["shares"] * ep * (1 - cfg_SR2.TRANS_COST)
                cash += proceeds
                positions.pop(fid)

            pending_sells.remove(fid)

        equity = cash
        for fid, pos in positions.items():
            price = marking_prices.get(fid, pos.get("last_price", 0))
            equity += pos["shares"] * price

        if target_alloc is not None:
            new_pos = {}
            for fid, pos in positions.items():
                if fid not in target_alloc:
                    if not is_lower_circuit(fid):
                        ep = exec_prices.get(fid, pos.get("last_price", 0))
                        if ep > 0:
                            proceeds = pos["shares"] * ep * (1 - cfg_SR2.TRANS_COST)
                            cash += proceeds
                            day_turnover += pos["shares"] * ep
                            rebalance_exits += 1
                            if fid not in trades_log:
                                trades_log[fid] = []
                            trades_log[fid].append(
                                {
                                    "type": "SELL",
                                    "date": exec_date,
                                    "price": ep,
                                    "shares": pos["shares"],
                                    "reason": "Rebalance",
                                }
                            )
                        else:
                            new_pos[fid] = pos
                            pending_sells.add(fid)
                            circuit_breaker_delays += 1
                else:
                    new_pos[fid] = pos
            positions = new_pos

            for fid, weight in target_alloc.items():
                ep = exec_prices.get(fid)
                if not ep or ep <= 0:
                    continue
                tgt = equity * weight
                cur_sh = positions.get(fid, {"shares": 0, "entry_price": ep})["shares"]
                cur_val = cur_sh * ep
                diff = tgt - cur_val

                if diff > 0 and not is_lower_circuit(fid):
                    sh = int(diff / ep)
                    cost = sh * ep * (1 + cfg_SR2.TRANS_COST)
                    if cash >= cost and sh > 0:
                        cash -= cost
                        old_sh = positions.get(fid, {}).get("shares", 0)
                        old_entry = positions.get(fid, {}).get("entry_price", ep)
                        new_entry = ((old_sh * old_entry) + (sh * ep)) / (old_sh + sh)
                        positions[fid] = {
                            "shares": old_sh + sh,
                            "entry_price": new_entry,
                            "last_price": ep,
                            "highest_price": ep,
                        }
                        day_turnover += sh * ep
                        total_trades += 1
                        if fid not in trades_log:
                            trades_log[fid] = []
                        trades_log[fid].append(
                            {
                                "type": "BUY",
                                "date": exec_date,
                                "price": ep,
                                "shares": sh,
                            }
                        )

                elif diff < 0:
                    if not is_lower_circuit(fid):
                        sh = int(abs(diff) / ep)
                        proc = sh * ep * (1 - cfg_SR2.TRANS_COST)
                        if sh > 0:
                            cash += proc
                            new_sh = cur_sh - sh
                            if new_sh > 0:
                                positions[fid]["shares"] = new_sh
                            else:
                                positions.pop(fid, None)
                            day_turnover += sh * ep
                            total_trades += 1
                            if fid not in trades_log:
                                trades_log[fid] = []
                            trades_log[fid].append(
                                {
                                    "type": "SELL",
                                    "date": exec_date,
                                    "price": ep,
                                    "shares": sh,
                                    "reason": "Trim",
                                }
                            )
                    else:
                        pending_sells.add(fid)
                        circuit_breaker_delays += 1

        for fid in positions.keys():
            if fid in marking_prices:
                positions[fid]["last_price"] = marking_prices[fid]

        equity = cash
        for fid, pos in positions.items():
            equity += pos["shares"] * marking_prices.get(fid, pos.get("last_price", 0))

        pbar.set_description(
            f"Strat 2 Weight: {strat2_weight:.2f} | Eq: ₹{equity / 1e6:.1f}M"
        )

        results.append(
            {
                "date": exec_date,
                "equity": equity,
                "cash": cash,
                "positions": len(positions),
                "turnover": day_turnover,
                "stop_loss_exits": stop_loss_exits,
                "total_trades": total_trades,
            }
        )

    return pd.DataFrame(results), trades_log


def plot_lookahead_check(df_cut, df_full, fid, metric):
    df_cut_stock = df_cut[df_cut["fid"] == fid].set_index("tradedate").sort_index()
    df_full_stock = df_full[df_full["fid"] == fid].set_index("tradedate").sort_index()
    plt.figure(figsize=(10, 5))
    plt.plot(
        df_full_stock[metric], label="Full Data (Future known)", linewidth=4, alpha=0.5
    )
    plt.plot(
        df_cut_stock[metric], label="Cut Data (Strictly historical)", linestyle="--"
    )
    plt.title(f"Lookahead Bias Check: {metric} for {fid}")
    plt.legend()
    plt.show()


def plot_all_lookahead_metrics(
    df_cut,
    df_full,
    fid,
    cutoff_date,
    metrics=["kf_price", "momentum", "vol", "rsi", "dist_kf"],
):
    df_cut_stock = df_cut[df_cut["fid"] == fid].set_index("tradedate").sort_index()
    df_full_stock = df_full[df_full["fid"] == fid].set_index("tradedate").sort_index()
    common_index = df_cut_stock.index.intersection(df_full_stock.index)
    plot_index = common_index[common_index <= cutoff_date][-60:]

    fig, axes = plt.subplots(
        nrows=len(metrics), ncols=2, figsize=(15, 3 * len(metrics))
    )
    fig.suptitle(
        f"Lookahead Bias Validation for {fid} (Cutoff: {pd.Timestamp(cutoff_date).date()})",
        fontsize=16,
        fontweight="bold",
    )

    for i, metric in enumerate(metrics):
        if metric not in df_cut_stock.columns or metric not in df_full_stock.columns:
            continue

        s_full = df_full_stock.loc[plot_index, metric]
        s_cut = df_cut_stock.loc[plot_index, metric]
        diff = s_full.fillna(0) - s_cut.fillna(0)

        ax_overlay = axes[i, 0]
        ax_overlay.plot(
            s_full.index,
            s_full,
            label="Full History",
            color="#2E86AB",
            linewidth=5,
            alpha=0.5,
        )
        ax_overlay.plot(
            s_cut.index,
            s_cut,
            label="Truncated History",
            color="#F24236",
            linestyle="--",
            linewidth=2,
        )
        ax_overlay.set_title(f"{metric.upper()} - Overlay")
        ax_overlay.legend(loc="upper left", fontsize=9)
        ax_overlay.grid(True, alpha=0.3)

        ax_diff = axes[i, 1]
        ax_diff.plot(
            diff.index,
            diff,
            label="Difference (Full - Cut)",
            color="green",
            linewidth=2,
        )
        ax_diff.axhline(0, color="black", linestyle="-", linewidth=1)
        ax_diff.set_title(f"{metric.upper()} - Bias/Error")
        ax_diff.set_ylim(-1e-5, 1e-5)
        ax_diff.legend(loc="upper left", fontsize=9)
        ax_diff.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("lookahead_validation_grid.png", dpi=200)
    print(f" Saved comprehensive lookahead plot: lookahead_validation_grid.png")
    plt.close()


def check_lookahead_bias(cfg):
    print("\n" + "=" * 70)
    print("️ LOOKAHEAD BIAS DETECTION TEST (DATA CUTTING)")
    print("=" * 70)

    try:
        if cfg.DATA_PATH.endswith(".csv"):
            df_raw = pd.read_csv(cfg.DATA_PATH)
        else:
            df_raw = pd.read_parquet(cfg.DATA_PATH)
        df_raw.columns = df_raw.columns.str.lower().str.strip()
        df_raw["tradedate"] = pd.to_datetime(df_raw["tradedate"]).dt.normalize()
        for col in ["open", "high", "low", "close"]:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        df_raw = df_raw.dropna(subset=["close", "fid"]).sort_values(
            ["fid", "tradedate"]
        )
    except Exception as e:
        print(f" Error loading data: {e}")
        return

    print(" Generating features on FULL history...")
    results_full = Parallel(n_jobs=-1, verbose=0)(
        delayed(calculate_features_per_stock)(fid, df_raw[df_raw["fid"] == fid])
        for fid in df_raw["fid"].unique()
    )
    df_full = pd.concat(results_full, ignore_index=True)

    dates = sorted(df_full["tradedate"].unique())
    cutoff_idx = int(len(dates) * 0.8)
    cutoff_date = dates[cutoff_idx]
    print(f"\n Testing Cutoff Date: {pd.Timestamp(cutoff_date).date()}")

    print(
        " Generating features on TRUNCATED history (Data strictly <= Cutoff Date)..."
    )
    df_cut_raw = df_raw[df_raw["tradedate"] <= cutoff_date].copy()
    results_cut = Parallel(n_jobs=-1, verbose=0)(
        delayed(calculate_features_per_stock)(fid, df_cut_raw[df_cut_raw["fid"] == fid])
        for fid in df_cut_raw["fid"].unique()
    )
    df_cut = pd.concat(results_cut, ignore_index=True)

    avail_fids = df_cut["fid"].unique()
    if len(avail_fids) == 0:
        return
    test_fid = avail_fids[0]

    row_full = df_full[
        (df_full["tradedate"] == cutoff_date) & (df_full["fid"] == test_fid)
    ]
    row_cut = df_cut[(df_cut["tradedate"] == cutoff_date) & (df_cut["fid"] == test_fid)]

    if row_full.empty or row_cut.empty:
        print(" No matching rows found for comparison.")
        return

    metrics = ["kf_price", "momentum", "vol", "rsi", "dist_kf"]
    bias_found = False

    print(
        f"\n Verifying metrics for {test_fid} on {pd.Timestamp(cutoff_date).date()}"
    )
    print(f"{'Metric':<15} | {'Full Run':<15} | {'Cut Run':<15} | {'Difference'}")
    print("-" * 65)

    for m in metrics:
        if m not in row_full.columns or m not in row_cut.columns:
            continue
        v1 = row_full[m].values[0]
        v2 = row_cut[m].values[0]

        if pd.isna(v1) and pd.isna(v2):
            continue

        diff = abs(v1 - v2)
        status = "PASS" if np.isclose(v1, v2, rtol=1e-5, atol=1e-5) else "FAIL"

        print(f"{m:<15} | {v1:<15.4f} | {v2:<15.4f} | {diff:.1e} {status}")

        if diff > 1e-5:
            bias_found = True
            print(f"   => Plotting failure for {m}...")
            plot_lookahead_check(df_cut, df_full, test_fid, metric=m)
            break

    print("-" * 65)
    if not bias_found:
        print("FINAL RESULT: NO LOOKAHEAD BIAS DETECTED.")
        print("   (Your Kalman Filter and rolling windows are mathematically safe)")
        plot_all_lookahead_metrics(df_cut, df_full, test_fid, cutoff_date, metrics)
    else:
        print("FINAL RESULT: LOOKAHEAD BIAS DETECTED.")
    print("=" * 70 + "\n")


def plot_var_vs_weights(nav_series, max_weight=1.0, step=0.05, confidence_level=0.05):
    print("\n Generating Value at Risk (VaR) vs Exposure Plot...")
    returns = nav_series.pct_change().dropna()
    weights = np.arange(step, max_weight + step, step)
    var_percentages = []

    for weight in weights:
        scaled_returns = returns * weight
        VaR = np.percentile(scaled_returns, 100 * confidence_level)
        var_percentages.append(-VaR * 100)

    plt.figure(figsize=(10, 6))
    plt.plot(
        weights,
        var_percentages,
        marker="o",
        linestyle="-",
        color="#1f77b4",
        linewidth=2,
    )
    plt.title(
        f"Portfolio VaR (%) vs Investment Weights\n(Based on {100 * (1 - confidence_level):.0f}% Confidence Level)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Portfolio Exposure Weight", fontsize=12)
    plt.ylabel("Value at Risk (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig("var_vs_exposure.png", dpi=200)
    print(" Saved 'var_vs_exposure.png'")
    plt.close()


def plot_rolling_sharpe(nav_series, window=252):
    print("\n Generating Rolling Sharpe Ratio Plot...")
    ret = nav_series.pct_change().dropna()
    rolling_std = ret.rolling(window).std()
    rolling_mean = ret.rolling(window).mean()

    rolling_sharpe = (rolling_mean / (rolling_std + 1e-9)) * np.sqrt(252)

    plt.figure(figsize=(12, 5))
    plt.plot(rolling_sharpe.index, rolling_sharpe.values, color="#F5A623", linewidth=2)
    plt.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Sharpe = 1.0")
    plt.axhline(0.0, color="red", linestyle="--", alpha=0.5, label="Zero Threshold")
    plt.title(f"Rolling {window}-Day Sharpe Ratio", fontsize=14, fontweight="bold")
    plt.ylabel("Sharpe Ratio", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rolling_sharpe.png", dpi=200)
    print("Saved 'rolling_sharpe.png'")
    plt.close()


def run_monte_carlo_gbm(nav_series, initial_capital, simulations=1000):
    print(f" MONTE CARLO SIMULATION ({simulations} Runs)")
    print("=" * 70)

    daily_rets = nav_series.pct_change().dropna()
    daily_rets = daily_rets.replace([np.inf, -np.inf], 0.0)

    if daily_rets.empty:
        print("Error: No valid returns for simulation.")
        return

    mu_daily = daily_rets.mean()
    sigma_daily = daily_rets.std()
    n_days = len(daily_rets)

    Z = np.random.normal(0, 1, size=(n_days, simulations))
    drift_component = mu_daily - 0.5 * sigma_daily**2
    shock_component = sigma_daily * Z

    daily_log_returns_sim = drift_component + shock_component
    cumulative_growth = np.exp(np.cumsum(daily_log_returns_sim, axis=0))
    sim_curves = initial_capital * cumulative_growth

    start_row = np.full((1, simulations), initial_capital)
    sim_curves = np.vstack([start_row, sim_curves])

    final_values = sim_curves[-1, :]
    worst_case = np.percentile(final_values, 5)
    median_case = np.percentile(final_values, 50)
    best_case = np.percentile(final_values, 95)
    original_final = nav_series.iloc[-1]

    print(f"Original Final Equity: ₹{original_final:,.0f}")
    print(f"Median Expectation:    ₹{median_case:,.0f}")
    print(f"Worst Case (5% VaR):   ₹{worst_case:,.0f}")
    print(f"Best Case (95%):       ₹{best_case:,.0f}")

    plt.figure(figsize=(12, 6))
    plt.plot(sim_curves[:, :100], color="gray", alpha=0.05, linewidth=1)
    plt.plot(
        np.median(sim_curves, axis=1),
        color="orange",
        label="Median Simulation",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        np.percentile(sim_curves, 5, axis=1),
        color="red",
        label="95% Risk Limit",
        linewidth=1.5,
    )
    plt.plot(
        np.percentile(sim_curves, 95, axis=1),
        color="green",
        label="95% Upside Limit",
        linewidth=1.5,
    )

    original_curve = nav_series.values
    if len(original_curve) < len(sim_curves):
        original_curve = np.insert(original_curve, 0, initial_capital)

    plt.plot(original_curve, color="blue", linewidth=3, label="Actual Strategy")
    plt.title(
        f"Monte Carlo Simulation ({simulations} Paths) - Is it Luck?",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("Equity (INR)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("monte_carlo_sim.png", dpi=200)
    print("Saved 'monte_carlo_sim.png'")
    plt.close()

if __name__ == "__main__":
    TOTAL_CAPITAL = 5_000_000

    print("\nRunning Strategy A (100% Initial Capital)")
    df_a = load_data_SR2()

    res_a, trades_log_a = run_backtest_SR2(df_a, TOTAL_CAPITAL)
    nav_a = res_a.set_index("date")["equity"]

    print("\nRunning Strategy B (Background Warmup Simulation)")
    bt = Backtester_SR1(Config_SR1(), TOTAL_CAPITAL * 0.30)
    bt.run()
    nav_b = bt.results["nav"]
    trades_log_b = bt.position_history

    print("\nExecuting Dynamic Capital Transition (100% A -> 70/30 A+B)...")
    combined = pd.concat([nav_a, nav_b], axis=1)
    combined.columns = ["strat_a", "strat_b"]

    combined["strat_b"] = combined["strat_b"].fillna(TOTAL_CAPITAL * 0.30)
    combined = combined.sort_index().ffill().dropna()

    b_start_date = None
    for trade in trades_log_b:
        if trade["action"] == "buy":
            b_start_date = pd.to_datetime(trade["date"])
            break

    master_nav = pd.Series(index=combined.index, dtype=float)

    if b_start_date and b_start_date in combined.index:
        pre_mask = combined.index < b_start_date
        master_nav[pre_mask] = combined.loc[pre_mask, "strat_a"]

        shift_value_A = combined.loc[b_start_date, "strat_a"]
        shifted_cash = shift_value_A * 0.30
        trans_cost = shifted_cash * 0.00268

        cap_A_post = shift_value_A * 0.70
        cap_B_post = shifted_cash - trans_cost

        print(f"  • Date of Transition: {b_start_date.date()}")
        print(f"  • Total Capital at Transition: ₹{shift_value_A:,.0f}")
        print(f"  • Moving 30% (₹{shifted_cash:,.0f}) to Strategy B")
        print(f"  • Transition Slippage/Cost: -₹{trans_cost:,.0f}")

        post_mask = combined.index >= b_start_date

        growth_A = (
            combined.loc[post_mask, "strat_a"] / combined.loc[b_start_date, "strat_a"]
        )
        growth_B = (
            combined.loc[post_mask, "strat_b"] / combined.loc[b_start_date, "strat_b"]
        )

        master_nav[post_mask] = (cap_A_post * growth_A) + (cap_B_post * growth_B)
    else:
        print("Strategy B never traded. Defaulting to 100% Strategy A.")
        master_nav = combined["strat_a"]

    nav = master_nav.dropna()

    ret = nav.pct_change().dropna()

    bench = pd.read_excel("/Users/vokirankumar/kriti quant/indexes.xlsx")
    bench.columns = bench.columns.str.lower().str.strip()

    if "index_name" in bench.columns:
        bench = bench[bench["index_name"] == "NSE500"]

    bench["tradedate"] = pd.to_datetime(bench["tradedate"])

    bench = bench.drop_duplicates(subset=["tradedate"], keep="last")

    bench = bench.set_index("tradedate")
    bench = bench.reindex(nav.index).ffill()

    bench_ret = bench["close"].pct_change().fillna(0)

    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() != 0 else 0
    dd = nav / nav.cummax() - 1
    max_dd = dd.min()

    aligned = pd.concat([ret, bench_ret], axis=1).dropna()
    aligned.columns = ["strategy", "benchmark"]

    active = aligned["strategy"] - aligned["benchmark"]

    information_ratio = (
        (active.mean() / active.std()) * np.sqrt(252) if active.std() != 0 else 0
    )

    df_cap = pd.DataFrame({"strat": ret, "bench": bench_ret})

    up_mask = df_cap["bench"] > 0
    down_mask = df_cap["bench"] < 0

    def calc_geom_mean(returns):
        return np.exp(np.mean(np.log1p(returns))) - 1

    if up_mask.sum() > 0:
        strat_up_gmean = calc_geom_mean(df_cap.loc[up_mask, "strat"])
        bench_up_gmean = calc_geom_mean(df_cap.loc[up_mask, "bench"])
        up_capture = (
            (strat_up_gmean / bench_up_gmean) * 100 if bench_up_gmean != 0 else 0.0
        )
    else:
        up_capture = 0.0

    if down_mask.sum() > 0:
        strat_down_gmean = calc_geom_mean(df_cap.loc[down_mask, "strat"])
        bench_down_gmean = calc_geom_mean(df_cap.loc[down_mask, "bench"])
        down_capture = (
            (strat_down_gmean / bench_down_gmean) * 100
            if bench_down_gmean != 0
            else 0.0
        )
    else:
        down_capture = 0.0


    turnover_a = res_a["turnover"].sum()
    turnover_b = bt.turnover["buy"] + bt.turnover["sell"]
    total_traded_value = turnover_a + turnover_b
    avg_portfolio_nav = nav.mean()
    annualized_turnover_pct = (total_traded_value / 2) / avg_portfolio_nav / years

    pos_a = res_a.set_index("date")["positions"]
    pos_b = bt.results["num_holdings"]
    pos_a = pos_a.reindex(nav.index).ffill().fillna(0)
    pos_b = pos_b.reindex(nav.index).ffill().fillna(0)
    total_positions = pos_a + pos_b
    avg_positions = total_positions.mean()

    total_daily_positions = (
        total_positions
    )

    violations = total_daily_positions[total_daily_positions > 100]

    if not violations.empty:
        print(
            f"\n RULE VIOLATION DETECTED: Exceeded 100 stocks on {len(violations)} days."
        )
        print(f"Peak positions: {total_daily_positions.max()}")
        print("Suggestion: Lower Config_SR2.MAX_POSITIONS or Config_SR1.MAX_STOCKS.")
    else:
        print("\n Rule Compliance: Total positions strictly <= 100 on all days.")

    windows = {"1Y": 252, "3Y": 252 * 3, "5Y": 252 * 5}
    print("\nROLLING OUTPERFORMANCE ANALYSIS")
    print("=" * 40)

    for name, days in windows.items():
        if len(nav) > days:
            roll_strat = nav.pct_change(days)
            roll_bench = bench["close"].pct_change(days).reindex(nav.index)

            roll_alpha = roll_strat - roll_bench
            roll_alpha = roll_alpha.dropna()

            avg_out = roll_alpha.mean() * 100
            worst_under = roll_alpha.min() * 100
            valid = roll_alpha
            win_rate = (valid > 0).sum() / len(valid) * 100

            print(f"\n{name} Window")
            print(f"Avg Outperformance:  {avg_out:.2f}%")
            print(f"Worst Underperf.:    {worst_under:.2f}%")
            print(f"Win Rate vs Bench:   {win_rate:.1f}%")
        else:
            print(f"{name} → Not enough data")

    final_equity = nav.iloc[-1]

    def calculate_overall_win_rate(trades_log_a, trades_log_b):
        all_pnl = []

        for fid, trades in trades_log_a.items():
            current_shares = 0
            avg_cost = 0.0
            for t in trades:
                if t["type"] == "BUY":
                    total_cost = (current_shares * avg_cost) + (
                        t["shares"] * t["price"]
                    )
                    current_shares += t["shares"]
                    avg_cost = total_cost / current_shares if current_shares > 0 else 0
                elif t["type"] == "SELL":
                    pnl = (t["price"] - avg_cost) * t["shares"]
                    all_pnl.append(pnl)
                    current_shares -= t["shares"]
                    if current_shares <= 0:
                        current_shares = 0
                        avg_cost = 0.0

        entry_book = {}
        for t in trades_log_b:
            if t["action"] == "buy":
                fid = t["fid"]
                shares = t["shares"]
                price = t["price"]
                if fid not in entry_book:
                    entry_book[fid] = {"shares": 0, "cost": 0}
                entry_book[fid]["cost"] += shares * price
                entry_book[fid]["shares"] += shares

            elif t["action"] in ["sell_full", "delist_exit"]:
                fid = t["fid"]
                if fid in entry_book and entry_book[fid]["shares"] > 0:
                    avg_cost = entry_book[fid]["cost"] / entry_book[fid]["shares"]
                    pnl = (t["price"] - avg_cost) * t["shares"]
                    all_pnl.append(pnl)
                    del entry_book[fid]

        wins = sum(1 for pnl in all_pnl if pnl > 0)
        total = len(all_pnl)
        win_rate = (wins / total * 100) if total > 0 else 0
        return win_rate, total

    win_rate, total_trades = calculate_overall_win_rate(trades_log_a, trades_log_b)

    print("\n==============================")
    print("MERGED 70/30 CAPITAL STRATEGY")
    print("==============================")
    print(f"Final Equity      : ₹{final_equity:,.0f}")
    print(f"Total Return      : {(final_equity / TOTAL_CAPITAL - 1)*100:.2f}%")
    print(f"CAGR              : {cagr:.2%}")
    print(f"Sharpe Ratio      : {sharpe:.2f}")
    print(f"Volatility        : {vol:.2%}")
    print(f"Max Drawdown      : {max_dd:.2%}")
    print(f"Information Ratio : {information_ratio:.2f}")
    print(f"Up Capture        : {up_capture:.2f}%")
    print(f"Down Capture      : {down_capture:.2f}%")
    print(f"Overall Win Rate  : {win_rate:.2f}% ({total_trades} closed trades)")
    print(f"Annual Turnover   : {annualized_turnover_pct:.2%}")
    print(f"Avg Positions Held: {avg_positions:.1f}")


    print("\nGenerating Trade Log CSV and applying Order Netting...")
    a_records = []
    for fid, trades in trades_log_a.items():
        for t in trades:
            a_records.append(
                {
                    "Date": t["date"],
                    "Ticker": fid,
                    "Action": t["type"],
                    "Price": t["price"],
                    "Shares": t["shares"],
                    "Value": t["price"] * t["shares"],
                }
            )
    df_log_a = pd.DataFrame(a_records)

    b_records = []
    for t in trades_log_b:
        action = "BUY" if t["action"] == "buy" else "SELL"
        b_records.append(
            {
                "Date": t["date"],
                "Ticker": t["fid"],
                "Action": action,
                "Price": t["price"],
                "Shares": t["shares"],
                "Value": t["value"],
            }
        )
    df_log_b = pd.DataFrame(b_records)

    combined_logs = pd.concat([df_log_a, df_log_b], ignore_index=True)
    if not combined_logs.empty:
        combined_logs["Signed_Shares"] = combined_logs.apply(
            lambda x: x["Shares"] if x["Action"] == "BUY" else -x["Shares"], axis=1
        )
        combined_logs["Signed_Value"] = combined_logs.apply(
            lambda x: x["Value"] if x["Action"] == "BUY" else -x["Value"], axis=1
        )

        pre_netting_trades = len(combined_logs)
        pre_netting_turnover = combined_logs["Value"].sum()

        netted = (
            combined_logs.groupby(["Date", "Ticker"])
            .agg(Net_Shares=("Signed_Shares", "sum"), Net_Value=("Signed_Value", "sum"))
            .reset_index()
        )

        netted = netted[netted["Net_Shares"] != 0].copy()

        netted["Action"] = netted["Net_Shares"].apply(
            lambda x: "BUY" if x > 0 else "SELL"
        )
        netted["Shares"] = netted["Net_Shares"].abs()
        netted["Value"] = netted["Net_Value"].abs()

        netted["Price"] = netted["Value"] / netted["Shares"]

        final_log = (
            netted[["Date", "Ticker", "Action", "Price", "Shares", "Value"]]
            .sort_values(["Date", "Ticker"])
            .reset_index(drop=True)
        )

        post_netting_trades = len(final_log)
        post_netting_turnover = final_log["Value"].sum()

        trades_saved = pre_netting_trades - post_netting_trades
        turnover_saved = pre_netting_turnover - post_netting_turnover
        cost_saved = turnover_saved * 0.00268

        print(f"  • Duplicate Trades Eliminated : {trades_saved}")
        print(f"  • Unnecessary Turnover Avoided: ₹{turnover_saved:,.0f}")
        print(f"  • Theoretical Fees Saved      : ₹{cost_saved:,.0f}")

        final_log.to_csv("trade_log.csv", index=False)
        print("Master Netted Trade log successfully saved to 'trade_log.csv'")

    print("Generating Visual Performance Dashboard...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(
        nav.index,
        nav.values,
        label="Master Strategy (100% A -> 70/30)",
        color="#1f77b4",
        lw=2,
    )

    if bench is not None:
        rebased_bench = bench["close"].reindex(nav.index).ffill().bfill()
        rebased_bench = (rebased_bench / rebased_bench.iloc[0]) * nav.iloc[0]
        axes[0].plot(
            rebased_bench.index,
            rebased_bench.values,
            label="NSE 500 Benchmark",
            color="gray",
            ls="--",
            lw=1.5,
        )

    axes[0].set_title(
        "Master Portfolio Equity Curve (Log Scale)", fontsize=12, fontweight="bold"
    )
    axes[0].set_ylabel("Net Asset Value (₹)", fontsize=10)
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(dd.index, dd.values * 100, 0, color="#d62728", alpha=0.3)
    axes[1].plot(dd.index, dd.values * 100, color="#d62728", lw=1)
    axes[1].set_title("Strategy Drawdown (%)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Drawdown %", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(total_positions.index, total_positions.values, color="#2ca02c", lw=1.5)
    axes[2].set_title(
        "Active Portfolio Positions Over Time", fontsize=12, fontweight="bold"
    )
    axes[2].set_ylabel("Number of Stocks", fontsize=10)
    axes[2].set_xlabel("Date", fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("performance_dashboard.png")
    print("Dashboard successfully saved to 'performance_dashboard.png'")


    check_lookahead_bias(cfg_SR2)

    plot_var_vs_weights(nav)

    plot_rolling_sharpe(nav)

    run_monte_carlo_gbm(nav, TOTAL_CAPITAL, simulations=1000)

    plt.show()