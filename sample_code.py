"""
Universe Construction Pipeline
T1: 交易日历 → 月末交易日（EOM, End of Month）与生效区间（Effective Period）
T2: 按点时（PIT, Point-in-Time）口径计算特征（ADDV（Average Daily Dollar Volume）等）
T3: 过滤与排名（Ranking）→ 宇成员表（Universe Membership）
T4: QA 汇总（每月成分数、未入选原因分布）

依赖：
  pip install pandas numpy yfinance pandas-market-calendars

示例运行：
  python universe_pipeline.py \
    --calendar XNYS \
    --start-month 2019-01 \
    --end-month 2019-03 \
    --tickers "AAPL,MSFT,AMZN,GOOGL,BRK-B,JNJ,JPM,V,PG,NVDA,HD,DIS,MA,BAC,XOM,PFE,T,CVX,VZ,KO" \
    --outdir ./outputs \
    --save-parquet

你也可以用 --tickers-file 提供一个每行一个代码的文本文件。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal

# ======= 口径参数（Spec Parameters） =======
ADDV_WINDOW = 60                   # ADDV 窗口（交易日）
ADDV_THRESHOLD = 5_000_000         # 流动性门槛（USD）
PRICE_THRESHOLD = 5.0              # 价格门槛（USD）
STALENESS_LOOKBACK = 20            # 陈旧度窗口（交易日）
STALENESS_THRESHOLD = 10           # 近20日中“零成交或价格未变”的天数阈值
MIN_HISTORY = 60                   # 最小历史长度（交易日）
TOP_N = 1000                       # 按 addv60 排名入选的上限（全量时生效，小样不足则都入选）

# ======= T1：EOM 与生效区间 =======
def build_eom_table(calendar_code="XNYS", start_month="2019-01", end_month="2019-03"):
    cal = mcal.get_calendar(calendar_code)

    # 覆盖到 end_month 的下一月即可（为计算“下一行的 effective_from”）
    start = pd.to_datetime(start_month) - pd.offsets.MonthBegin(1)
    end = pd.to_datetime(end_month)   + pd.offsets.MonthEnd(2)

    schedule = cal.schedule(start_date=start.date(), end_date=end.date())
    trading_days = pd.DatetimeIndex(schedule.index.normalize())  # 交易日（无时区）

    # 月份列表：主区间（用来产出最终行）与“多一月”的辅助区间（用来拿下一行的 effective_from）
    months_main = pd.period_range(start=start_month, end=end_month, freq="M")
    months_plus = pd.period_range(start=start_month, end=(pd.Period(end_month, freq="M") + 1), freq="M")

    out = []
    for m in months_plus:
        in_m  = trading_days[trading_days.to_period("M") == m]
        if len(in_m) == 0:
            continue
        date_eom = in_m.max().date()

        in_m1 = trading_days[trading_days.to_period("M") == (m + 1)]
        if len(in_m1) == 0:
            continue
        effective_from = in_m1.min().date()
        effective_to = in_m1.max().date()

        out.append({"month": m, "date_eom": date_eom, "effective_from": effective_from, "effective_to": effective_to})
        
    out = pd.DataFrame(out).sort_values("month").reset_index(drop=True)

    
    return out

# ======= yfinance 拉取 OHLCV =======
def _coerce_list_maybe_csv(s: str | None) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def load_tickers(tickers_arg: str | None, tickers_file: str | None) -> List[str]:
    tickers = _coerce_list_maybe_csv(tickers_arg)
    if tickers_file:
        fp = Path(tickers_file)
        if fp.exists():
            more = [line.strip() for line in fp.read_text().splitlines() if line.strip()]
            tickers.extend(more)
    # 去重、保持顺序
    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq

def fetch_ohlcv(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    用 yfinance 拉日频 OHLCV（未复权 Close/Volume；另取 Adj Close）。
    返回 long 表：[date, ticker, open, high, low, close, adj_close, volume]
    """
    if not tickers:
        raise ValueError("tickers 列表为空。请通过 --tickers 或 --tickers-file 提供代码。")

    data = yf.download(
        tickers, start=start, end=end, auto_adjust=False,
        progress=False, group_by='ticker', threads=True
    )

    rows = []
    # yfinance 多 ticker 返回多层列，顶层是 ticker
    col_top = data.columns.get_level_values(0) if isinstance(data.columns, pd.MultiIndex) else []
    for t in tickers:
        if isinstance(data.columns, pd.MultiIndex):
            if t not in col_top:
                continue
            df_t = data[t].copy()
        else:
            # 单一股票情况
            df_t = data.copy()

        df_t = df_t.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
        df_t["ticker"] = t
        df_t = df_t.reset_index().rename(columns={"Date": "date"})  # yfinance 默认索引名为 "Date"
        rows.append(df_t[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]])

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["close", "volume"])
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out

# ======= T2：点时（PIT）特征 =======
def compute_point_in_time_features(ohlcv: pd.DataFrame, eom_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    对每支股票在其时间轴内 rolling 计算（仅用 <= 当日数据，满足点时（PIT））：
      - addv60 = Median( Close*Volume )，过去 60 交易日
      - seasoning_days = 历史交易天数（到当日为止）
      - staleness_days20 = 近 20 日（零成交 or 价格未变）天数
    然后仅保留 EOM 当天记录。
    """
    df = ohlcv.sort_values(["ticker", "date"]).copy()
    df["dollar_vol"] = df["close"] * df["volume"]

    def per_ticker_rolling(x: pd.DataFrame) -> pd.DataFrame:
        x = x.sort_values("date").copy()
        
        # 60日中位数（Median）
        x["addv60"] = (x["dollar_vol"].rolling(ADDV_WINDOW, min_periods=1).median())

        # 历史交易天数
        x["seasoning_days"] = np.arange(1, len(x) + 1)

        # 近20日“陈旧” = (价格未变 OR 零量)
        price_unchanged = x["close"].diff().fillna(0).eq(0).astype(int)
        zero_volume = (x["volume"] == 0).astype(int)
        stale_flag = (price_unchanged | zero_volume)
        x["staleness_days20"] = stale_flag.rolling(STALENESS_LOOKBACK, min_periods=1).sum()
        return x

    # 消除未来版本 pandas 的 groupby.apply 弃用告警：仅把需要列传入
    cols = ["date", "close", "volume", "dollar_vol"]
    df = (df.groupby("ticker", group_keys=False)[cols]
            .apply(per_ticker_rolling)
            .reset_index(level=0))  # 把 ticker 放回列

    # 仅保留 EOM 当天的点时值
    eom_dates = pd.to_datetime(eom_dates).tz_localize(None).normalize()
    out = (df[df["date"].isin(eom_dates)]
           .loc[:, ["date", "ticker", "close", "addv60", "seasoning_days", "staleness_days20"]]
           .rename(columns={"date": "date_eom", "close": "close_eom"}))
    return out

# ======= T2：过滤与排名 =======
def apply_filters_and_rank(eom_df: pd.DataFrame) -> pd.DataFrame:
    """
    应用四个过滤（价格/流动性/历史长度/陈旧）并在通过过滤的集合内按 addv60 降序排名。
    生成 rank 与 in_universe（TopN=1 其余=0）。
    """
    f = eom_df.copy()
    f["price_filter_pass"]     = (f["close_eom"] > PRICE_THRESHOLD).astype(int)
    f["addv_filter_pass"]      = (f["addv60"]    > ADDV_THRESHOLD).astype(int)
    f["seasoning_pass"]        = (f["seasoning_days"] >= MIN_HISTORY).astype(int)
    f["staleness_filter_pass"] = (f["staleness_days20"] < STALENESS_THRESHOLD).astype(int)

    mask = (f["price_filter_pass"] &
            f["addv_filter_pass"] &
            f["seasoning_pass"] &
            f["staleness_filter_pass"]).astype(bool)

    f["rank"] = np.nan
    f.loc[mask, "rank"] = f.loc[mask].groupby("date_eom")["addv60"].rank(method="first", ascending=False)

    f["in_universe"] = 0
    f.loc[mask & (f["rank"] <= TOP_N), "in_universe"] = 1
    return f

# ======= T3：宇成员表 =======
def build_universe_membership(universe_flags: pd.DataFrame, eom_table: pd.DataFrame) -> pd.DataFrame:
    """
    合并 EOM 生效区间（effective_from/to），并生成 exclusion_reason（未入选原因）。
    """
    uf = universe_flags.copy()
    out = uf.merge(
        eom_table[["date_eom", "effective_from", "effective_to"]],
        on="date_eom", how="left"
    )

    def _reason(row) -> str:
        reasons = []
        if row.get("price_filter_pass", 1) == 0:
            reasons.append("Price<=$5（Price Filter）")
        if row.get("addv_filter_pass", 1) == 0:
            reasons.append("ADDV60<=$5M（Liquidity Filter）")
        if row.get("seasoning_pass", 1) == 0:
            reasons.append("历史<60日（Seasoning）")
        if row.get("staleness_filter_pass", 1) == 0:
            reasons.append("陈旧≥10/20（Staleness）")
        return "; ".join(reasons) if reasons else ""

    out["exclusion_reason"] = out.apply(_reason, axis=1)

    cols = [
        "date_eom", "effective_from", "effective_to", "ticker",
        "close_eom", "addv60", "seasoning_days", "staleness_days20",
        "price_filter_pass", "addv_filter_pass", "seasoning_pass", "staleness_filter_pass",
        "rank", "in_universe", "exclusion_reason"
    ]
    out = out.loc[:, cols].sort_values(
        ["date_eom", "in_universe", "rank", "ticker"],
        ascending=[True, False, True, True]
    ).reset_index(drop=True)

    return out

# ======= T4：QA 汇总 =======
def qa_summary(universe_membership: pd.DataFrame) -> pd.DataFrame:
    """
    返回两个表：
      - monthly_count：每月入选数量（n_constituents）
      - excl：未入选原因分布（含 AllPassButNotTopN 占位）
    """
    um = universe_membership.copy()

    monthly_count = (um.query("in_universe==1")
                       .groupby("date_eom")["ticker"].nunique()
                       .rename("n_constituents")
                       .to_frame())

    excl = (um.query("in_universe==0")
              .assign(reason=lambda x: x["exclusion_reason"].replace("", f"AllPassButNotTop{TOP_N}"))
              .groupby(["date_eom", "reason"])["ticker"].nunique()
              .rename("count")
              .reset_index()
           )

    return monthly_count, excl

# ======= 工具：保存 =======
def save_table(df: pd.DataFrame, outdir: Path, name: str, save_parquet: bool = False):
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    if save_parquet:
        pq_path = outdir / f"{name}.parquet"
        df.to_parquet(pq_path, index=False)
    print(f"[Saved] {name} -> {csv_path}{' (+parquet)' if save_parquet else ''}")

# ======= 主流程 =======
def main():
    parser = argparse.ArgumentParser(description="Universe Construction Pipeline (PIT, ADDV, Ranking)")
    parser.add_argument("--calendar", default="XNYS", help="交易日历代号（Exchange Calendar Code），如 XNYS/XNAS")
    parser.add_argument("--start-month", required=True, help="开始月份（YYYY-MM）")
    parser.add_argument("--end-month", required=True, help="结束月份（YYYY-MM）")
    parser.add_argument("--tickers", default=None, help="逗号分隔的代码列表")
    parser.add_argument("--tickers-file", default=None, help="每行一个 ticker 的文本文件路径")
    parser.add_argument("--outdir", default="./outputs", help="输出目录")
    parser.add_argument("--save-parquet", action="store_true", help="同时保存 parquet")
    parser.add_argument("--data-start-pad", default="120D", help="为 rolling 预拉数据的额外天数（如 120D/6M）")
    args = parser.parse_args()

    outdir = Path(args.outdir)

    # 1) T1：EOM 表
    eom_table = build_eom_table(args.calendar, args.start_month, args.end_month)
    print("=== EOM & Effective Period ===")
    print(eom_table)

    # 2) 准备 tickers
    tickers = load_tickers(args.tickers, args.tickers_file)
    print(f"\nTickers ({len(tickers)}):", tickers[:10], "..." if len(tickers) > 10 else "")

    # 3) 计算数据抓取窗口：覆盖到最早 EOM 往前 ADDV_WINDOW 的交易日
    start_date = (pd.to_datetime(eom_table["date_eom"].min()) - pd.Timedelta(args.data_start_pad)).date().isoformat()
    end_date   = (pd.to_datetime(eom_table["effective_to"].max()) + pd.Timedelta(days=5)).date().isoformat()  # 向后多几天，稳健
    print(f"\nData window: {start_date} -> {end_date}")

    # 4) 拉取 OHLCV
    ohlcv = fetch_ohlcv(tickers, start_date, end_date)
    print(f"OHLCV shape: {ohlcv.shape}")

    # 5) T2：按点时（PIT）计算特征
    eom_dates = pd.to_datetime(eom_table["date_eom"]).dt.tz_localize(None).dt.normalize()
    eom_feat = compute_point_in_time_features(ohlcv, eom_dates)
    print("\n=== EOM Features (sample) ===")
    print(eom_feat.head())

    # 6) 过滤与排名
    universe_flags = apply_filters_and_rank(eom_feat)
    print("\n=== Flags & Rank (sample) ===")
    print(universe_flags.head())

    # 7) T3：宇成员表
    universe_membership = build_universe_membership(universe_flags, eom_table)
    print("\n=== Universe Membership (sample) ===")
    print(universe_membership.head(12))

    # 8) T4：QA
    monthly_count, excl = qa_summary(universe_membership)
    print("\n=== QA: 每月入选数量 ===")
    print(monthly_count)
    print("\n=== QA: 未入选的过滤原因分布 ===")
    print(excl.sort_values(["date_eom", "count"], ascending=[True, False]).head(20))

    # 9) 保存
    save_table(eom_table, outdir, "eom_table", args.save_parquet)
    save_table(eom_feat, outdir, "eom_features", args.save_parquet)
    save_table(universe_flags, outdir, "universe_flags", args.save_parquet)
    save_table(universe_membership, outdir, "universe_membership", args.save_parquet)
    save_table(monthly_count.reset_index(), outdir, "qa_monthly_count", args.save_parquet)
    save_table(excl, outdir, "qa_exclusion_breakdown", args.save_parquet)

if __name__ == "__main__":
    main()
