# import pandas as pd
# import pandas_market_calendars as mcal
# from datetime import timedelta

# def build_eom_table(calendar_code="XNYS", start_month="2019-01", end_month="2019-03"):
#     cal = mcal.get_calendar(calendar_code)

#     # 覆盖到 end_month 的下一月即可（为计算“下一行的 effective_from”）
#     start = pd.to_datetime(start_month) - pd.offsets.MonthBegin(1)
#     end = pd.to_datetime(end_month)   + pd.offsets.MonthEnd(2)

#     schedule = cal.schedule(start_date=start.date(), end_date=end.date())
#     trading_days = pd.DatetimeIndex(schedule.index.normalize())  # 交易日（无时区）

#     # 月份列表：主区间（用来产出最终行）与“多一月”的辅助区间（用来拿下一行的 effective_from）
#     months_main = pd.period_range(start=start_month, end=end_month, freq="M")
#     months_plus = pd.period_range(start=start_month, end=(pd.Period(end_month, freq="M") + 1), freq="M")

#     output = []
#     for m in months_plus:
#         in_m  = trading_days[trading_days.to_period("M") == m]
#         if len(in_m) == 0:
#             continue
#         date_eom = in_m.max().date()

#         in_m1 = trading_days[trading_days.to_period("M") == (m + 1)]
#         if len(in_m1) == 0:
#             continue
#         effective_from = in_m1.min().date()
#         effective_to = in_m1.max().date()

#         output.append({"month": m, "date_eom": date_eom, "effective_from": effective_from, "effective_to": effective_to})
        
#     output = pd.DataFrame(output).sort_values("month").reset_index(drop=True)

    
#     return output

# if __name__ == "__main__":
#     table = build_eom_table("XNYS", "2019-01", "2019-03")
#     print(table)
        


#--------------------------------
import pandas as pd
import numpy as np
import yfinance as yf

# —— 口径参数（Spec） ——
ADDV_WINDOW = 60                   # ADDV回看窗口（交易日）
ADDV_THRESHOLD = 5_000_000         # 流动性门槛（美元）
PRICE_THRESHOLD = 5.0              # 价格门槛（美元）
STALENESS_LOOKBACK = 20            # 陈旧度回看窗口（交易日）
STALENESS_THRESHOLD = 10           # 近20日中“零成交或价格未变”的天数阈值
MIN_HISTORY = 60                   # 最小历史长度（交易日）

# —— 你的 EOM 表（用你自己的三行结果） ——
eom_table = pd.DataFrame({
    "date_eom":      pd.to_datetime(["2019-01-31","2019-02-28","2019-03-29"]),
    "effective_from":pd.to_datetime(["2019-02-01","2019-03-01","2019-04-01"]),
    "effective_to":  pd.to_datetime(["2019-02-28","2019-03-29","2019-04-30"]),
})
eom_dates = pd.to_datetime(eom_table["date_eom"]).dt.tz_localize(None).dt.normalize()

# —— 先用一小批 S&P1500 股票（示例：20 支；你可以替换成你自己的列表） ——
tickers = ["AAPL","MSFT","AMZN","GOOGL","BRK-B","JNJ","JPM","V","PG","NVDA",
           "HD","DIS","MA","BAC","XOM","PFE","T","CVX","VZ","KO"]

def fetch_ohlcv(tickers, start, end):
    """
    用 yfinance 拉日频 OHLCV（未复权 Close/Volume；另取 Adj Close 用收益）。
    返回 long 表：[date, ticker, open, high, low, close, adj_close, volume]
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=False,
                       progress=False, group_by='ticker', threads=True)
    rows = []
    # yfinance 多ticker时会生成多层列：顶层是 ticker
    for t in tickers:
        # 有些代码可能拉不到数据（例如退市/拼写），要保护
        if t not in data.columns.get_level_values(0):
            continue
        df_t = data[t].copy()
        df_t = df_t.rename(columns=str.lower).rename(columns={"adj close":"adj_close"})
        df_t["ticker"] = t
        df_t = df_t.reset_index().rename(columns={"Date":"date"})
        rows.append(df_t[["date","ticker","open","high","low","close","adj_close","volume"]])
    out = pd.concat(rows, ignore_index=True)
    # 去除关键字段缺失
    out = out.dropna(subset=["close","volume"])
    # 统一成“无时区”的日期
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out

def compute_point_in_time_features(ohlcv, eom_dates):
    """
    对每支股票在其时间轴内 rolling 计算：
      - addv60 = Median( Close*Volume ) 过去60日（含当日）
      - seasoning_days = 历史交易天数（到当日为止）
      - staleness_days20 = 近20日中（零成交 or 价格未变）天数
    然后只保留 EOM 当天的记录作为点时截面。
    """
    df = ohlcv.sort_values(["ticker","date"]).copy()
    df["dollar_vol"] = df["close"] * df["volume"]

    def per_ticker_rolling(x):
        x = x.sort_values("date").copy()
        # 60日中位数（更稳健；也可用均值+温莎化）
        x["addv60"] = (x["dollar_vol"]
                       .rolling(ADDV_WINDOW, min_periods=1)
                       .median())
        # 历史长度（第1天到第n天）
        x["seasoning_days"] = np.arange(1, len(x) + 1)
        # 近20日“陈旧”：零成交或价格未变
        price_unchanged = x["close"].diff().fillna(0).eq(0).astype(int)
        zero_volume = (x["volume"] == 0).astype(int)
        stale_flag = (price_unchanged | zero_volume)
        x["staleness_days20"] = stale_flag.rolling(STALENESS_LOOKBACK, min_periods=1).sum()
        return x

    df = df.groupby("ticker", group_keys=False).apply(per_ticker_rolling)

    # 仅保留 EOM 当天的点时值
    out = (df[df["date"].isin(eom_dates)]
           .loc[:, ["date","ticker","close","addv60","seasoning_days","staleness_days20"]]
           .rename(columns={"date":"date_eom","close":"close_eom"}))
    return out

def apply_filters_and_rank(eom_df):
    """
    应用四个过滤并在通过过滤的集合内，按 addv60 由高到低排名，标记 in_universe（Top1000=1）。
    """
    f = eom_df.copy()
    f["price_filter_pass"]    = (f["close_eom"]   > PRICE_THRESHOLD).astype(int)
    f["addv_filter_pass"]     = (f["addv60"]      > ADDV_THRESHOLD).astype(int)
    f["seasoning_pass"]       = (f["seasoning_days"] >= MIN_HISTORY).astype(int)
    f["staleness_filter_pass"]= (f["staleness_days20"] < STALENESS_THRESHOLD).astype(int)

    mask = (f["price_filter_pass"] &
            f["addv_filter_pass"] &
            f["seasoning_pass"] &
            f["staleness_filter_pass"]).astype(bool)

    # 分 EOM 排名
    f["rank"] = np.nan
    f.loc[mask, "rank"] = f.loc[mask].groupby("date_eom")["addv60"].rank(
        method="first", ascending=False
    )

    # Top 1000 = 1 其余 = 0（小样数量可能远小于1000）
    f["in_universe"] = 0
    f.loc[mask & (f["rank"] <= 1000), "in_universe"] = 1
    return f

if __name__ == "__main__":
    # 覆盖窗口：为能算到 2019-03 的60日回看，起始尽量提前
    start = "2018-08-01"
    end   = "2019-04-10"

    ohlcv = fetch_ohlcv(tickers, start, end)
    eom_feat = compute_point_in_time_features(ohlcv, eom_dates)
    universe_flags = apply_filters_and_rank(eom_feat)

    # 自检：看前 12 行
    print(universe_flags.head(12))
    # 自检：每个 EOM 有多少通过过滤并入选的股票
    print("\nCounts by EOM:")
    print(universe_flags.query("in_universe==1").groupby("date_eom").size())



#--------------------------------

# ==== T3：Universe Membership ====
def build_universe_membership(universe_flags, eom_table):
    """
    输入：universe_flags（含四个过滤、rank、in_universe），eom_table（EOM与生效区间）
    输出：universe_membership（每个EOM×ticker一行，含生效区间与过滤标记）
    """
    uf = universe_flags.copy()
    # 合并 EOM 生效区间
    out = uf.merge(
        eom_table[["date_eom","effective_from","effective_to"]],
        on="date_eom",
        how="left"
    )

    # 排除原因（exclusion_reason）：只要有一个过滤不通过，就记录原因；通过则为空
    def _reason(row):
        reasons = []
        if row["price_filter_pass"] == 0:
            reasons.append("Price<=$5（Price Filter）")
        if row["addv_filter_pass"] == 0:
            reasons.append("ADDV60<=$5M（Liquidity Filter）")
        if row["seasoning_pass"] == 0:
            reasons.append("历史<60日（Seasoning）")
        if row["staleness_filter_pass"] == 0:
            reasons.append("陈旧≥10/20（Staleness）")
        return "; ".join(reasons) if reasons else ""

    out["exclusion_reason"] = out.apply(_reason, axis=1)

    # 选出需要保留的核心字段，便于下游使用
    cols = [
        "date_eom","effective_from","effective_to","ticker",
        "close_eom","addv60","seasoning_days","staleness_days20",
        "price_filter_pass","addv_filter_pass","seasoning_pass","staleness_filter_pass",
        "rank","in_universe","exclusion_reason"
    ]
    out = out.loc[:, cols].sort_values(["date_eom","in_universe","rank","ticker"],
                                       ascending=[True, False, True, True]).reset_index(drop=True)
    return out

universe_membership = build_universe_membership(universe_flags, eom_table)
print("\n=== Universe Membership (sample 12 rows) ===")
print(universe_membership.head(12))

# ==== T4：QA 汇总 ====
def qa_summary(universe_membership):
    """
    产出月度维度的 QA：成分数、进出场、换手率、各过滤原因占比。
    小样先给成分数与过滤原因分布示例。
    """
    um = universe_membership.copy()

    # 1) 每月入选数量
    monthly_count = (um.query("in_universe==1")
                       .groupby("date_eom")["ticker"].nunique()
                       .rename("n_constituents"))

    # 2) 过滤原因分布（在未入选集合中统计）
    excl = (um.query("in_universe==0")
              .assign(reason=lambda x: x["exclusion_reason"].replace("", "AllPassButNotTop1000"))
              .groupby(["date_eom","reason"])["ticker"].nunique()
              .rename("count")
              .reset_index())

    print("\n=== QA: 每月入选数量 ===")
    print(monthly_count)
    print("\n=== QA: 未入选的过滤原因分布（TopN外或未通过过滤） ===")
    print(excl.sort_values(["date_eom","count"], ascending=[True, False]).head(20))

qa_summary(universe_membership)






