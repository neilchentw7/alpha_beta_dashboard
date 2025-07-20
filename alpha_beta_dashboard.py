import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os

# --- 設定參數 ---
WINDOW = 60  # rolling window大小 (交易日)
CSV_FILE = "strategy_pnl.csv"  # 您每日更新的 P&L 檔
TAIEX_TICKER = "^TWII"  # 台股加權指數 Yahoo Finance 代碼

# --- 1. 輔助函式 ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_taiex(start: str, end: str) -> pd.Series:
    """下載台灣加權指數收盤價並回傳日報酬率"""
    df = yf.download(TAIEX_TICKER, start=start, end=end, progress=False)
    ret = df["Adj Close"].pct_change().dropna().rename("twii_ret")
    return ret

def calc_metrics(strategy_ret: pd.Series, twii_ret: pd.Series) -> pd.DataFrame:
    """合併並計算 rolling 相關係數與 β"""
    merged = pd.concat([strategy_ret, twii_ret], axis=1).dropna()
    merged.columns = ["ret", "twii_ret"]
    # 滾動相關係數
    merged["corr"] = merged["ret"].rolling(WINDOW).corr(merged["twii_ret"])
    # 滾動 β = Cov / Var
    cov = merged["ret"].rolling(WINDOW).cov(merged["twii_ret"])
    var = merged["twii_ret"].rolling(WINDOW).var()
    merged["beta"] = cov / var
    return merged

# --- 2. 主程式 ---

def main():
    st.title("📈 Alpha → Beta 監控儀表板")
    st.write(
        f"本儀表板每日監控 **{WINDOW} 日滾動 β 與相關係數 ρ**，當任一指標 > 0.4 即跳出警示，協助您及時發現策略是否由 Alpha 轉為隱含 Beta。\n"
        "只需維護 `strategy_pnl.csv`（兩欄：`date,ret`），或在下方手動輸入昨日報酬即可。"
    )

    # 2‑A 上傳最新 CSV 或使用現有檔案
    uploaded = st.file_uploader("上傳 / 取代 strategy_pnl.csv", type=["csv"])
    if uploaded is not None:
        pnl_df = pd.read_csv(uploaded, parse_dates=["date"])
        pnl_df.to_csv(CSV_FILE, index=False)
        st.success("✅ 已覆寫本地 CSV，請重新載入頁面以更新儀表板。")
        st.stop()

    # 2‑B 若本地已有檔案則載入；否則提示建立
    if os.path.exists(CSV_FILE):
        pnl_df = pd.read_csv(CSV_FILE, parse_dates=["date"])
    else:
        pnl_df = pd.DataFrame(columns=["date", "ret"])

    # 2‑C 手動新增昨日報酬
    with st.expander("手動新增昨日報酬記錄"):
        col1, col2 = st.columns(2)
        new_date = col1.date_input("日期", datetime.today().date() - timedelta(days=1))
        new_ret = col2.number_input("報酬率 (小數，例如 0.002 表 0.2%)", format="%.6f")
        if st.button("加入記錄"):
            pnl_df = pd.concat([
                pnl_df,
                pd.DataFrame({"date": [new_date], "ret": [new_ret]})
            ], ignore_index=True)
            pnl_df.to_csv(CSV_FILE, index=False)
            st.success("已寫入 CSV，請重新載入頁面以更新。")
            st.stop()

    if pnl_df.empty:
        st.warning("❗ 尚未有任何策略資料。請上傳或新增後重新整理。")
        return

    # --- 3. 下載 TAIEX 並計算指標 ---
    start = pnl_df["date"].min().strftime("%Y-%m-%d")
    end = (pnl_df["date"].max() + timedelta(days=1)).strftime("%Y-%m-%d")
    twii_ret = fetch_taiex(start, end)

    pnl_series = pnl_df.set_index("date")["ret"].astype(float)
    metrics = calc_metrics(pnl_series, twii_ret)

    latest = metrics.iloc[-1]
    latest_corr = latest["corr"]
    latest_beta = latest["beta"]
    delta_corr = latest_corr - metrics["corr"].iloc[-2] if len(metrics) > 1 else 0
    delta_beta = latest_beta - metrics["beta"].iloc[-2] if len(metrics) > 1 else 0

    # --- 4. 儀表板呈現 ---
    c1, c2 = st.columns(2)
    c1.metric("Rolling 60 日相關係數 ρ", f"{latest_corr:.2f}", f"{delta_corr:+.2f}")
    c2.metric("Rolling 60 日 β", f"{latest_beta:.2f}", f"{delta_beta:+.2f}")

    if (latest_corr > 0.4) or (latest_beta > 0.4):
        st.error("⚠️ ρ 或 β 已超過 0.4，策略可能從 Alpha 轉向隱含 Beta，請檢視持倉 / 因子曝險！")

    st.subheader("ρ 與 β 時序圖")
    st.line_chart(metrics[["corr", "beta"]])

    st.subheader("策略 vs 大盤 日報酬率 (累積)")
    perf = metrics[["ret", "twii_ret"]].cumsum()
    perf.columns = ["strategy_cum", "twii_cum"]
    st.line_chart(perf)

    st.download_button(
        "下載對齊後指標 CSV",
        metrics.reset_index().to_csv(index=False).encode(),
        file_name="metrics_with_beta_corr.csv",
    )

if __name__ == "__main__":
    main()
