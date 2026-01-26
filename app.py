import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="raw + rpm 1分平均/±σ（階段）", layout="wide")

CT_OPTIONS = [250, 300, 500]
CT_TO_FACTOR = {250: 3000, 300: 3600, 500: 6000}

GRID_CHOICES = ["なし", "0.5秒", "1秒", "1分", "5分"]
GRID_TO_PANDAS_FREQ = {
    "0.5秒": "500ms",
    "1秒": "1s",
    "1分": "1min",
    "5分": "5min",
}

def load_csv(uploaded_file, ct_ratio: int) -> pd.DataFrame:
    """
    CSV仕様（固定）:
      col0: YYYY/MM/DD
      col1: HH:MM:SS.mmm
      col2: power (数値)
    """
    df = pd.read_csv(uploaded_file, header=None)

    dt_str = df[0].astype(str).str.strip() + " " + df[1].astype(str).str.strip()
    t = pd.to_datetime(dt_str, format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    p = pd.to_numeric(df[2], errors="coerce")

    out = pd.DataFrame({"time": t, "power": p}).dropna()
    out = out.sort_values("time").set_index("time")

    factor = CT_TO_FACTOR[int(ct_ratio)]
    out["realpower"] = out["power"] * -factor / 100.0
    return out

def apply_time_range(df: pd.DataFrame, start_hms: str, end_hms: str) -> pd.DataFrame:
    """
    start/end は HH:MM:SS のみ。
    未入力なら df の最小/最大。
    入力ありなら df最小日付を基準日にして合成。
    end < start なら翌日に繰り上げ。
    """
    if df.empty:
        return df

    has_start = bool(start_hms.strip())
    has_end = bool(end_hms.strip())
    if not has_start and not has_end:
        return df

    base_date = df.index.min().normalize()

    start = df.index.min() if not has_start else pd.to_datetime(f"{base_date.date()} {start_hms.strip()}")
    end   = df.index.max() if not has_end   else pd.to_datetime(f"{base_date.date()} {end_hms.strip()}")

    if end < start:
        end = end + pd.Timedelta(days=1)

    return df.loc[(df.index >= start) & (df.index <= end)]

def rpm_minute_stats_step(rpm_realpower: pd.Series) -> pd.DataFrame:
    """
    rpmのみ：1分ごとの平均とσ（標準偏差）を計算
    - 1分平均なので階段表示向けに「1分境界（:00）に揃った index」になる
    - σは ddof=0（母標準偏差）
    """
    mean_ = rpm_realpower.resample("1min").mean()
    std_  = rpm_realpower.resample("1min").std(ddof=0)

    stats = pd.DataFrame({
        "mean_1min": mean_,
        "sigma_1min": std_,
        "mean_plus_sigma": mean_ + std_,
        "mean_minus_sigma": mean_ - std_,
    }).dropna()

    return stats

def add_vertical_gridlines(ax, x_start, x_end, choice: str):
    """選択した間隔で縦補助線（時間軸の縦線）を追加"""
    if choice == "なし":
        return
    freq = GRID_TO_PANDAS_FREQ[choice]
    ticks = pd.date_range(start=x_start, end=x_end, freq=freq)
    for t in ticks:
        ax.axvline(t, linewidth=0.8, alpha=0.25)

st.title("CSVグラフ化ツール")
st.caption("start/end は HH:MM:SS。未入力なら全範囲。アップロードしたファイルだけ描画します。")

# --- アップロードとCT比選択（各ファイルごと） ---
col1, col2, col3 = st.columns(3)
with col1:
    up_rpm = st.file_uploader("rpm.csv（任意）", type=["csv"], key="rpm_file")
    ct_rpm = st.selectbox("rpm CT比", CT_OPTIONS, index=2, key="rpm_ct")
with col2:
    up_pv = st.file_uploader("pv.csv（任意）", type=["csv"], key="pv_file")
    ct_pv = st.selectbox("pv CT比", CT_OPTIONS, index=1, key="pv_ct")
with col3:
    up_batt = st.file_uploader("batt.csv（任意）", type=["csv"], key="batt_file")
    ct_batt = st.selectbox("batt CT比", CT_OPTIONS, index=0, key="batt_ct")

c1, c2 = st.columns(2)
with c1:
    start_hms = st.text_input("start（HH:MM:SS）", value="")
with c2:
    end_hms = st.text_input("end（HH:MM:SS）", value="")

calc_rpm_stats = st.checkbox("rpmの1分平均・±σをrawグラフに重ねて表示する", value=True)

grid_choice = st.radio("縦補助線（時間間隔）", GRID_CHOICES, index=2, horizontal=True)  # デフォルト=1秒

run = st.button("実行", type="primary")

if run:
    def try_load(name, up, ct):
        if not up:
            return None
        try:
            df = load_csv(up, ct)
            df = apply_time_range(df, start_hms, end_hms)
            return df
        except Exception as e:
            st.error(f"{name} 読み込みエラー: {e}")
            return None

    rpm_df = try_load("rpm", up_rpm, ct_rpm)
    pv_df = try_load("pv", up_pv, ct_pv)
    batt_df = try_load("batt", up_batt, ct_batt)

    if (rpm_df is None) and (pv_df is None) and (batt_df is None):
        st.warning("ファイルが選択されていません。rpm/pv/battのいずれかをアップロードしてください。")
        st.stop()

    # 表示範囲（縦線用）: 描画するデータのmin/max
    ranges = []
    for df in [rpm_df, pv_df, batt_df]:
        if df is not None and not df.empty:
            ranges.append((df.index.min(), df.index.max()))
    x_start = min(r[0] for r in ranges)
    x_end   = max(r[1] for r in ranges)

    st.subheader("raw グラフ＋ rpm 1分平均/±σ")
    fig, ax = plt.subplots(figsize=(14, 6))

    # ★ 縦補助線
    add_vertical_gridlines(ax, x_start, x_end, grid_choice)

    # --- raw（太線） 色指定：rpm青 / pv赤 / batt緑 ---
    if rpm_df is not None and not rpm_df.empty:
        ax.plot(rpm_df.index, rpm_df["realpower"], linewidth=0.5, color="blue", label="rpm raw")
    if pv_df is not None and not pv_df.empty:
        ax.plot(pv_df.index, pv_df["realpower"], linewidth=0.5, color="red", label="pv raw")
    if batt_df is not None and not batt_df.empty:
        ax.plot(batt_df.index, batt_df["realpower"], linewidth=0.5, color="green", label="batt raw")

    rpm_stats = None

    # --- rpm 1分平均/±σ（階段表示） ---
    if calc_rpm_stats and (rpm_df is not None) and (not rpm_df.empty):
        rpm_stats = rpm_minute_stats_step(rpm_df["realpower"])
        if not rpm_stats.empty:
            # 平均：赤の破線（階段）
            ax.step(
                rpm_stats.index, rpm_stats["mean_1min"],
                where="post", linewidth=2.5, color="red", linestyle="--",
                label="rpm mean (1min, step)"
            )
            # ±σ：薄い赤の破線（階段）
            ax.step(
                rpm_stats.index, rpm_stats["mean_plus_sigma"],
                where="post", linewidth=1.8, color="red", linestyle="--", alpha=0.35,
                label="rpm mean + σ (step)"
            )
            ax.step(
                rpm_stats.index, rpm_stats["mean_minus_sigma"],
                where="post", linewidth=1.8, color="red", linestyle="--", alpha=0.35,
                label="rpm mean - σ (step)"
            )

    ax.set_xlim(x_start, x_end)
    ax.set_xlabel("time")
    ax.set_ylabel("realpower")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    st.pyplot(fig, use_container_width=True)

    # rpm統計の表（チェックONかつrpmあり）
    if calc_rpm_stats:
        if rpm_df is None or rpm_df.empty:
            st.info("rpm.csv が未選択（または範囲内データなし）のため、rpmの1分平均/σの表は表示しません。")
        else:
            st.subheader("rpm：1分ごとの平均/σ（表）")
            st.dataframe(rpm_stats, use_container_width=True)

            st.download_button(
                "rpm_1min_stats.csv をダウンロード",
                data=rpm_stats.to_csv(index=True).encode("utf-8-sig"),
                file_name="rpm_1min_stats.csv",
                mime="text/csv",
            )

    st.success("完了")
