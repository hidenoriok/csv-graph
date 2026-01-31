import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

st.set_page_config(page_title="raw 1分平均/±σ（階段）", layout="wide")
st.info("DEBUG: axis-toggle code version = 2026-01-31-AXIS1")


CT_OPTIONS = [250, 300, 500]
CT_TO_FACTOR = {250: 3000, 300: 3600, 500: 6000}

GRID_CHOICES = ["なし", "0.5秒", "1秒", "1分", "5分"]
GRID_STEP_SECONDS = {"0.5秒": 0.5, "1秒": 1.0, "1分": 60.0, "5分": 300.0}
GRID_TO_PANDAS_FREQ = {"0.5秒": "500ms", "1秒": "1s", "1分": "1min", "5分": "5min"}

ELAPSED_UNIT_CHOICES = ["秒", "分"]
UNIT_TO_DIV = {"秒": 1.0, "分": 60.0}

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

def apply_time_range(df: pd.DataFrame, start_hms: str, end_hms: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    start/end は HH:MM:SS のみ。
    未入力なら df の最小/最大。
    入力ありなら df最小日付を基準日にして合成。
    end < start なら翌日に繰り上げ。

    戻り値：(スライス済みdf, base_time)
    base_time は
      - start入力あり: その時刻（経過時間 0 の基準）
      - start入力なし: スライス後の最小時刻
    """
    if df.empty:
        return df, pd.NaT

    has_start = bool(start_hms.strip())
    has_end = bool(end_hms.strip())
    if not has_start and not has_end:
        return df, df.index.min()

    base_date = df.index.min().normalize()

    start = df.index.min() if not has_start else pd.to_datetime(f"{base_date.date()} {start_hms.strip()}")
    end   = df.index.max() if not has_end   else pd.to_datetime(f"{base_date.date()} {end_hms.strip()}")

    if end < start:
        end = end + pd.Timedelta(days=1)

    sliced = df.loc[(df.index >= start) & (df.index <= end)]
    base_time = start if has_start else (sliced.index.min() if not sliced.empty else start)
    return sliced, base_time

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

def to_elapsed_x(index: pd.DatetimeIndex, base: pd.Timestamp, unit: str) -> np.ndarray:
    div = UNIT_TO_DIV[unit]
    return ((index - base).total_seconds() / div).astype(float)

def add_vertical_gridlines_datetime(ax, x_start, x_end, choice: str):
    """実時間軸：選択した間隔で縦補助線（axvline(datetime))"""
    if choice == "なし":
        return
    freq = GRID_TO_PANDAS_FREQ[choice]
    ticks = pd.date_range(start=x_start, end=x_end, freq=freq)
    for t in ticks:
        ax.axvline(t, linewidth=0.8, alpha=0.25)

def add_vertical_gridlines_elapsed(ax, x_end: float, choice: str, unit: str):
    """経過時間軸：選択した間隔で縦補助線（axvline(x))"""
    if choice == "なし":
        return
    step_sec = GRID_STEP_SECONDS[choice]
    step = step_sec / UNIT_TO_DIV[unit]  # 秒→そのまま、分→/60
    ticks = np.arange(0.0, x_end + step, step)
    for x in ticks:
        ax.axvline(x, linewidth=0.8, alpha=0.25)

# -----------------------------
# UI
# -----------------------------
st.title("CSVグラフ化ツール")
st.caption("start/end は HH:MM:SS。未入力なら全範囲。アップロードしたファイルだけ描画します。")

# ★ 追加：横軸モード（実時間 / 経過時間）
xaxis_mode = st.radio("横軸の表示方法", ["実時間", "経過時間"], index=0, horizontal=True)

elapsed_unit = st.radio(
    "経過時間の単位（経過時間を選んだ時）",
    ELAPSED_UNIT_CHOICES,
    index=0,
    horizontal=True,
    disabled=(xaxis_mode == "実時間"),
)

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

# -----------------------------
# 実行
# -----------------------------
if run:
    def try_load(name, up, ct):
        if not up:
            return None, pd.NaT
        try:
            df = load_csv(up, ct)
            df, base = apply_time_range(df, start_hms, end_hms)
            return df, base
        except Exception as e:
            st.error(f"{name} 読み込みエラー: {e}")
            return None, pd.NaT

    rpm_df, rpm_base = try_load("rpm", up_rpm, ct_rpm)
    pv_df, pv_base = try_load("pv", up_pv, ct_pv)
    batt_df, batt_base = try_load("batt", up_batt, ct_batt)

    if (rpm_df is None) and (pv_df is None) and (batt_df is None):
        st.warning("ファイルが選択されていません。rpm/pv/battのいずれかをアップロードしてください。")
        st.stop()

    # 経過時間の基準（start入力があればそれ、なければ表示対象の最小時刻）
    bases = []
    for df, base in [(rpm_df, rpm_base), (pv_df, pv_base), (batt_df, batt_base)]:
        if df is not None and not df.empty and pd.notna(base):
            bases.append(base)
    global_base = min(bases) if bases else pd.Timestamp.now()

    # 表示範囲（実時間のmin/max）: 描画するデータのmin/max
    ranges = []
    for df in [rpm_df, pv_df, batt_df]:
        if df is not None and not df.empty:
            ranges.append((df.index.min(), df.index.max()))
    x_start = min(r[0] for r in ranges)
    x_end   = max(r[1] for r in ranges)

    # rpm統計
    rpm_stats = None
    if calc_rpm_stats and (rpm_df is not None) and (not rpm_df.empty):
        rpm_stats = rpm_minute_stats_step(rpm_df["realpower"])

    # -----------------------------
    # グラフ描画
    # -----------------------------
    st.subheader("グラフ結果")
    fig, ax = plt.subplots(figsize=(14, 6))

    if xaxis_mode == "実時間":
        # 縦補助線（実時間）
        add_vertical_gridlines_datetime(ax, x_start, x_end, grid_choice)

        # raw（太線） 色指定：rpm青 / pv赤 / batt緑
        if rpm_df is not None and not rpm_df.empty:
            ax.plot(rpm_df.index, rpm_df["realpower"], linewidth=2.5, color="blue", label="rpm raw")
        if pv_df is not None and not pv_df.empty:
            ax.plot(pv_df.index, pv_df["realpower"], linewidth=2.5, color="red", label="pv raw")
        if batt_df is not None and not batt_df.empty:
            ax.plot(batt_df.index, batt_df["realpower"], linewidth=2.5, color="green", label="batt raw")

        # rpm 1分平均/±σ（階段）
        if rpm_stats is not None and not rpm_stats.empty:
            ax.step(rpm_stats.index, rpm_stats["mean_1min"], where="post",
                    linewidth=2.5, color="red", linestyle="--", label="rpm mean (1min, step)")
            ax.step(rpm_stats.index, rpm_stats["mean_plus_sigma"], where="post",
                    linewidth=1.8, color="red", linestyle="--", alpha=0.35, label="rpm mean + σ (step)")
            ax.step(rpm_stats.index, rpm_stats["mean_minus_sigma"], where="post",
                    linewidth=1.8, color="red", linestyle="--", alpha=0.35, label="rpm mean - σ (step)")

        ax.set_xlim(x_start, x_end)
        ax.set_xlabel("time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    else:
        # 経過時間軸（数値）
        # 表示範囲（経過時間の最大）
        x_end_elapsed = float(((x_end - global_base).total_seconds()) / UNIT_TO_DIV[elapsed_unit])

        # 縦補助線（経過時間）
        add_vertical_gridlines_elapsed(ax, x_end_elapsed, grid_choice, elapsed_unit)

        # raw（太線）
        if rpm_df is not None and not rpm_df.empty:
            x = to_elapsed_x(rpm_df.index, global_base, elapsed_unit)
            ax.plot(x, rpm_df["realpower"], linewidth=2.5, color="blue", label="rpm raw")
        if pv_df is not None and not pv_df.empty:
            x = to_elapsed_x(pv_df.index, global_base, elapsed_unit)
            ax.plot(x, pv_df["realpower"], linewidth=2.5, color="red", label="pv raw")
        if batt_df is not None and not batt_df.empty:
            x = to_elapsed_x(batt_df.index, global_base, elapsed_unit)
            ax.plot(x, batt_df["realpower"], linewidth=2.5, color="green", label="batt raw")

        # rpm 1分平均/±σ（階段）
        if rpm_stats is not None and not rpm_stats.empty:
            xs = to_elapsed_x(rpm_stats.index, global_base, elapsed_unit)
            ax.step(xs, rpm_stats["mean_1min"], where="post",
                    linewidth=2.5, color="red", linestyle="--", label="rpm mean (1min, step)")
            ax.step(xs, rpm_stats["mean_plus_sigma"], where="post",
                    linewidth=1.8, color="red", linestyle="--", alpha=0.35, label="rpm mean + σ (step)")
            ax.step(xs, rpm_stats["mean_minus_sigma"], where="post",
                    linewidth=1.8, color="red", linestyle="--", alpha=0.35, label="rpm mean - σ (step)")

        ax.set_xlim(0, x_end_elapsed)
        ax.set_xlabel(f"elapsed time from start ({elapsed_unit})")

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

