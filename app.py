import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

st.set_page_config(page_title="raw 1分平均/±σ（階段）", layout="wide")

CT_OPTIONS = [250, 300, 500]
CT_TO_FACTOR = {250: 3000, 300: 3600, 500: 6000}

GRID_CHOICES = ["なし", "0.5秒", "1秒", "1分", "5分"]
GRID_STEP_SECONDS = {"0.5秒": 0.5, "1秒": 1.0, "1分": 60.0, "5分": 300.0}
GRID_TO_PANDAS_FREQ = {"0.5秒": "500ms", "1秒": "1s", "1分": "1min", "5分": "5min"}

ELAPSED_UNIT_CHOICES = ["秒", "分"]
UNIT_TO_DIV = {"秒": 1.0, "分": 60.0}


# -----------------------------
# Functions
# -----------------------------
def load_csv(uploaded_file, ct_ratio: int) -> pd.DataFrame:
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
    if df is None or df.empty:
        return df, pd.NaT

    has_start = bool(start_hms.strip())
    has_end = bool(end_hms.strip())

    # どちらも未入力なら全範囲＆基準は最小
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
    if choice == "なし":
        return
    freq = GRID_TO_PANDAS_FREQ[choice]
    ticks = pd.date_range(start=x_start, end=x_end, freq=freq)
    for t in ticks:
        ax.axvline(t, linewidth=0.8, alpha=0.25)


def add_vertical_gridlines_elapsed(ax, x_end: float, choice: str, unit: str):
    if choice == "なし":
        return
    step_sec = GRID_STEP_SECONDS[choice]
    step = step_sec / UNIT_TO_DIV[unit]
    ticks = np.arange(0.0, x_end + step, step)
    for x in ticks:
        ax.axvline(x, linewidth=0.8, alpha=0.25)


# -----------------------------
# Session state init
# -----------------------------
if "data" not in st.session_state:
    st.session_state["data"] = {"rpm": None, "pv": None, "batt": None}
if "meta" not in st.session_state:
    st.session_state["meta"] = {"rpm_ct": 300, "pv_ct": 300, "batt_ct": 250}


# -----------------------------
# UI
# -----------------------------
st.title("CSVグラフ化ツール（データ固定→再描画）")
st.caption("①データ読込（更新）でCSVを読み込み固定。②その後はパラメータ変更で同じデータを再描画します。")

# ★ 表示パラメータ（ここはいつでも変更OK）
xaxis_mode = st.radio("横軸の表示方法", ["実時間", "経過時間"], index=0, horizontal=True)
elapsed_unit = st.radio(
    "経過時間の単位（経過時間を選んだ時）",
    ELAPSED_UNIT_CHOICES,
    index=0,
    horizontal=True,
    disabled=(xaxis_mode == "実時間"),
)

c1, c2 = st.columns(2)
with c1:
    start_hms = st.text_input("start（HH:MM:SS）", value="")
with c2:
    end_hms = st.text_input("end（HH:MM:SS）", value="")

calc_rpm_stats = st.checkbox("rpmの1分平均・±σをrawグラフに表示する", value=False)
grid_choice = st.radio("縦補助線（時間間隔）", GRID_CHOICES, index=2, horizontal=True)

st.divider()

# ★ データ読み込み（更新）エリア：ここだけ押したときに data が変わる
st.subheader("データ読込（更新）")
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

load_btn = st.button("データ読込（更新）", type="primary")
clear_btn = st.button("データクリア（全削除）")

if clear_btn:
    st.session_state["data"] = {"rpm": None, "pv": None, "batt": None}
    st.success("データをクリアしました（再度「データ読込（更新）」してください）。")

if load_btn:
    def load_one(name, up, ct):
        if not up:
            return None
        try:
            return load_csv(up, ct)
        except Exception as e:
            st.error(f"{name} 読み込みエラー: {e}")
            return None

    st.session_state["data"]["rpm"] = load_one("rpm", up_rpm, ct_rpm)
    st.session_state["data"]["pv"] = load_one("pv", up_pv, ct_pv)
    st.session_state["data"]["batt"] = load_one("batt", up_batt, ct_batt)
    st.session_state["meta"] = {"rpm_ct": ct_rpm, "pv_ct": ct_pv, "batt_ct": ct_batt}

    if (st.session_state["data"]["rpm"] is None
        and st.session_state["data"]["pv"] is None
        and st.session_state["data"]["batt"] is None):
        st.warning("どのファイルも読み込まれていません。")
    else:
        st.success("データを読み込みました。以後は同じデータで再描画できます。")

st.divider()

# -----------------------------
# 描画（データが存在すれば常に再描画）
# -----------------------------
rpm_raw = st.session_state["data"]["rpm"]
pv_raw = st.session_state["data"]["pv"]
batt_raw = st.session_state["data"]["batt"]

if (rpm_raw is None) and (pv_raw is None) and (batt_raw is None):
    st.info("まだデータがありません。上でファイルを選択して「データ読込（更新）」してください。")
    st.stop()

# 表示範囲でスライス（表示パラメータ変更のたびにここが再計算される）
rpm_df, rpm_base = (None, pd.NaT)
pv_df, pv_base = (None, pd.NaT)
batt_df, batt_base = (None, pd.NaT)

if rpm_raw is not None and not rpm_raw.empty:
    rpm_df, rpm_base = apply_time_range(rpm_raw, start_hms, end_hms)
if pv_raw is not None and not pv_raw.empty:
    pv_df, pv_base = apply_time_range(pv_raw, start_hms, end_hms)
if batt_raw is not None and not batt_raw.empty:
    batt_df, batt_base = apply_time_range(batt_raw, start_hms, end_hms)

# 表示範囲（共通）
ranges = []
for df in [rpm_df, pv_df, batt_df]:
    if df is not None and not df.empty:
        ranges.append((df.index.min(), df.index.max()))
if not ranges:
    st.warning("指定した start/end の範囲にデータがありません。")
    st.stop()

x_start = min(r[0] for r in ranges)
x_end   = max(r[1] for r in ranges)

# 経過時間の基準（start入力があればその時刻、なければ表示対象の最小時刻）
bases = []
for df, base in [(rpm_df, rpm_base), (pv_df, pv_base), (batt_df, batt_base)]:
    if df is not None and not df.empty and pd.notna(base):
        bases.append(base)
global_base = min(bases) if bases else x_start

# rpm統計（表示がONなら毎回計算）
rpm_stats = None
if calc_rpm_stats and (rpm_df is not None) and (not rpm_df.empty):
    rpm_stats = rpm_minute_stats_step(rpm_df["realpower"])

# ---- グラフ ----
st.subheader("グラフ結果（パラメータ変更で即再描画）")
fig, ax = plt.subplots(figsize=(14, 6))

if xaxis_mode == "実時間":
    add_vertical_gridlines_datetime(ax, x_start, x_end, grid_choice)

    if rpm_df is not None and not rpm_df.empty:
        ax.plot(rpm_df.index, rpm_df["realpower"], linewidth=2.5, color="blue", label="rpm raw")
    if pv_df is not None and not pv_df.empty:
        ax.plot(pv_df.index, pv_df["realpower"], linewidth=2.5, color="red", label="pv raw")
    if batt_df is not None and not batt_df.empty:
        ax.plot(batt_df.index, batt_df["realpower"], linewidth=2.5, color="green", label="batt raw")

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
    x_end_elapsed = float(((x_end - global_base).total_seconds()) / UNIT_TO_DIV[elapsed_unit])
    add_vertical_gridlines_elapsed(ax, x_end_elapsed, grid_choice, elapsed_unit)

    if rpm_df is not None and not rpm_df.empty:
        x = to_elapsed_x(rpm_df.index, global_base, elapsed_unit)
        ax.plot(x, rpm_df["realpower"], linewidth=2.5, color="blue", label="rpm raw")
    if pv_df is not None and not pv_df.empty:
        x = to_elapsed_x(pv_df.index, global_base, elapsed_unit)
        ax.plot(x, pv_df["realpower"], linewidth=2.5, color="red", label="pv raw")
    if batt_df is not None and not batt_df.empty:
        x = to_elapsed_x(batt_df.index, global_base, elapsed_unit)
        ax.plot(x, batt_df["realpower"], linewidth=2.5, color="green", label="batt raw")

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

# ---- rpm stats table ----
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

st.success("完了（データは保持中：再読込するまで固定）")


