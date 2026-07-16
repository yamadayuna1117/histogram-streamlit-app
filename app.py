from __future__ import annotations

import hashlib
import math
import re
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# アプリ基本設定
# ============================================================
st.set_page_config(
    page_title="ヒストグラム分析ツール",
    page_icon="📊",
    layout="wide",
)

matplotlib.rcParams["font.family"] = [
    "Noto Sans CJK JP",  # Streamlit Community Cloud
    "Yu Gothic",        # Windows
    "Meiryo",           # Windows
    "DejaVu Sans",      # fallback
]
matplotlib.rcParams["axes.unicode_minus"] = False

MATPLOTLIB_LOCK = RLock()
MIN_POSITIVE_VALUE = 1e-9
MAX_BIN_COUNT = 5000

BIN_METHODS = {
    "Freedman–Diaconis（外れ値に比較的強い）": "fd",
    "平方根則（シンプル）": "sqrt",
    "スタージェスの公式（少標本向け）": "sturges",
    "Scottの公式（正規分布に近いデータ向け）": "scott",
}

Y_MODES = {
    "度数": "count",
    "割合（%）": "percent",
    "確率密度": "density",
}


# ============================================================
# 見た目
# ============================================================
st.markdown(
    """
    <style>
      .stApp { background: #f7f9fc; }
      .block-container { padding-top: 1.8rem; padding-bottom: 3rem; }
      h1, h2, h3 { color: #17365d; }
      [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e3e8ef;
        border-radius: 12px;
        padding: 0.7rem 0.9rem;
        box-shadow: 0 2px 8px rgba(20, 45, 80, 0.04);
      }
      div[data-testid="stDownloadButton"] button { width: 100%; }
      .small-note { color: #5d6b7a; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 共通関数
# ============================================================
def stable_key(*parts: Any) -> str:
    """任意の文字列からStreamlit用の安定した短いキーを作る。"""
    source = "||".join(str(part) for part in parts)
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


def safe_filename(text: str, suffix: str) -> str:
    """OS依存文字を避けたファイル名を作る。"""
    cleaned = re.sub(r"[\\/:*?\"<>|\s]+", "_", text).strip("_")
    return f"{cleaned or 'histogram'}{suffix}"


def format_number(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "算出不可"
    return f"{value:,.{digits}f}"


@st.cache_data(scope="session", show_spinner="Excelファイルを確認しています…")
def get_sheet_names(file_bytes: bytes) -> list[str]:
    return pd.ExcelFile(BytesIO(file_bytes)).sheet_names


@st.cache_data(scope="session", show_spinner="シートを読み込んでいます…")
def read_sheet(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)


def prepare_numeric_data(series: pd.Series) -> dict[str, Any]:
    """空欄・非数値・無限大を分類し、有限の数値だけを返す。"""
    original = series.copy()

    string_view = original.astype("string")
    blank_mask = string_view.str.strip().eq("").fillna(False)
    missing_mask = original.isna() | blank_mask

    numeric = pd.to_numeric(original.mask(blank_mask), errors="coerce")
    non_numeric_mask = (~missing_mask) & numeric.isna()

    numeric_float = numeric.astype(float)
    infinite_mask = numeric.notna() & ~np.isfinite(numeric_float)
    valid_mask = numeric.notna() & np.isfinite(numeric_float)

    values = numeric_float[valid_mask].to_numpy(dtype=float)
    source_rows = np.flatnonzero(valid_mask.to_numpy()) + 2  # Excelのヘッダーを1行目と仮定

    grid_df = pd.DataFrame(
        {
            "Excel行": source_rows,
            "値": values,
        }
    )

    return {
        "values": values,
        "grid_df": grid_df,
        "original_count": int(len(original)),
        "valid_count": int(valid_mask.sum()),
        "missing_count": int(missing_mask.sum()),
        "non_numeric_count": int(non_numeric_mask.sum()),
        "infinite_count": int(infinite_mask.sum()),
        "excluded_count": int((~valid_mask).sum()),
    }


def calculate_statistics(values: np.ndarray) -> dict[str, float | None]:
    n = len(values)
    if n == 0:
        raise ValueError("数値データがありません。")

    series = pd.Series(values)
    modes = series.mode(dropna=True)
    value_counts = series.value_counts(dropna=True)
    mode_value: float | None = None
    if not value_counts.empty and int(value_counts.iloc[0]) >= 2 and len(modes) > 0:
        mode_value = float(modes.iloc[0])

    return {
        "n": float(n),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if n >= 2 else None,
        "min": float(np.min(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "max": float(np.max(values)),
        "mode": mode_value,
    }


def raw_bin_width(values: np.ndarray, method: str) -> float:
    """各公式による丸め前の階級幅を返す。"""
    n = len(values)
    x_min = float(np.min(values))
    x_max = float(np.max(values))
    data_range = x_max - x_min

    if n <= 1 or data_range <= 0:
        return 0.0

    if method == "sqrt":
        bin_count = max(1, math.ceil(math.sqrt(n)))
        return data_range / bin_count

    if method == "sturges":
        bin_count = max(1, math.ceil(math.log2(n) + 1))
        return data_range / bin_count

    if method == "scott":
        sample_std = float(np.std(values, ddof=1))
        width = 3.5 * sample_std * (n ** (-1 / 3))
        if width > 0 and np.isfinite(width):
            return width
        return raw_bin_width(values, "sqrt")

    # Freedman–Diaconis
    q1, q3 = np.percentile(values, [25, 75])
    iqr = float(q3 - q1)
    width = 2.0 * iqr * (n ** (-1 / 3))
    if width > 0 and np.isfinite(width):
        return width

    # IQRが0の場合はScott、それも無理なら平方根則へフォールバック
    width = raw_bin_width(values, "scott")
    if width > 0 and np.isfinite(width):
        return width
    return raw_bin_width(values, "sqrt")


def snap_width(width: float, measurement_unit: float) -> float:
    unit = max(float(measurement_unit), MIN_POSITIVE_VALUE)
    if width <= 0 or not np.isfinite(width):
        return unit
    return max(unit, round(width / unit) * unit)


def automatic_bin_settings(
    values: np.ndarray,
    method: str,
    measurement_unit: float,
) -> tuple[float, float]:
    width = snap_width(raw_bin_width(values, method), measurement_unit)
    start = float(np.min(values)) - float(measurement_unit) / 2.0
    return width, start


def build_bins(values: np.ndarray, width: float, start: float) -> np.ndarray:
    width = float(width)
    start = float(start)
    x_min = float(np.min(values))
    x_max = float(np.max(values))

    if not np.isfinite(width) or width <= 0:
        raise ValueError("区間幅は0より大きい値にしてください。")
    if not np.isfinite(start):
        raise ValueError("区間開始が正しい数値ではありません。")
    if start > x_min:
        raise ValueError(f"区間開始は最小値 {x_min:g} 以下にしてください。")
    if start >= x_max:
        raise ValueError(f"区間開始は最大値 {x_max:g} より小さくしてください。")

    bin_count = max(1, math.ceil((x_max - start) / width))
    if bin_count > MAX_BIN_COUNT:
        raise ValueError(
            f"区間数が {bin_count:,} 個になります。区間幅を大きくしてください。"
        )

    bins = start + np.arange(bin_count + 1, dtype=float) * width
    if bins[-1] < x_max:
        bins = np.append(bins, bins[-1] + width)
    return bins


def histogram_arguments(values: np.ndarray, y_mode: str) -> tuple[dict[str, Any], str]:
    if y_mode == "percent":
        weights = np.full(len(values), 100.0 / len(values), dtype=float)
        return {"weights": weights, "density": False}, "割合（%）"
    if y_mode == "density":
        return {"density": True}, "確率密度"
    return {"density": False}, "度数"


def make_histogram_figure(
    item: dict[str, Any],
    y_mode: str,
    show_mean: bool,
    show_median: bool,
    show_stats: bool,
    selected_value: float | None,
) -> plt.Figure:
    with MATPLOTLIB_LOCK:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        hist_kwargs, y_label = histogram_arguments(item["values"], y_mode)

        ax.hist(
            item["values"],
            bins=item["bins"],
            color="#72b7e2",
            edgecolor="#245a86",
            linewidth=1.0,
            alpha=0.82,
            **hist_kwargs,
        )

        if show_mean:
            ax.axvline(
                item["stats"]["mean"],
                color="#d9485f",
                linestyle="--",
                linewidth=1.5,
                label="平均",
            )

        if show_median:
            ax.axvline(
                item["stats"]["median"],
                color="#7b4ab5",
                linestyle=":",
                linewidth=1.7,
                label="中央値",
            )

        if selected_value is not None and np.isfinite(selected_value):
            ax.axvline(
                selected_value,
                color="#178746",
                linewidth=1.7,
                label="表で選択した値",
            )

        if show_stats:
            std_text = format_number(item["stats"]["std"], 3)
            stat_text = (
                f"n = {int(item['stats']['n'])}\n"
                f"階級幅 = {item['bin_width']:.4g}\n"
                f"平均 = {item['stats']['mean']:.4g}\n"
                f"中央値 = {item['stats']['median']:.4g}\n"
                f"標準偏差 = {std_text}"
            )
            ax.text(
                0.98,
                0.96,
                stat_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.82,
                    "edgecolor": "#d9e0e8",
                    "boxstyle": "round,pad=0.4",
                },
            )

        ax.set_xlabel("値")
        ax.set_ylabel(y_label)
        ax.grid(axis="y", linestyle="--", alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)

        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, loc="best")

        fig.tight_layout()
        return fig


def make_comparison_figure(
    items: list[dict[str, Any]],
    bins: np.ndarray,
    y_mode: str,
    style: str,
    show_means: bool,
) -> plt.Figure:
    with MATPLOTLIB_LOCK:
        fig, ax = plt.subplots(figsize=(10, 5.4))
        colors = plt.cm.tab10.colors

        for index, item in enumerate(items):
            color = colors[index % len(colors)]
            hist_kwargs, y_label = histogram_arguments(item["values"], y_mode)

            if style == "輪郭線":
                ax.hist(
                    item["values"],
                    bins=bins,
                    histtype="step",
                    linewidth=2.0,
                    color=color,
                    label=item["label"],
                    **hist_kwargs,
                )
            else:
                ax.hist(
                    item["values"],
                    bins=bins,
                    alpha=0.33,
                    linewidth=1.0,
                    color=color,
                    edgecolor=color,
                    label=item["label"],
                    **hist_kwargs,
                )

            if show_means:
                ax.axvline(
                    item["stats"]["mean"],
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.9,
                )

        ax.set_xlabel("値")
        ax.set_ylabel(y_label)
        ax.grid(axis="y", linestyle="--", alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        return fig


def figure_to_png(fig: plt.Figure) -> bytes:
    buffer = BytesIO()
    with MATPLOTLIB_LOCK:
        fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def unique_excel_sheet_name(base: str, used: set[str]) -> str:
    cleaned = re.sub(r"[\\/*?:\[\]]", "_", base).strip() or "Data"
    cleaned = cleaned[:31]
    candidate = cleaned
    number = 2
    while candidate in used:
        suffix = f"_{number}"
        candidate = f"{cleaned[:31-len(suffix)]}{suffix}"
        number += 1
    used.add(candidate)
    return candidate


def create_excel_report(
    stats_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    items: list[dict[str, Any]],
) -> bytes:
    buffer = BytesIO()
    used_names: set[str] = set()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        stats_df.to_excel(
            writer,
            sheet_name=unique_excel_sheet_name("統計量", used_names),
            index=False,
        )
        quality_df.to_excel(
            writer,
            sheet_name=unique_excel_sheet_name("データ品質", used_names),
            index=False,
        )

        for item in items:
            sheet_name = unique_excel_sheet_name(item["label"], used_names)
            item["grid_df"].to_excel(writer, sheet_name=sheet_name, index=False)

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# タイトル
# ============================================================
st.title("📊 ヒストグラム分析ツール")
st.markdown(
    "Excelの数値データから、ヒストグラム作成・統計確認・分布比較・結果保存ができます。"
)


# ============================================================
# 1. ファイル読み込み
# ============================================================
st.sidebar.header("1️⃣ ファイル")
uploaded_file = st.sidebar.file_uploader(
    "Excelファイルをアップロード",
    type=["xlsx"],
    help=".xlsx形式に対応しています。ファイルはこのアプリの保存領域には書き込みません。",
)

if uploaded_file is None:
    st.info("左のサイドバーからExcelファイルをアップロードしてください。")
    st.stop()

file_bytes = uploaded_file.getvalue()
file_signature = hashlib.sha1(file_bytes).hexdigest()

try:
    sheet_names = get_sheet_names(file_bytes)
except Exception as exc:
    st.error(f"Excelファイルを読み込めませんでした：{exc}")
    st.stop()

selected_sheets = st.sidebar.multiselect(
    "分析するシート",
    options=sheet_names,
    default=sheet_names[:1],
)

if not selected_sheets:
    st.info("分析するシートを1つ以上選択してください。")
    st.stop()

sheet_frames: dict[str, pd.DataFrame] = {}
for sheet_name in selected_sheets:
    try:
        sheet_frames[sheet_name] = read_sheet(file_bytes, sheet_name)
    except Exception as exc:
        st.sidebar.error(f"{sheet_name}を読み込めません：{exc}")

if not sheet_frames:
    st.error("読み込めるシートがありません。")
    st.stop()

source_signature = stable_key(file_signature, *selected_sheets)
if st.session_state.get("source_signature") != source_signature:
    st.session_state["source_signature"] = source_signature
    st.session_state.pop("analysis_config", None)


# ============================================================
# 2. 分析設定（フォームで一括反映）
# ============================================================
st.sidebar.header("2️⃣ 分析設定")
show_advanced = st.sidebar.toggle(
    "詳細設定を表示",
    value=False,
    help="階級幅の決定方法や測定単位を調整できます。",
)

existing_config = st.session_state.get("analysis_config", {})

with st.sidebar.form("analysis_settings_form"):
    selected_columns: dict[str, list[Any]] = {}

    for sheet_name, frame in sheet_frames.items():
        numeric_candidates = [
            column
            for column in frame.columns
            if pd.to_numeric(frame[column], errors="coerce").notna().any()
        ]
        previous = existing_config.get("selected_columns", {}).get(sheet_name)
        default_columns = previous if previous is not None else numeric_candidates[:4]

        selected_columns[sheet_name] = st.multiselect(
            f"列を選択：{sheet_name}",
            options=list(frame.columns),
            default=[column for column in default_columns if column in frame.columns],
        )

    st.markdown("##### 表示")
    y_mode_label = st.selectbox(
        "個別ヒストグラムの縦軸",
        options=list(Y_MODES),
        index=list(Y_MODES).index(existing_config.get("y_mode_label", "度数"))
        if existing_config.get("y_mode_label", "度数") in Y_MODES
        else 0,
    )
    cards_per_row = st.selectbox(
        "1行に表示するグラフ数",
        options=[1, 2, 3],
        index=[1, 2, 3].index(existing_config.get("cards_per_row", 2))
        if existing_config.get("cards_per_row", 2) in [1, 2, 3]
        else 1,
    )
    show_mean = st.checkbox(
        "平均線を表示",
        value=existing_config.get("show_mean", True),
    )
    show_median = st.checkbox(
        "中央値線を表示",
        value=existing_config.get("show_median", True),
    )
    show_stats = st.checkbox(
        "グラフ内に統計量を表示",
        value=existing_config.get("show_stats", True),
    )

    if show_advanced:
        st.markdown("##### 階級の自動設定")
        bin_method_label = st.selectbox(
            "階級幅の決定方法",
            options=list(BIN_METHODS),
            index=list(BIN_METHODS).index(
                existing_config.get(
                    "bin_method_label",
                    "Freedman–Diaconis（外れ値に比較的強い）",
                )
            )
            if existing_config.get(
                "bin_method_label",
                "Freedman–Diaconis（外れ値に比較的強い）",
            )
            in BIN_METHODS
            else 0,
        )
        measurement_unit = st.number_input(
            "既定の測定単位",
            min_value=MIN_POSITIVE_VALUE,
            value=float(existing_config.get("measurement_unit", 0.1)),
            step=0.1,
            format="%.6f",
            help="自動計算した階級幅を、この単位の倍数に丸めます。",
        )
    else:
        bin_method_label = existing_config.get(
            "bin_method_label",
            "Freedman–Diaconis（外れ値に比較的強い）",
        )
        measurement_unit = float(existing_config.get("measurement_unit", 0.1))

    submitted = st.form_submit_button(
        "設定を適用",
        type="primary",
        width="stretch",
    )

if submitted:
    st.session_state["analysis_config"] = {
        "selected_columns": selected_columns,
        "y_mode_label": y_mode_label,
        "cards_per_row": cards_per_row,
        "show_mean": show_mean,
        "show_median": show_median,
        "show_stats": show_stats,
        "bin_method_label": bin_method_label,
        "measurement_unit": float(measurement_unit),
    }
    existing_config = st.session_state["analysis_config"]

if not existing_config:
    st.info("サイドバーで列を確認し、「設定を適用」を押してください。")
    st.stop()

config = existing_config


# ============================================================
# 3. データ整形
# ============================================================
raw_items: list[dict[str, Any]] = []

for sheet_name, columns in config["selected_columns"].items():
    frame = sheet_frames.get(sheet_name)
    if frame is None:
        continue

    for column in columns:
        if column not in frame.columns:
            continue

        prepared = prepare_numeric_data(frame[column])
        if prepared["valid_count"] == 0:
            st.sidebar.warning(f"{sheet_name} / {column}：有効な数値がありません。")
            continue

        values = prepared["values"]
        item_id = stable_key(file_signature, sheet_name, column)
        raw_items.append(
            {
                "id": item_id,
                "sheet": sheet_name,
                "column": str(column),
                "label": f"{sheet_name} / {column}",
                "values": values,
                "grid_df": prepared["grid_df"],
                "quality": prepared,
                "stats": calculate_statistics(values),
            }
        )

if not raw_items:
    st.warning("選択された列に表示可能な数値データがありません。")
    st.stop()


# ============================================================
# 4. 列ごとの詳細設定
# ============================================================
items: list[dict[str, Any]] = []
bin_method = BIN_METHODS.get(config["bin_method_label"], "fd")
default_unit = max(float(config.get("measurement_unit", 0.1)), MIN_POSITIVE_VALUE)

if show_advanced:
    settings_container = st.sidebar.expander("列ごとの階級設定", expanded=False)
else:
    settings_container = None

for raw_item in raw_items:
    unit = default_unit
    auto_width, auto_start = automatic_bin_settings(
        raw_item["values"], bin_method, unit
    )
    width = auto_width
    start = auto_start

    if settings_container is not None:
        with settings_container:
            st.markdown(f"**{raw_item['label']}**")
            unit = st.number_input(
                "測定単位",
                min_value=MIN_POSITIVE_VALUE,
                value=default_unit,
                step=0.1,
                format="%.6f",
                key=f"unit_{raw_item['id']}",
            )
            auto_width, auto_start = automatic_bin_settings(
                raw_item["values"], bin_method, unit
            )

            manual = st.checkbox(
                "階級幅と開始位置を手動指定",
                value=False,
                key=f"manual_{raw_item['id']}",
            )
            if manual:
                width = st.number_input(
                    "区間幅",
                    min_value=MIN_POSITIVE_VALUE,
                    value=float(auto_width),
                    step=float(unit),
                    format="%.6f",
                    key=f"width_{raw_item['id']}",
                )
                start = st.number_input(
                    "区間開始",
                    value=float(auto_start),
                    step=float(unit),
                    format="%.6f",
                    key=f"start_{raw_item['id']}",
                )
            else:
                width, start = auto_width, auto_start
            st.caption(
                f"自動値：区間幅 {auto_width:.6g} ／ 開始 {auto_start:.6g}"
            )

    try:
        bins = build_bins(raw_item["values"], width, start)
    except ValueError as exc:
        st.sidebar.error(f"{raw_item['label']}：{exc}")
        continue

    item = dict(raw_item)
    item.update(
        {
            "measurement_unit": float(unit),
            "bin_width": float(width),
            "bin_start": float(start),
            "bins": bins,
            "bin_method": config["bin_method_label"],
        }
    )
    items.append(item)

if not items:
    st.error("階級設定が正しくないため、グラフを作成できません。")
    st.stop()


# ============================================================
# 集計表
# ============================================================
quality_rows: list[dict[str, Any]] = []
stats_rows: list[dict[str, Any]] = []

for item in items:
    quality = item["quality"]
    stats = item["stats"]

    quality_rows.append(
        {
            "シート": item["sheet"],
            "列": item["column"],
            "元データ数": quality["original_count"],
            "有効な数値": quality["valid_count"],
            "除外合計": quality["excluded_count"],
            "空欄": quality["missing_count"],
            "数値以外": quality["non_numeric_count"],
            "無限大": quality["infinite_count"],
        }
    )

    stats_rows.append(
        {
            "シート": item["sheet"],
            "列": item["column"],
            "n": int(stats["n"]),
            "平均": stats["mean"],
            "中央値": stats["median"],
            "標本標準偏差": stats["std"],
            "最小値": stats["min"],
            "第1四分位数": stats["q1"],
            "第3四分位数": stats["q3"],
            "四分位範囲": stats["iqr"],
            "最大値": stats["max"],
            "最頻値": stats["mode"],
            "区間幅": item["bin_width"],
            "区間開始": item["bin_start"],
            "区間幅の決定方法": item["bin_method"],
        }
    )

quality_df = pd.DataFrame(quality_rows)
stats_df = pd.DataFrame(stats_rows)


# ============================================================
# メイン画面
# ============================================================
tab_labels = [
    "概要",
    "ヒストグラム",
    "データ一覧",
    "統計量",
    "重ね比較",
]
summary_tab, histogram_tab, data_tab, stats_tab, comparison_tab = st.tabs(
    tab_labels,
    key="main_tabs",
    on_change="rerun",
)


# -------------------------
# 概要
# -------------------------
if summary_tab.open:
    total_valid = int(quality_df["有効な数値"].sum())
    total_excluded = int(quality_df["除外合計"].sum())

    metric_cols = st.columns(4)
    metric_cols[0].metric("分析シート", len({item["sheet"] for item in items}))
    metric_cols[1].metric("分析列", len(items))
    metric_cols[2].metric("有効データ合計", f"{total_valid:,}")
    metric_cols[3].metric("除外データ合計", f"{total_excluded:,}")

    st.subheader("データ品質")
    st.dataframe(
        quality_df,
        width="stretch",
        hide_index=True,
        column_config={
            "元データ数": st.column_config.NumberColumn(format="%d"),
            "有効な数値": st.column_config.NumberColumn(format="%d"),
            "除外合計": st.column_config.NumberColumn(format="%d"),
        },
    )

    if total_excluded:
        st.warning(
            "空欄・数値として解釈できない値・無限大は、ヒストグラムと統計計算から除外しています。"
        )
    else:
        st.success("選択された列は、すべての値を数値として利用できました。")

    st.subheader("基本統計量")
    st.dataframe(
        stats_df[["シート", "列", "n", "平均", "中央値", "標本標準偏差", "最小値", "最大値"]],
        width="stretch",
        hide_index=True,
    )


# -------------------------
# データ一覧
# -------------------------
if data_tab.open:
    st.subheader("有効な数値データ")
    selected_label = st.selectbox(
        "表示する列",
        options=[item["label"] for item in items],
        key="data_item_selector",
    )
    item = next(item for item in items if item["label"] == selected_label)

    selected_value_key = f"selected_value_{item['id']}"
    table_key = f"table_{item['id']}"

    top_left, top_right = st.columns([1, 3])
    with top_left:
        if st.button("選択を解除", key=f"clear_{item['id']}", width="stretch"):
            st.session_state[table_key] = {"selection": {"rows": []}}
            st.session_state[selected_value_key] = None
    with top_right:
        selected_value = st.session_state.get(selected_value_key)
        if selected_value is None:
            st.info("行を1つ選択すると、ヒストグラム上にその値を緑線で表示します。")
        else:
            st.success(f"選択中の値：{selected_value:g}")

    event = st.dataframe(
        item["grid_df"],
        key=table_key,
        on_select="rerun",
        selection_mode="single-row",
        width="stretch",
        height=430,
        hide_index=True,
        column_config={
            "Excel行": st.column_config.NumberColumn("元のExcel行", format="%d"),
            "値": st.column_config.NumberColumn("数値", format="%.8g"),
        },
    )

    rows = list(event.selection.rows)
    if rows:
        st.session_state[selected_value_key] = float(
            item["grid_df"].iloc[rows[0]]["値"]
        )

    st.download_button(
        "この列の有効データをCSVで保存",
        data=item["grid_df"].to_csv(index=False).encode("utf-8-sig"),
        file_name=safe_filename(item["label"], "_data.csv"),
        mime="text/csv",
        on_click="ignore",
        icon=":material/download:",
    )


# -------------------------
# ヒストグラム
# -------------------------
if histogram_tab.open:
    st.subheader("個別ヒストグラム")
    st.caption(
        "緑線を表示するには「データ一覧」タブで行を選択してください。"
    )

    per_row = int(config.get("cards_per_row", 2))
    for start_index in range(0, len(items), per_row):
        columns = st.columns(per_row)
        row_items = items[start_index : start_index + per_row]

        for column_ui, item in zip(columns, row_items):
            with column_ui:
                st.markdown(f"### {item['label']}")

                metric_cols = st.columns(4)
                metric_cols[0].metric("n", f"{int(item['stats']['n']):,}")
                metric_cols[1].metric("平均", format_number(item["stats"]["mean"], 3))
                metric_cols[2].metric("中央値", format_number(item["stats"]["median"], 3))
                metric_cols[3].metric("除外", f"{item['quality']['excluded_count']:,}")

                selected_value = st.session_state.get(
                    f"selected_value_{item['id']}"
                )
                fig = make_histogram_figure(
                    item=item,
                    y_mode=Y_MODES[config["y_mode_label"]],
                    show_mean=bool(config["show_mean"]),
                    show_median=bool(config["show_median"]),
                    show_stats=bool(config["show_stats"]),
                    selected_value=selected_value,
                )
                png_data = figure_to_png(fig)
                st.pyplot(fig, width="stretch")
                plt.close(fig)

                st.download_button(
                    "PNGで保存",
                    data=png_data,
                    file_name=safe_filename(item["label"], "_histogram.png"),
                    mime="image/png",
                    key=f"png_{item['id']}",
                    on_click="ignore",
                    icon=":material/download:",
                )
                st.caption(
                    f"区間幅：{item['bin_width']:.6g}　開始：{item['bin_start']:.6g}"
                )


# -------------------------
# 統計量
# -------------------------
if stats_tab.open:
    st.subheader("統計量")
    st.dataframe(
        stats_df,
        width="stretch",
        hide_index=True,
        column_config={
            "n": st.column_config.NumberColumn(format="%d"),
            "平均": st.column_config.NumberColumn(format="%.6g"),
            "中央値": st.column_config.NumberColumn(format="%.6g"),
            "標本標準偏差": st.column_config.NumberColumn(format="%.6g"),
            "最小値": st.column_config.NumberColumn(format="%.6g"),
            "最大値": st.column_config.NumberColumn(format="%.6g"),
            "区間幅": st.column_config.NumberColumn(format="%.6g"),
            "区間開始": st.column_config.NumberColumn(format="%.6g"),
        },
    )

    csv_data = stats_df.to_csv(index=False).encode("utf-8-sig")
    excel_data = create_excel_report(stats_df, quality_df, items)

    download_cols = st.columns(2)
    with download_cols[0]:
        st.download_button(
            "統計量をCSVで保存",
            data=csv_data,
            file_name="histogram_statistics.csv",
            mime="text/csv",
            on_click="ignore",
            icon=":material/download:",
        )
    with download_cols[1]:
        st.download_button(
            "統計量・品質・有効データをExcelで保存",
            data=excel_data,
            file_name="histogram_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            on_click="ignore",
            icon=":material/download:",
        )


# -------------------------
# 重ね比較
# -------------------------
if comparison_tab.open:
    st.subheader("共通の階級を使った分布比較")
    st.caption(
        "選択したすべての列に同じ区間開始・区間幅を適用するため、棒の位置を正しく比較できます。"
    )

    labels = [item["label"] for item in items]
    default_labels = labels[: min(3, len(labels))]
    selected_labels = st.multiselect(
        "比較する列",
        options=labels,
        default=default_labels,
        key="comparison_items",
    )

    if not selected_labels:
        st.info("比較する列を1つ以上選択してください。")
    else:
        selected_items = [item for item in items if item["label"] in selected_labels]

        control_cols = st.columns(4)
        with control_cols[0]:
            comparison_y_label = st.selectbox(
                "縦軸",
                options=list(Y_MODES),
                index=1 if len(selected_items) >= 2 else 0,
                key="comparison_y_mode",
            )
        with control_cols[1]:
            comparison_style = st.selectbox(
                "表示形式",
                options=["輪郭線", "半透明の棒"],
                key="comparison_style",
            )
        with control_cols[2]:
            comparison_method_label = st.selectbox(
                "共通階級の決定方法",
                options=list(BIN_METHODS),
                index=list(BIN_METHODS).index(config["bin_method_label"]),
                key="comparison_method",
            )
        with control_cols[3]:
            show_comparison_means = st.checkbox(
                "平均線を表示",
                value=False,
                key="comparison_means",
            )

        combined_values = np.concatenate([item["values"] for item in selected_items])
        common_method = BIN_METHODS[comparison_method_label]
        common_width_auto, common_start_auto = automatic_bin_settings(
            combined_values,
            common_method,
            default_unit,
        )

        with st.expander("共通階級を手動調整", expanded=False):
            manual_common = st.checkbox(
                "手動指定する",
                value=False,
                key="manual_common_bins",
            )
            if manual_common:
                common_width = st.number_input(
                    "共通区間幅",
                    min_value=MIN_POSITIVE_VALUE,
                    value=float(common_width_auto),
                    step=float(default_unit),
                    format="%.6f",
                    key="common_width",
                )
                common_start = st.number_input(
                    "共通区間開始",
                    value=float(common_start_auto),
                    step=float(default_unit),
                    format="%.6f",
                    key="common_start",
                )
            else:
                common_width = common_width_auto
                common_start = common_start_auto
            st.caption(
                f"自動値：区間幅 {common_width_auto:.6g} ／ 開始 {common_start_auto:.6g}"
            )

        try:
            common_bins = build_bins(combined_values, common_width, common_start)
        except ValueError as exc:
            st.error(str(exc))
        else:
            comparison_fig = make_comparison_figure(
                items=selected_items,
                bins=common_bins,
                y_mode=Y_MODES[comparison_y_label],
                style=comparison_style,
                show_means=show_comparison_means,
            )
            comparison_png = figure_to_png(comparison_fig)
            st.pyplot(comparison_fig, width="stretch")
            plt.close(comparison_fig)

            st.download_button(
                "比較グラフをPNGで保存",
                data=comparison_png,
                file_name="histogram_comparison.png",
                mime="image/png",
                on_click="ignore",
                icon=":material/download:",
            )

            st.caption(
                f"共通区間幅：{common_width:.6g}　共通開始：{common_start:.6g}　区間数：{len(common_bins)-1:,}"
            )

            comparison_stats = stats_df[
                stats_df.apply(
                    lambda row: f"{row['シート']} / {row['列']}" in selected_labels,
                    axis=1,
                )
            ][["シート", "列", "n", "平均", "中央値", "標本標準偏差", "最小値", "最大値"]]
            st.dataframe(comparison_stats, width="stretch", hide_index=True)


# ============================================================
# フッター
# ============================================================
st.sidebar.markdown("---")
st.sidebar.caption(
    "設定を変更したら「設定を適用」を押してください。詳細な階級設定は即時反映されます。"
)
