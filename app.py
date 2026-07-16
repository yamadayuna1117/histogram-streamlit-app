import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode


# ==================================================
# 基本設定
# ==================================================
st.set_page_config(layout="wide")

# 日本語フォント。環境に存在するものが優先して使われます。
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

MIN_POSITIVE_VALUE = 1e-6
MAX_BIN_COUNT = 5000


def darker_color(color, factor=0.7):
    """MatplotlibのRGB色を指定倍率で濃くする。"""
    return tuple(max(0.0, min(1.0, component * factor)) for component in color)


def reset_number_input(key, value):
    """number_inputの値をリセットするコールバック。"""
    st.session_state[key] = float(value)


def get_selected_value(grid_response, column):
    """st-aggridのバージョン差を吸収して選択値を取得する。"""
    selected_rows = grid_response.get("selected_rows")

    if isinstance(selected_rows, pd.DataFrame):
        if selected_rows.empty:
            return None
        value = selected_rows.iloc[0][column]
    elif isinstance(selected_rows, list):
        if not selected_rows:
            return None
        value = selected_rows[0].get(column)
    else:
        return None

    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value) or not np.isfinite(numeric_value):
        return None

    return float(numeric_value)


# ==================================================
# CSS
# ==================================================
st.markdown(
    """
    <style>
    .main {background-color: #FFF0F5;}
    h1 {color: #FF69B4;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# タイトル
# ==================================================
st.title("📊 ヒストグラム作成ツール")
st.markdown("Excelデータからヒストグラムを作成できます。")


# ==================================================
# サイドバー：ファイルとシート選択
# ==================================================
st.sidebar.header("⚙️ 設定")

uploaded_file = st.sidebar.file_uploader(
    "📥 Excel ファイルをアップロード (.xlsx)",
    type=["xlsx"],
)

if uploaded_file is None:
    st.info("左のサイドバーから Excel ファイルをアップロードしてください")
    st.stop()

try:
    xlsx = pd.ExcelFile(uploaded_file)
except Exception as exc:
    st.error(f"Excelファイルを読み込めませんでした: {exc}")
    st.stop()

sheets = st.sidebar.multiselect(
    "📄 シートを選択",
    xlsx.sheet_names,
)

if not sheets:
    st.info("シートを選択してください")
    st.stop()


# ==================================================
# ヒストグラム情報を集約
# ==================================================
plot_items = []
overlay_items = []

for sheet in sheets:
    try:
        df = pd.read_excel(xlsx, sheet_name=sheet)
    except Exception as exc:
        st.warning(f"シート「{sheet}」を読み込めませんでした: {exc}")
        continue

    st.sidebar.markdown("---")
    st.sidebar.subheader(f"📄 {sheet}")

    columns = st.sidebar.multiselect(
        f"📌 列選択（{sheet}）",
        df.columns,
        key=f"columns_{sheet}",
    )

    if not columns:
        continue

    for column in columns:
        # 文字列・空欄・無限大を除外し、有限の数値だけを使用する。
        numeric_series = (
            pd.to_numeric(df[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .astype(float)
        )
        dataset = numeric_series.to_numpy()
        grid_df = numeric_series.to_frame(name=column)
        n = len(dataset)

        if n == 0:
            st.sidebar.warning(f"{sheet}/{column}: 使用できる数値データがありません")
            continue

        # -----------------------------
        # サイドバー：数値設定
        # -----------------------------
        m = st.sidebar.number_input(
            f"測定単位（{sheet}/{column}）",
            min_value=MIN_POSITIVE_VALUE,
            value=0.1,
            step=0.1,
            format="%.6f",
            key=f"measure_{sheet}_{column}",
        )

        x_min = float(np.min(dataset))
        x_max = float(np.max(dataset))
        data_range = x_max - x_min

        # 平方根選択に基づく自動区間幅。
        c_raw = data_range / math.sqrt(n)
        c_auto = max(m, round(c_raw / m) * m)

        bin_width_key = f"binwidth_{sheet}_{column}"
        if bin_width_key not in st.session_state:
            st.session_state[bin_width_key] = float(c_auto)

        c = st.sidebar.number_input(
            f"区間幅（{sheet}/{column}）",
            min_value=MIN_POSITIVE_VALUE,
            step=0.1,
            format="%.6f",
            key=bin_width_key,
        )

        st.sidebar.button(
            "区間幅リセット",
            key=f"reset_{sheet}_{column}",
            on_click=reset_number_input,
            args=(bin_width_key, c_auto),
        )

        # 自動区間開始：最小値より測定単位の半分だけ小さい位置。
        bin_start_auto = x_min - m / 2
        bin_start_key = f"binstart_{sheet}_{column}"

        if bin_start_key not in st.session_state:
            st.session_state[bin_start_key] = float(bin_start_auto)

        bin_start = st.sidebar.number_input(
            f"区間開始（{sheet}/{column}）",
            step=float(m),
            format="%.6f",
            key=bin_start_key,
        )

        st.sidebar.button(
            "区間開始リセット",
            key=f"reset_binstart_{sheet}_{column}",
            on_click=reset_number_input,
            args=(bin_start_key, bin_start_auto),
        )

        # -----------------------------
        # bins
        # -----------------------------
        if bin_start >= x_max:
            st.sidebar.error(
                f"{sheet}/{column}: 区間開始は最大値 {x_max:.6f} より小さくしてください"
            )
            continue

        bin_count = max(1, math.ceil((x_max - bin_start) / c))

        if bin_count > MAX_BIN_COUNT:
            st.sidebar.error(
                f"{sheet}/{column}: 区間数が {bin_count:,} 個になります。"
                f"区間幅を大きくしてください（上限 {MAX_BIN_COUNT:,} 個）。"
            )
            continue

        bins = bin_start + np.arange(bin_count + 1, dtype=float) * c

        # 浮動小数点誤差で末尾が最大値未満になった場合に1区間追加する。
        if bins[-1] < x_max:
            bins = np.append(bins, bins[-1] + c)

        if bin_start > x_min:
            st.sidebar.caption(
                f"⚠️ {sheet}/{column}: 最小値未満のデータがヒストグラムから除外されます"
            )

        # データが1件の場合、標本標準偏差は定義できない。
        sample_std = float(np.std(dataset, ddof=1)) if n >= 2 else None

        plot_items.append(
            {
                "sheet": sheet,
                "column": column,
                "grid_df": grid_df,
                "dataset": dataset,
                "bins": bins,
                "n": n,
                "x_min": x_min,
                "x_max": x_max,
                "x_bar": float(np.mean(dataset)),
                "s": sample_std,
                "c": float(c),
            }
        )


# ==================================================
# 横並び描画（シートまたぎOK）
# ==================================================
if not plot_items:
    st.warning("表示できるデータがありません")
    st.stop()

st.markdown("---")
st.subheader("📈 ヒストグラム一覧")

for i in range(0, len(plot_items), 2):
    col_a, col_b = st.columns(2)

    for col_ui, item in zip([col_a, col_b], plot_items[i : i + 2]):
        with col_ui:
            st.markdown(f"### {item['sheet']} / {item['column']}")

            # -----------------------------
            # AgGrid
            # -----------------------------
            gb = GridOptionsBuilder.from_dataframe(item["grid_df"])
            gb.configure_selection(selection_mode="single")
            gb.configure_grid_options(
                onRowClicked=JsCode(
                    """
                    function(e) {
                        const rowNode = e.node;
                        rowNode.setSelected(!rowNode.isSelected());
                    }
                    """
                )
            )

            grid_response = AgGrid(
                item["grid_df"],
                gridOptions=gb.build(),
                height=200,
                fit_columns_on_grid_load=True,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                theme="alpine",
                allow_unsafe_jscode=True,
            )

            selected_value = get_selected_value(grid_response, item["column"])

            # -----------------------------
            # 表示設定
            # -----------------------------
            c1, c2, c3 = st.columns([0.3, 0.3, 0.7])

            with c1:
                show_mean = st.checkbox(
                    "平均",
                    value=True,
                    key=f"show_mean_{item['sheet']}_{item['column']}",
                )

            with c2:
                show_stat = st.checkbox(
                    "統計",
                    value=True,
                    key=f"show_stat_{item['sheet']}_{item['column']}",
                )

            with c3:
                add_overlay = st.checkbox(
                    "重ね表示に追加",
                    key=f"overlay_{item['sheet']}_{item['column']}",
                )

            if add_overlay:
                overlay_items.append(item)

            # -----------------------------
            # ヒストグラム
            # -----------------------------
            fig, ax = plt.subplots(figsize=(5, 3))

            ax.hist(
                item["dataset"],
                bins=item["bins"],
                edgecolor="#1E5BA1",
                linewidth=1.2,
                color="#87CEFA",
                alpha=0.7,
            )

            if show_mean:
                ax.axvline(
                    item["x_bar"],
                    color="#E65656",
                    linestyle="--",
                    linewidth=1,
                    label="平均",
                )

            if selected_value is not None:
                ax.axvline(
                    selected_value,
                    color="green",
                    linewidth=1.2,
                    label="選択値",
                )

            if show_stat:
                s_text = f"{item['s']:.2f}" if item["s"] is not None else "算出不可"
                stat_text = (
                    f"n = {item['n']}\n"
                    f"c = {item['c']:.2f}\n"
                    f"min = {item['x_min']:.2f}\n"
                    f"max = {item['x_max']:.2f}\n"
                    f"μ = {item['x_bar']:.2f}\n"
                    f"s = {s_text}"
                )
                ax.text(
                    0.98,
                    0.95,
                    stat_text,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.6,
                        "edgecolor": "none",
                        "boxstyle": "round,pad=0.3",
                    },
                )

            ax.set_xlabel("値")
            ax.set_ylabel("度数")
            ax.grid(True, linestyle="--", alpha=0.3)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(frameon=False)

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ==================================================
# 選択ヒストグラムの重ね表示
# ==================================================
st.markdown("---")
st.subheader("📊 選択ヒストグラムの重ね表示")

if overlay_items:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.tab10.colors

    for i, item in enumerate(overlay_items):
        base_color = colors[i % len(colors)]
        edge_color = darker_color(base_color, factor=0.6)

        ax.hist(
            item["dataset"],
            bins=item["bins"],
            alpha=0.4,
            linewidth=1.2,
            color=base_color,
            edgecolor=edge_color,
            label=f"{item['sheet']} / {item['column']}",
        )

    ax.set_xlabel("値")
    ax.set_ylabel("度数")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
else:
    st.info("重ね表示するヒストグラムを選択してください")
