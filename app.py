import os
import json
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from openai import OpenAI


# 基础设置

load_dotenv()

app = Flask(__name__, static_folder="static")
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

DATA_PATH = Path("data") / "Results_21Mar2022.csv"
TREEMAP_DATA_PATH = Path("data") / "treemap_data.csv"
RADAR_DATA_PATH = Path("data") / "radar_data.csv"
SCATTER_DATA_PATH = Path("data") / "scatter_matrix_data.csv"

CACHED_DF = None
AI_DATA_MEMORY = ""


# 数据字段设置

DIET_ORDER = [
    "High meat", "Medium meat", "Low meat",
    "Fish eater", "Vegetarian", "Vegan"
]

AGE_ORDER = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"]

DIET_NAME_MAP = {
    "meat100": "High meat", "meat": "Medium meat", "meat50": "Low meat",
    "fish": "Fish eater", "veggie": "Vegetarian", "vegan": "Vegan",
    "high meat": "High meat", "medium meat": "Medium meat", "low meat": "Low meat",
    "fish eater": "Fish eater", "fish eaters": "Fish eater",
    "vegetarian": "Vegetarian", "vegetarians": "Vegetarian", "vegans": "Vegan"
}

IMPACT_FIELDS = {
    "GHG Emissions": "mean_ghgs", "CH4 Emissions": "mean_ghgs_ch4",
    "N2O Emissions": "mean_ghgs_n2o", "Land Use": "mean_land",
    "Water Use": "mean_watuse", "Water Scarcity": "mean_watscar",
    "Eutrophication": "mean_eut", "Acidification": "mean_acid",
    "Biodiversity Impact": "mean_bio"
}

ALL_INDICATORS = list(IMPACT_FIELDS.keys())

# Tableau 导出的字段名和项目中使用的正式指标名之间的映射
TABLEAU_INDICATOR_MAP = {
    "mean_ghgs": "GHG Emissions",
    "mean_ghgs_ch4": "CH4 Emissions",
    "mean_ghgs_n2o": "N2O Emissions",
    "mean_land": "Land Use",
    "mean_watuse": "Water Use",
    "mean_watscar": "Water Scarcity",
    "mean_eut": "Eutrophication",
    "mean_acid": "Acidification",
    "mean_bio": "Biodiversity Impact"
}

SCATTER_EXPORT_FIELDS = {
    "mean_ghgs": "GHG Emissions",
    "mean_ghgs_ch4": "CH4 Emissions",
    "mean_ghgs_n2o": "N2O Emissions",
    "mean_land": "Land Use",
    "mean_watuse": "Water Use",
    "mean_watscar": "Water Scarcity",
    "mean_eut": "Eutrophication",
    "mean_acid": "Acidification",
    "mean_bio": "Biodiversity Impact"
}


# 一些工具函数

def clean_column_name(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def round_number(value, digits=3):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def mean_or_none(series):
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def map_diet_name(value):
    key = str(value).strip().lower()
    if key in DIET_NAME_MAP:
        return DIET_NAME_MAP[key]
    return str(value).strip()


def parse_grouping(value):
    parts = str(value).strip().lower().split("_")
    result = {"diet_group": None, "sex": None, "age_group": None}
    if len(parts) >= 3:
        result["diet_group"] = map_diet_name(parts[0])
        result["sex"] = parts[1]
        result["age_group"] = parts[2]
    return result


def extract_json_from_text(text):
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {
        "title": "AI Generated Chart",
        "caption": "AI-generated visualisation.",
        "image_prompt": text
    }


# 读取 CSV 数据

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Cannot find data/Results_21Mar2022.csv. "
            "Please put the CSV file inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    new_columns = []
    for col in df.columns:
        new_columns.append(clean_column_name(col))
    df.columns = new_columns

    if "grouping" in df.columns:
        parsed = []
        for i in range(len(df)):
            parsed.append(parse_grouping(df.iloc[i]["grouping"]))
        parsed_df = pd.DataFrame(parsed)

        if "diet_group" not in df.columns:
            diet_list = []
            for i in range(len(parsed_df)):
                diet_list.append(parsed_df.iloc[i]["diet_group"])
            df["diet_group"] = diet_list

        if "sex" not in df.columns:
            sex_list = []
            for i in range(len(parsed_df)):
                sex_list.append(parsed_df.iloc[i]["sex"])
            df["sex"] = sex_list

        if "age_group" not in df.columns:
            age_list = []
            for i in range(len(parsed_df)):
                age_list.append(parsed_df.iloc[i]["age_group"])
            df["age_group"] = age_list

    for col in ["diet_group", "sex", "age_group"]:
        if col not in df.columns:
            df[col] = "Unknown"

    # 把原始数据里的饮食名称统一成 dashboard 使用的名称
    new_diet_names = []
    for i in range(len(df)):
        old_name = df.iloc[i]["diet_group"]
        new_diet_names.append(map_diet_name(old_name))
    df["diet_group"] = new_diet_names

    for field in IMPACT_FIELDS.values():
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")

    return df


def get_cached_df():
    global CACHED_DF
    if CACHED_DF is None:
        CACHED_DF = load_data()
    return CACHED_DF


# 计算图表需要的数据

def compute_bar_data(df, selected_indicator):
    field = IMPACT_FIELDS.get(selected_indicator, "mean_ghgs")
    result = []

    for diet in DIET_ORDER:
        subset = df[df["diet_group"] == diet]

        if subset.empty:
            continue

        if field not in subset.columns:
            continue

        value = mean_or_none(subset[field])

        row = {
            "diet_group": diet,
            "selected_indicator": selected_indicator,
            "selected_impact_value": round_number(value, 3)
        }
        result.append(row)

    # 让数值高的饮食排在前面，没有数值的放最后
    def get_sort_value(item):
        v = item["selected_impact_value"]
        if v is None:
            return -999
        return v

    result.sort(key=get_sort_value, reverse=True)
    return result


def compute_radar_data(df):
    # 先用 Tableau 导出的数据，因为这样和 dashboard 上的雷达图保持一致
    if RADAR_DATA_PATH.exists():
        radar_df = pd.read_csv(RADAR_DATA_PATH)

        new_columns = []
        for col in radar_df.columns:
            new_columns.append(clean_column_name(col))
        radar_df.columns = new_columns

        required_cols = ["diet_group", "indicator", "avg_impact", "relative_impact"]
        missing_cols = []
        for col in required_cols:
            if col not in radar_df.columns:
                missing_cols.append(col)

        if len(missing_cols) > 0:
            raise ValueError(
                "radar_data.csv is missing columns: " + ", ".join(missing_cols)
            )

        # 把这里的饮食名称统一成 dashboard 使用的名称
        new_diet_list = []
        for i in range(len(radar_df)):
            old_name = radar_df.iloc[i]["diet_group"]
            new_diet_list.append(map_diet_name(old_name))
        radar_df["diet_group"] = new_diet_list

        # 把 Tableau 字段名换成 dashboard 上显示的指标名称
        new_indicator_list = []
        for i in range(len(radar_df)):
            old_ind = radar_df.iloc[i]["indicator"]
            cleaned = clean_column_name(old_ind)
            if cleaned in TABLEAU_INDICATOR_MAP:
                new_indicator_list.append(TABLEAU_INDICATOR_MAP[cleaned])
            else:
                new_indicator_list.append(str(old_ind).strip())
        radar_df["indicator"] = new_indicator_list

        radar_df["avg_impact"] = pd.to_numeric(radar_df["avg_impact"], errors="coerce")
        radar_df["relative_impact"] = pd.to_numeric(radar_df["relative_impact"], errors="coerce")

        grouped = (
            radar_df
            .groupby(["diet_group", "indicator"], dropna=False)
            .agg(
                average_value=("avg_impact", "mean"),
                relative_impact=("relative_impact", "mean")
            )
            .reset_index()
        )

        # 给饮食组和其他分类一个固定顺序，显示时不会乱
        diet_order_list = []
        for i in range(len(grouped)):
            dg = grouped.iloc[i]["diet_group"]
            if dg in DIET_ORDER:
                diet_order_list.append(DIET_ORDER.index(dg))
            else:
                diet_order_list.append(999)

        indicator_order_list = []
        for i in range(len(grouped)):
            ind = grouped.iloc[i]["indicator"]
            if ind in ALL_INDICATORS:
                indicator_order_list.append(ALL_INDICATORS.index(ind))
            else:
                indicator_order_list.append(999)

        grouped["diet_order"] = diet_order_list
        grouped["indicator_order"] = indicator_order_list
        grouped = grouped.sort_values(by=["diet_order", "indicator_order"])

        result = []
        for i in range(len(grouped)):
            row = grouped.iloc[i]
            result.append({
                "diet_group": row["diet_group"],
                "indicator": row["indicator"],
                "average_value": round_number(row["average_value"], 3),
                "relative_impact": round_number(row["relative_impact"], 4)
            })

        return result

    # 如果没有对应的 Tableau 导出文件，就直接从原始 CSV 计算
    raw_values = {}
    for diet in DIET_ORDER:
        subset = df[df["diet_group"] == diet]
        if subset.empty:
            continue
        raw_values[diet] = {}
        for indicator, field in IMPACT_FIELDS.items():
            if field in subset.columns:
                raw_values[diet][indicator] = mean_or_none(subset[field])
            else:
                raw_values[diet][indicator] = None

    result = []
    for indicator in ALL_INDICATORS:
        values = []
        for diet in DIET_ORDER:
            if diet in raw_values:
                v = raw_values[diet].get(indicator)
                if v is not None:
                    values.append(v)

        if len(values) == 0:
            continue

        max_value = max(values)

        for diet in DIET_ORDER:
            if diet not in raw_values:
                continue
            value = raw_values[diet].get(indicator)
            if value is None:
                continue
            if max_value == 0:
                relative = 0
            else:
                relative = value / max_value
            result.append({
                "diet_group": diet,
                "indicator": indicator,
                "average_value": round_number(value, 3),
                "relative_impact": round_number(relative, 4)
            })

    return result


def compute_treemap_data(df):
    # 如果有 Tableau 导出的 treemap 数据就先使用它
    if TREEMAP_DATA_PATH.exists():
        treemap_df = pd.read_csv(TREEMAP_DATA_PATH)

        new_columns = []
        for col in treemap_df.columns:
            new_columns.append(clean_column_name(col))
        treemap_df.columns = new_columns

        required_cols = [
            "age_group",
            "diet_group",
            "impact_category",
            "impact_value",
            "normalized_impact"
        ]
        missing_cols = []
        for col in required_cols:
            if col not in treemap_df.columns:
                missing_cols.append(col)

        if len(missing_cols) > 0:
            raise ValueError(
                "treemap_data.csv is missing columns: " + ", ".join(missing_cols)
            )

        # 把这里的饮食名称统一成 dashboard 使用的名称
        new_diet_list = []
        for i in range(len(treemap_df)):
            old_name = treemap_df.iloc[i]["diet_group"]
            new_diet_list.append(map_diet_name(old_name))
        treemap_df["diet_group"] = new_diet_list

        treemap_df["impact_value"] = pd.to_numeric(treemap_df["impact_value"], errors="coerce")
        treemap_df["normalized_impact"] = pd.to_numeric(treemap_df["normalized_impact"], errors="coerce")

        grouped = (
            treemap_df
            .groupby(["diet_group", "age_group", "impact_category"], dropna=False)
            .agg(
                average_value=("impact_value", "mean"),
                normalized_impact=("normalized_impact", "mean"),
                number_of_rows=("normalized_impact", "count")
            )
            .reset_index()
        )

        # 给饮食组和其他分类一个固定顺序，显示时不会乱
        diet_order_list = []
        for i in range(len(grouped)):
            dg = grouped.iloc[i]["diet_group"]
            if dg in DIET_ORDER:
                diet_order_list.append(DIET_ORDER.index(dg))
            else:
                diet_order_list.append(999)

        age_order_list = []
        for i in range(len(grouped)):
            ag = str(grouped.iloc[i]["age_group"])
            if ag in AGE_ORDER:
                age_order_list.append(AGE_ORDER.index(ag))
            else:
                age_order_list.append(999)

        grouped["diet_order"] = diet_order_list
        grouped["age_order"] = age_order_list
        grouped = grouped.sort_values(by=["diet_order", "age_order", "impact_category"])

        result = []
        for i in range(len(grouped)):
            row = grouped.iloc[i]
            result.append({
                "diet_group": row["diet_group"],
                "age_group": row["age_group"],
                "impact_category": row["impact_category"],
                "average_value": round_number(row["average_value"], 3),
                "normalized_impact": round_number(row["normalized_impact"], 4),
                "number_of_source_rows": int(row["number_of_rows"])
            })

        return result

    # 如果没有 treemap_data.csv，就从原始数据自己计算
    long_rows = []
    for indicator, field in IMPACT_FIELDS.items():
        if field not in df.columns:
            continue

        grouped = df.groupby(["diet_group", "age_group"], dropna=False)[field].mean().reset_index()
        grouped["impact_category"] = indicator
        grouped["average_value"] = grouped[field]

        for i in range(len(grouped)):
            row = grouped.iloc[i]
            long_rows.append({
                "diet_group": row["diet_group"],
                "age_group": row["age_group"],
                "impact_category": row["impact_category"],
                "average_value": row["average_value"]
            })

    if len(long_rows) == 0:
        return []

    long_df = pd.DataFrame(long_rows)

    # 把不同环境指标的数值转成 0 到 1，方便 treemap 比较
    min_val = long_df["average_value"].min()
    max_val = long_df["average_value"].max()

    norm_values = []
    for i in range(len(long_df)):
        val = long_df.iloc[i]["average_value"]
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            norm_values.append(0)
        else:
            norm_values.append((val - min_val) / (max_val - min_val))

    long_df["normalized_impact"] = norm_values

    # 按饮食组和年龄组的正常顺序整理结果
    diet_order_list = []
    for i in range(len(long_df)):
        dg = long_df.iloc[i]["diet_group"]
        if dg in DIET_ORDER:
            diet_order_list.append(DIET_ORDER.index(dg))
        else:
            diet_order_list.append(999)

    age_order_list = []
    for i in range(len(long_df)):
        ag = str(long_df.iloc[i]["age_group"])
        if ag in AGE_ORDER:
            age_order_list.append(AGE_ORDER.index(ag))
        else:
            age_order_list.append(999)

    long_df["diet_order"] = diet_order_list
    long_df["age_order"] = age_order_list
    long_df = long_df.sort_values(by=["diet_order", "age_order", "impact_category"])

    result = []
    for i in range(len(long_df)):
        row = long_df.iloc[i]
        result.append({
            "diet_group": row["diet_group"],
            "age_group": row["age_group"],
            "impact_category": row["impact_category"],
            "average_value": round_number(row["average_value"], 3),
            "normalized_impact": round_number(row["normalized_impact"], 4)
        })

    return result


def compute_scatter_matrix_data(df):
    # scatter matrix 也优先使用 Tableau 导出的结果
    if SCATTER_DATA_PATH.exists():
        scatter_df = pd.read_csv(SCATTER_DATA_PATH)

        new_columns = []
        for col in scatter_df.columns:
            new_columns.append(clean_column_name(col))
        scatter_df.columns = new_columns

        required_cols = ["diet_group", "sex", "age_group"] + list(SCATTER_EXPORT_FIELDS.keys())
        missing_cols = []
        for col in required_cols:
            if col not in scatter_df.columns:
                missing_cols.append(col)

        if len(missing_cols) > 0:
            raise ValueError(
                "scatter_matrix_data.csv is missing columns: " + ", ".join(missing_cols)
            )

        # 把这里的饮食名称统一成 dashboard 使用的名称
        new_diet_list = []
        for i in range(len(scatter_df)):
            old_name = scatter_df.iloc[i]["diet_group"]
            new_diet_list.append(map_diet_name(old_name))
        scatter_df["diet_group"] = new_diet_list

        for col in SCATTER_EXPORT_FIELDS.keys():
            scatter_df[col] = pd.to_numeric(scatter_df[col], errors="coerce")

        grouped = (
            scatter_df
            .groupby(["diet_group", "age_group", "sex"], dropna=False)[list(SCATTER_EXPORT_FIELDS.keys())]
            .mean()
            .reset_index()
        )

        # 按饮食组和年龄组的正常顺序整理结果
        diet_order_list = []
        for i in range(len(grouped)):
            dg = grouped.iloc[i]["diet_group"]
            if dg in DIET_ORDER:
                diet_order_list.append(DIET_ORDER.index(dg))
            else:
                diet_order_list.append(999)

        age_order_list = []
        for i in range(len(grouped)):
            ag = str(grouped.iloc[i]["age_group"])
            if ag in AGE_ORDER:
                age_order_list.append(AGE_ORDER.index(ag))
            else:
                age_order_list.append(999)

        grouped["diet_order"] = diet_order_list
        grouped["age_order"] = age_order_list
        grouped = grouped.sort_values(by=["diet_order", "age_order", "sex"])

        result = []
        for i in range(len(grouped)):
            row = grouped.iloc[i]
            item = {
                "diet_group": row["diet_group"],
                "age_group": row["age_group"],
                "sex": row["sex"]
            }
            for source_col, output_name in SCATTER_EXPORT_FIELDS.items():
                item[output_name] = round_number(row[source_col], 3)
            result.append(item)

        return result

    # 如果没有对应的 Tableau 导出文件，就直接从原始 CSV 计算
    fields = []
    for f in IMPACT_FIELDS.values():
        if f in df.columns:
            fields.append(f)

    if len(fields) == 0:
        return []

    grouped = df.groupby(["diet_group", "age_group", "sex"], dropna=False)[fields].mean().reset_index()

    # 按饮食组和年龄组的正常顺序整理结果
    diet_order_list = []
    for i in range(len(grouped)):
        dg = grouped.iloc[i]["diet_group"]
        if dg in DIET_ORDER:
            diet_order_list.append(DIET_ORDER.index(dg))
        else:
            diet_order_list.append(999)

    age_order_list = []
    for i in range(len(grouped)):
        ag = str(grouped.iloc[i]["age_group"])
        if ag in AGE_ORDER:
            age_order_list.append(AGE_ORDER.index(ag))
        else:
            age_order_list.append(999)

    grouped["diet_order"] = diet_order_list
    grouped["age_order"] = age_order_list
    grouped = grouped.sort_values(by=["diet_order", "age_order", "sex"])

    result = []
    for i in range(len(grouped)):
        row = grouped.iloc[i]
        item = {
            "diet_group": row["diet_group"],
            "age_group": row["age_group"],
            "sex": row["sex"]
        }
        for indicator, field in IMPACT_FIELDS.items():
            if field in grouped.columns:
                item[indicator] = round_number(row[field], 3)
        result.append(item)

    return result


# Dashboard context 和 AI memory

def _round_df(df, cols, digits=3):
    # 把摘要里的数字统一保留相同的小数位
    for c in cols:
        if c in df.columns:
            new_vals = []
            for i in range(len(df)):
                new_vals.append(round_number(df.iloc[i][c], digits))
            df[c] = new_vals
    return df


def build_dashboard_context(df, selected_indicator):
    bar_data = compute_bar_data(df, selected_indicator)
    radar_data = compute_radar_data(df)
    treemap_data = compute_treemap_data(df)
    scatter_data = compute_scatter_matrix_data(df)

    # Treemap 的摘要
    treemap_df = pd.DataFrame(treemap_data)
    treemap_top = []
    treemap_by_diet = []
    treemap_by_category = []

    if len(treemap_df) > 0:
        treemap_df_sorted = treemap_df.sort_values("normalized_impact", ascending=False)
        top_30 = treemap_df_sorted.head(30)
        for i in range(len(top_30)):
            treemap_top.append(top_30.iloc[i].to_dict())

        td = treemap_df.groupby("diet_group", dropna=False).agg(
            mean_normalized_impact=("normalized_impact", "mean"),
            max_normalized_impact=("normalized_impact", "max"),
            number_of_rectangles=("normalized_impact", "count")
        ).reset_index()
        td = _round_df(td, ["mean_normalized_impact", "max_normalized_impact"])
        for i in range(len(td)):
            treemap_by_diet.append(td.iloc[i].to_dict())

        tc = treemap_df.groupby("impact_category", dropna=False).agg(
            mean_normalized_impact=("normalized_impact", "mean"),
            max_normalized_impact=("normalized_impact", "max")
        ).reset_index()
        tc = _round_df(tc, ["mean_normalized_impact", "max_normalized_impact"])
        for i in range(len(tc)):
            treemap_by_category.append(tc.iloc[i].to_dict())

    # Radar chart 的摘要
    radar_df = pd.DataFrame(radar_data)
    radar_by_diet = []
    radar_by_indicator = []

    if len(radar_df) > 0:
        rd = radar_df.groupby("diet_group", dropna=False).agg(
            mean_relative_impact=("relative_impact", "mean"),
            max_relative_impact=("relative_impact", "max"),
            number_of_axes=("relative_impact", "count")
        ).reset_index()
        rd = _round_df(rd, ["mean_relative_impact", "max_relative_impact"])
        for i in range(len(rd)):
            radar_by_diet.append(rd.iloc[i].to_dict())

        radar_sorted = radar_df.sort_values("relative_impact", ascending=False)
        radar_grouped = radar_sorted.groupby("indicator", dropna=False).head(2).reset_index(drop=True)
        for i in range(len(radar_grouped)):
            radar_by_indicator.append(radar_grouped.iloc[i].to_dict())

    # Scatter matrix 的摘要
    scatter_df = pd.DataFrame(scatter_data)
    scatter_correlation_examples = []

    if len(scatter_df) > 0:
        available = []
        for ind in ALL_INDICATORS:
            if ind in scatter_df.columns:
                available.append(ind)

        corr_rows = []

        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                x = available[i]
                y = available[j]
                valid = scatter_df[[x, y]].dropna()
                if len(valid) >= 3:
                    corr_val = valid[x].corr(valid[y])
                    corr_rows.append({
                        "x_indicator": x,
                        "y_indicator": y,
                        "correlation": round_number(corr_val, 3)
                    })

        # 相关系数绝对值越大，就越值得放到摘要里
        def abs_sort_key(x):
            if x["correlation"] is None:
                return -1
            return abs(x["correlation"])

        corr_rows.sort(key=abs_sort_key, reverse=True)
        scatter_correlation_examples = corr_rows[:15]

    # 这些基本信息会一起发给 AI，帮助它理解整个数据集
    sex_values = df["sex"].dropna().unique().tolist()
    sex_str_list = []
    for s in sex_values:
        sex_str_list.append(str(s))
    sex_str_list.sort()

    dataset_summary = {
        "number_of_rows": int(len(df)),
        "diet_groups": DIET_ORDER,
        "age_groups": AGE_ORDER,
        "sex_groups": sex_str_list,
        "environmental_indicators": ALL_INDICATORS
    }

    return {
        "dashboard_scope": "Full Dashboard",
        "selected_indicator": selected_indicator,
        "dataset_summary": dataset_summary,
        "bar_chart_context": {
            "what_it_shows": (
                "The bar chart compares diet groups by the selected environmental indicator. "
                "Each bar is the average selected_impact_value for one diet group."
            ),
            "selected_indicator": selected_indicator,
            "data": bar_data
        },
        "treemap_context": {
            "what_it_shows": (
                "The treemap shows environmental impact using nested rectangles. "
                "It is not a simple bar chart."
            ),
            "hierarchy": "Diet Group -> Age Group -> Impact Category",
            "rectangle_meaning": "Each rectangle represents one combination of Diet Group, Age Group, and Impact Category.",
            "size_encoding": "Rectangle size represents normalized_impact.",
            "colour_encoding": "Rectangle colour represents normalized_impact.",
            "normalization_explanation": (
                "normalized_impact is the same field shown in the Tableau treemap tooltip. "
                "When treemap_data.csv is available, it is read directly from the Tableau-exported data and aggregated to the rectangle level. "
                "A value near 1 means higher relative impact in the treemap, while a value near 0 means lower relative impact. "
                "It is not the original raw environmental value."
            ),
            "diet_group_summary": treemap_by_diet,
            "impact_category_summary": treemap_by_category,
            "top_normalized_impact_records": treemap_top
        },
        "radar_chart_context": {
            "what_it_shows": "The radar chart compares diet groups across nine environmental indicators. Each axis is one environmental indicator.",
            "axis_meaning": "Each radar axis is one environmental impact indicator.",
            "polygon_meaning": "Each polygon represents one diet group.",
            "radius_meaning": "relative_impact is used as the radius. For each indicator, the highest diet group is scaled to 1.",
            "how_to_read": "A polygon extending further outward on an axis means that diet group has a higher relative impact for that specific indicator.",
            "diet_group_summary": radar_by_diet,
            "top_indicator_records": radar_by_indicator,
            "data": radar_data
        },
        "scatterplot_matrix_context": {
            "what_it_shows": "The scatterplot matrix shows pairwise relationships between the nine environmental indicators. Each small panel compares two indicators.",
            "point_meaning": "Each summarized point represents one Diet Group + Age Group + Sex combination.",
            "variables": ALL_INDICATORS,
            "how_to_read": "If points move upward as they move right, the two indicators have a positive relationship. If points are widely spread, the relationship is weaker. Colour can help compare diet groups.",
            "correlation_examples": scatter_correlation_examples,
            "sample_points": scatter_data[:72]
        }
    }


def initialise_ai_data_memory():
    global CACHED_DF, AI_DATA_MEMORY

    print("Loading CSV data...")
    CACHED_DF = load_data()
    print("CSV loaded successfully. Rows: " + str(len(CACHED_DF)))

    dashboard_context = build_dashboard_context(CACHED_DF, selected_indicator="GHG Emissions")

    # 没有 API Key 时也保留一段简单的数据说明
    if not os.getenv("OPENAI_API_KEY"):
        AI_DATA_MEMORY = (
            "The dataset is about food choice and environmental impact. "
            "Diet groups include High meat, Medium meat, Low meat, Fish eater, Vegetarian, and Vegan. "
            "Environmental indicators include GHG emissions, CH4 emissions, N2O emissions, land use, "
            "water use, water scarcity, eutrophication, acidification, and biodiversity impact."
        )
        print("OPENAI_API_KEY missing. Local memory created instead.")
        return

    memory_prompt = (
        "You are creating a dataset memory for a FoodVis dashboard.\n\n"
        "Dashboard context:\n"
        + json.dumps(dashboard_context, ensure_ascii=False, indent=2)
        + "\n\nWrite a concise dataset memory.\n\n"
        "Include:\n"
        "1. What the dataset is about\n"
        "2. Diet groups\n"
        "3. Environmental indicators\n"
        "4. What the four dashboard views show\n"
        "5. Main overall pattern if supported by the data\n"
        "6. Limitation: this shows association, not direct causation\n\n"
        "Do not invent values.\n"
        "Keep it concise.\n"
    )

    try:
        print("Creating AI data memory...")
        response = client.responses.create(
            model=CHAT_MODEL,
            input=[
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Create a concise and accurate dataset memory. Do not invent facts."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": memory_prompt}]
                }
            ]
        )
        AI_DATA_MEMORY = response.output_text
        print("AI data memory created.")
    except Exception as error:
        AI_DATA_MEMORY = (
            "The dataset is about food choice and environmental impact. "
            "Diet groups include High meat, Medium meat, Low meat, Fish eater, Vegetarian, and Vegan. "
            "Environmental indicators include GHG emissions, CH4 emissions, N2O emissions, land use, "
            "water use, water scarcity, eutrophication, acidification, and biodiversity impact."
        )
        print("AI memory creation failed. Local memory created instead.")
        print(str(error))


# 判断用户的问题类型

def detect_request_mode(question):
    # 只有用户明确说 generate / 生成时才进入生图模式
    q = (question or "").lower().strip()

    generate_keywords = ["generate", "生成", "生成图"]
    found_generate = False
    for keyword in generate_keywords:
        if keyword in q:
            found_generate = True
            break

    if not found_generate:
        return "text"

    bar_keywords = ["bar chart", "bar", "柱状图", "条形图"]
    for keyword in bar_keywords:
        if keyword in q:
            return "image_bar"

    treemap_keywords = ["treemap", "tree map", "树图", "矩形树图"]
    for keyword in treemap_keywords:
        if keyword in q:
            return "image_treemap"

    radar_keywords = ["radar chart", "radar", "雷达图"]
    for keyword in radar_keywords:
        if keyword in q:
            return "image_radar"

    scatter_keywords = ["scatterplot matrix", "scatter plot matrix", "scatter matrix", "散点矩阵"]
    for keyword in scatter_keywords:
        if keyword in q:
            return "image_scatter"

    return "image_summary"


def detect_question_focus(question):
    q = (question or "").lower().strip()

    treemap_keywords = ["treemap", "tree map", "树图", "矩形树图"]
    for keyword in treemap_keywords:
        if keyword in q:
            return "treemap"

    bar_keywords = ["bar chart", "bar", "柱状图", "条形图"]
    for keyword in bar_keywords:
        if keyword in q:
            return "bar_chart"

    radar_keywords = ["radar chart", "radar", "雷达图"]
    for keyword in radar_keywords:
        if keyword in q:
            return "radar_chart"

    scatter_keywords = ["scatter matrix", "scatterplot matrix", "scatter plot matrix", "散点矩阵"]
    for keyword in scatter_keywords:
        if keyword in q:
            return "scatterplot_matrix"

    if "highest" in q or "最高" in q:
        return "highest_impact"

    if "lowest" in q or "最低" in q:
        return "lowest_impact"

    return "general"


# 普通问答的 prompt

def build_text_answer_prompt(question, dashboard_context):
    question_focus = detect_question_focus(question)

    prompt = (
        "You are an AI assistant for a FoodVis dashboard.\n\n"
        "Dataset memory created when the system started:\n"
        + AI_DATA_MEMORY
        + "\n\nQuestion focus:\n"
        + question_focus
        + "\n\nCurrent dashboard context:\n"
        + json.dumps(dashboard_context, ensure_ascii=False, indent=2)
        + "\n\nUser question:\n"
        + question
        + "\n\nAnswer rules:\n"
        "1. Use the dataset memory and dashboard context.\n"
        "2. Do not invent numbers.\n"
        "3. Do not use markdown bold formatting such as **text**.\n"
        "4. Explain clearly for a non-expert user.\n"
        "5. This project shows association, not direct causation.\n"
        "6. Answer the user's actual question directly.\n\n"
        "Chart-specific rules:\n"
        "- If question_focus is bar_chart, use bar_chart_context and explain categories, bar length, selected indicator, and main ranking.\n"
        "- If question_focus is treemap, use treemap_context and explain hierarchy, rectangle size, colour, and normalized_impact. Do not answer mainly as a bar chart.\n"
        "- If question_focus is radar_chart, use radar_chart_context and explain axes, polygons, relative_impact radius, and how to compare diet groups. Do not answer mainly as a bar chart or treemap.\n"
        "- If question_focus is scatterplot_matrix, use scatterplot_matrix_context and explain variables, small panels, points, and relationships. Do not answer mainly as a bar chart, treemap, or radar chart.\n"
        "- If question_focus is highest_impact or lowest_impact, use bar_chart_context first, and radar_chart_context only if useful.\n\n"
        "Language:\n"
        "- If the user asks in Chinese, answer in Chinese.\n"
        "- If the user asks in English, answer in English.\n"
    )

    return prompt


# AI 生图需要的数据和规则

def prepare_chart_data_for_image(df, mode, selected_indicator):
    if mode == "image_bar":
        return {
            "chart_type": "bar chart",
            "data_meaning": "Average selected environmental impact by diet group",
            "selected_indicator": selected_indicator,
            "basic_rule": "Diet Group is the category. selected_impact_value is the bar length.",
            "data": compute_bar_data(df, selected_indicator)
        }

    if mode == "image_treemap":
        all_rows = compute_treemap_data(df)
        # 只取 normalized impact 最高的 30 条，避免图片 prompt 太长
        def treemap_sort_key(x):
            val = x["normalized_impact"]
            if val is not None:
                return val
            return -1

        sorted_rows = sorted(all_rows, key=treemap_sort_key, reverse=True)
        return {
            "chart_type": "treemap",
            "data_meaning": "Environmental impact by diet group, age group, and impact category",
            "hierarchy": "Diet Group -> Age Group -> Impact Category",
            "basic_rule": "Rectangle size and colour both represent normalized_impact.",
            "data": sorted_rows[:30]
        }

    if mode == "image_radar":
        return {
            "chart_type": "radar chart",
            "data_meaning": "Relative environmental impact of each diet group across nine indicators",
            "axes": ALL_INDICATORS,
            "basic_rule": "Each axis is one environmental indicator. relative_impact is the radius. Each diet group is one polygon.",
            "data": compute_radar_data(df)
        }

    if mode == "image_scatter":
        return {
            "chart_type": "scatterplot matrix",
            "data_meaning": "Relationships between environmental impact indicators",
            "variables": ALL_INDICATORS,
            "basic_rule": "Each small panel compares two indicators. Each point is Diet Group + Age Group + Sex.",
            "data": compute_scatter_matrix_data(df)[:72]
        }

    if mode == "image_summary":
        treemap_all = compute_treemap_data(df)

        def summary_sort_key(x):
            val = x["normalized_impact"]
            if val is not None:
                return val
            return -1

        treemap_sorted = sorted(treemap_all, key=summary_sort_key, reverse=True)

        return {
            "chart_type": "visual summary",
            "data_meaning": "Summary of main dashboard patterns",
            "basic_rule": "Create a simple dashboard-style summary using the provided data samples.",
            "bar_chart_data": compute_bar_data(df, selected_indicator),
            "radar_chart_data": compute_radar_data(df),
            "treemap_data_sample": treemap_sorted[:15]
        }

    return {}


def build_chart_prompt_with_llm(mode, chart_data):
    system_prompt = (
        "You are helping generate chart images for a FoodVis dashboard.\n"
        "Create a simple image-generation prompt based on the provided chart data.\n"
        "Return JSON only.\n\n"
        "JSON format:\n"
        "{\n"
        '  "title": "chart title",\n'
        '  "caption": "short caption",\n'
        '  "image_prompt": "prompt for image generation"\n'
        "}"
    )

    user_prompt = (
        "Dataset memory:\n"
        + AI_DATA_MEMORY
        + "\n\nRequest mode:\n"
        + mode
        + "\n\nChart data:\n"
        + json.dumps(chart_data, ensure_ascii=False, indent=2)
        + "\n\nCreate a simple image-generation prompt.\n\n"
        "Basic chart rules:\n"
        "- Bar chart: use diet groups as categories and selected_impact_value as bar length.\n"
        "- Treemap: use hierarchy Diet Group -> Age Group -> Impact Category. Use normalized_impact for rectangle size and colour.\n"
        "- Radar chart: use nine environmental indicators as axes. Use relative_impact as radius. Use one polygon per diet group.\n"
        "- Scatterplot matrix: use nine environmental indicators as variables. Each point represents Diet Group + Age Group + Sex.\n"
        "- Visual summary: use the provided data samples to create a simple dashboard-style summary.\n\n"
        "Style: Clean dashboard style. White or light background. Readable labels. "
        "Similar to a Tableau dashboard panel. Do not invent data values."
    )

    response = client.responses.create(
        model=CHAT_MODEL,
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}
        ]
    )
    return extract_json_from_text(response.output_text)


def generate_chart_image(image_prompt):
    result = client.images.generate(model=IMAGE_MODEL, prompt=image_prompt, size="1024x1024")
    image_item = result.data[0]
    if hasattr(image_item, "b64_json") and image_item.b64_json:
        return image_item.b64_json
    raise RuntimeError("No base64 image was returned by the image model.")


# Flask 路由

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/app.js")
def serve_js():
    return send_from_directory("static", "app.js")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "chat_model": CHAT_MODEL,
        "image_model": IMAGE_MODEL,
        "memory_created": bool(AI_DATA_MEMORY)
    })


@app.route("/api/debug-data", methods=["GET"])
def debug_data():
    try:
        df = get_cached_df()
        return jsonify({
            "rows": int(len(df)),
            "columns": list(df.columns),
            "sample": df.head(3).fillna("").to_dict(orient="records"),
            "export_files": {
                "treemap_data_exists": TREEMAP_DATA_PATH.exists(),
                "radar_data_exists": RADAR_DATA_PATH.exists(),
                "scatter_matrix_data_exists": SCATTER_DATA_PATH.exists()
            },
            "bar_ghg_data": compute_bar_data(df, "GHG Emissions"),
            "treemap_sample": compute_treemap_data(df)[:5],
            "radar_sample": compute_radar_data(df)[:5],
            "scatter_sample": compute_scatter_matrix_data(df)[:5]
        })
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/api/debug-memory", methods=["GET"])
def debug_memory():
    return jsonify({
        "memory_created": bool(AI_DATA_MEMORY),
        "ai_data_memory": AI_DATA_MEMORY
    })


@app.route("/api/ask-ai", methods=["POST"])
def ask_ai():
    try:
        if not os.getenv("OPENAI_API_KEY"):
            return jsonify({"error": "OPENAI_API_KEY is missing. Please check your .env file."}), 500

        body = request.get_json(force=True) or {}
        question = body.get("question", "")
        screenshot = body.get("screenshot", "")
        selected_indicator = body.get("environmentalIndicator", "GHG Emissions")

        has_question = bool(str(question).strip())
        has_screenshot = bool(str(screenshot).strip())

        if not has_question and not has_screenshot:
            return jsonify({"error": "Please enter a question or paste a screenshot."}), 400

        df = get_cached_df()

        if has_question:
            mode = detect_request_mode(question)
        else:
            mode = "text"

        # 普通文字问答，也可以同时带一张 dashboard 截图
        if mode == "text":
            dashboard_context = build_dashboard_context(df, selected_indicator)

            if has_question:
                user_question = question
            else:
                user_question = (
                    "No text question was provided. "
                    "Please analyse the pasted dashboard screenshot and explain the main visible patterns. "
                    "Use the dataset memory and dashboard context as support."
                )

            user_prompt = build_text_answer_prompt(
                question=user_question,
                dashboard_context=dashboard_context
            )

            user_content = [{"type": "input_text", "text": user_prompt}]

            if has_screenshot and str(screenshot).startswith("data:image"):
                user_content.append({
                    "type": "input_image",
                    "image_url": screenshot,
                    "detail": "low"
                })

            developer_text = (
                "You are a clear and accurate dashboard explanation assistant. "
                "Use the dataset memory, dashboard context, and screenshot if provided. "
                "Do not invent numbers."
            )

            response = client.responses.create(
                model=CHAT_MODEL,
                input=[
                    {"role": "developer", "content": [{"type": "input_text", "text": developer_text}]},
                    {"role": "user", "content": user_content}
                ]
            )
            return jsonify({
                "mode": "text",
                "answer": response.output_text,
                "image_base64": ""
            })

        # 用户明确要求生成图时才进入这里
        chart_data = prepare_chart_data_for_image(df, mode, selected_indicator)
        prompt_info = build_chart_prompt_with_llm(mode, chart_data)
        image_prompt = prompt_info.get("image_prompt", "")

        if not image_prompt:
            return jsonify({"error": "Image prompt was not generated."}), 500

        image_base64 = generate_chart_image(image_prompt)
        return jsonify({
            "mode": "image",
            "chartType": mode,
            "title": prompt_info.get("title", "AI Generated Chart"),
            "answer": prompt_info.get("caption", "AI-generated visualisation."),
            "image_base64": image_base64
        })

    except Exception as error:
        return jsonify({"error": "AI request failed.", "detail": str(error)}), 500


# 启动 Flask

if __name__ == "__main__":
    initialise_ai_data_memory()
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)

