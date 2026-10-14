"""
探索性分析的統計：清點影像的尺寸與色彩模式，不畫圖。

刻意不 import matplotlib——繪圖留在 engine/visualize.py，data/ 這邊只吐純資料
結構，才不會為了算幾個數字就把 data/ 綁上 engine/。
"""

import statistics
from collections import Counter
from pathlib import Path

from PIL import Image


def survey_images(all_images):
    """逐張讀出影像的尺寸與色彩模式。

    吃 collect_all_images() 的回傳值（{class_name: [Path, ...]}），形狀直接相容。

    Image.open 是惰性的：只解析檔頭就能拿到 .size 與 .mode，不會 decode 像素，
    所以掃完 5856 張的成本跟 _verify_paths 的全量 stat 同一個量級，不需要抽樣。

    回傳 {class_name: [record, ...]}：
      外層  鍵沿用 all_images 的類別名，順序也一致。
      record  {"path": Path, "width": int, "height": int, "mode": str}
              width/height 是原圖的像素尺寸（未經任何 resize），mode 是 PIL 的
              色彩模式字串（這份資料集會同時出現 "L" 與 "RGB"）。
              順序與 all_images[class_name] 一一對應。
    """
    survey = {}
    for class_name, files in all_images.items():
        records = []
        for file in files:
            # 惰性開檔只讀檔頭
            with Image.open(file) as img:
                width, height = img.size
                mode = img.mode
            records.append(
                {"path": Path(file), "width": width, "height": height, "mode": mode}
            )
        survey[class_name] = records
    return survey


def _describe(values):
    """一組數值的 min / max / median / mean。

    回傳 {"min": int|float, "max": ..., "median": ..., "mean": float}；
    空清單回傳四個 0，讓呼叫端不用到處檢查 None。
    """
    if not values:
        return {"min": 0, "max": 0, "median": 0, "mean": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "mean": float(statistics.fmean(values)),
    }


def size_summary(survey):
    """每個類別的尺寸統計。

    unique_sizes 是重點：它若等於 1，表示全資料集同一個尺寸，「混雜狀況」就不
    存在、resize 也就不必解釋；實際上這份資料集每張都不一樣。

    回傳 {class_name: {...}}：
      count         該類別張數。
      unique_sizes  相異 (width, height) 組合的個數。
      width/height  _describe() 的四個數，單位是像素。
      aspect        width / height 的 _describe()，1.0 代表正方形。
    數值一律是 python 原生 int / float（不是 np 純量），可直接 json.dump。
    """
    summary = {}
    for class_name, records in survey.items():
        widths = [r["width"] for r in records]
        heights = [r["height"] for r in records]
        aspects = [r["width"] / r["height"] for r in records]
        summary[class_name] = {
            "count": len(records),
            "unique_sizes": len({(r["width"], r["height"]) for r in records}),
            "width": _describe(widths),
            "height": _describe(heights),
            "aspect": _describe(aspects),
        }
    return summary


def mode_counts(survey):
    """每個類別的色彩模式分佈。

    回傳的形狀刻意對齊 plot_class_distribution 吃的 nested dict
    （{外層: {系列: 數量}}），所以色彩模式長條圖直接重用那個函式就好，
    不必再寫一個幾乎一樣的繪圖函式。

    回傳 {class_name: {mode: count}}：
      mode   PIL 的色彩模式字串，例如 "L"（灰階）或 "RGB"。
             只列出實際出現過的模式——沒出現的模式不會補 0，因為模式的集合
             事先不知道（不像 CLASS_NAMES 是固定的）。
      count  張數，python int。
    """
    return {
        class_name: dict(Counter(r["mode"] for r in records))
        for class_name, records in survey.items()
    }
