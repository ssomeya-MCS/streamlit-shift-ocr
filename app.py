from __future__ import annotations

import datetime as dt
import hashlib
import io
import re
import statistics
import unicodedata
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
from paddleocr import PaddleOCR


APP_TITLE = "手書き勤務シフト表 OCR"
MAX_PDF_PAGES = 5
LOW_CONFIDENCE = 0.85
SCHEDULE_COLUMNS = ["氏名"] + [f"{day}日" for day in range(1, 32)]
REVIEW_COLUMNS = [
    "ページ", "OCR行", "分類", "推定氏名", "推定日", "認識テキスト",
    "信頼度", "要確認", "X中心(%)", "Y中心(%)", "左端(px)", "右端(px)",
]
META_COLUMNS = ["項目", "値"]
DEFAULT_SHIFT_TOKENS = "D1,D2,休,M,A,B,C,夜,明,早,遅,日,有休,有,公,欠,研,代,半休,○,〇,×,-"
HEADER_NAMES = {"氏名", "職員名", "従業員名", "名前", "職員", "従業員", "スタッフ", "name", "employee"}


@st.cache_resource(show_spinner=False)
def get_ocr_engine() -> PaddleOCR:
    """PaddleOCR 2.10系の日本語モデル。モデルは初回にダウンロードされる。"""
    return PaddleOCR(use_angle_cls=True, lang="japan", use_gpu=False, show_log=False)


def load_images(file_bytes: bytes, filename: str, dpi: int) -> tuple[list[Image.Image], int, int]:
    """PDF/画像をRGB画像へ変換。PDFは最大MAX_PDF_PAGESページまで処理。"""
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix == "pdf":
        info = pdfinfo_from_bytes(file_bytes)
        total_pages = int(info.get("Pages", 0))
        if total_pages < 1:
            raise ValueError("PDFからページ数を取得できません。PDFが破損していないか確認してください。")
        count = min(total_pages, MAX_PDF_PAGES)
        pages = convert_from_bytes(
            file_bytes, dpi=dpi, first_page=1, last_page=count,
            fmt="png", thread_count=1,
        )
        return [im.convert("RGB") for im in pages], total_pages, len(pages)

    try:
        with Image.open(io.BytesIO(file_bytes)) as im:
            return [ImageOps.exif_transpose(im).convert("RGB")], 1, 1
    except Exception as exc:
        raise ValueError(f"画像を読み込めませんでした: {exc}") from exc


def extract_detections(result: Any, page_no: int, width: int, height: int) -> list[dict[str, Any]]:
    """PaddleOCR 2.xの戻り値からテキスト、信頼度、バウンディングボックスを抽出。"""
    detections: list[dict[str, Any]] = []
    if not result or not isinstance(result, list):
        return detections
    page_items = result[0] if result else None
    if not page_items:
        return detections

    for item in page_items:
        if not item or len(item) < 2:
            continue
        box, recognition = item[0], item[1]
        if not recognition or len(recognition) < 2:
            continue
        text = str(recognition[0]).strip()
        if not text:
            continue
        try:
            score = float(recognition[1])
            points = np.asarray(box, dtype=float).reshape(-1, 2)
        except (TypeError, ValueError):
            continue
        if points.size == 0:
            continue
        x_min, y_min = float(points[:, 0].min()), float(points[:, 1].min())
        x_max, y_max = float(points[:, 0].max()), float(points[:, 1].max())
        detections.append({
            "page": page_no,
            "text": text,
            "score": score,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "center_x": (x_min + x_max) / 2,
            "center_y": (y_min + y_max) / 2,
            "width": max(1.0, x_max - x_min),
            "height": max(1.0, y_max - y_min),
            "image_width": width,
            "image_height": height,
        })
    return detections


def normalized_text(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).strip()


def cluster_page_rows(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Y座標の近いOCR領域を暫定的な1行にまとめる。"""
    if not items:
        return []
    ordered = sorted(items, key=lambda d: (d["center_y"], d["center_x"]))
    typical_height = statistics.median(d["height"] for d in ordered)
    tolerance = max(7.0, typical_height * 0.70)
    clusters: list[list[dict[str, Any]]] = []

    for item in ordered:
        best = None
        best_distance = None
        for cluster in clusters:
            cy = statistics.median(x["center_y"] for x in cluster)
            distance = abs(item["center_y"] - cy)
            if distance <= tolerance and (best_distance is None or distance < best_distance):
                best = cluster
                best_distance = distance
        if best is None:
            clusters.append([item])
        else:
            best.append(item)
    return sorted(clusters, key=lambda c: statistics.median(d["center_y"] for d in c))


def header_day_mentions(item: dict[str, Any]) -> list[tuple[int, float]]:
    """認識テキストから日付見出し候補を取得。

    「1日」のような単独ラベルと、OCRが「1 2 3 ...」を1領域にまとめたケースを扱う。
    英数字のコード（D1等）の末尾数字を日付と誤検出しないよう英数字境界を確認する。
    """
    text = normalized_text(item["text"])
    if not text:
        return []
    matches = list(re.finditer(r"(?<![A-Za-z0-9])(\d{1,2})\s*日?(?![A-Za-z0-9])", text))
    found: list[tuple[int, float]] = []
    for match in matches:
        day = int(match.group(1))
        if not 1 <= day <= 31:
            continue
        char_position = (match.start(1) + match.end(1)) / 2
        proportion = char_position / max(1, len(text))
        x = item["x_min"] + proportion * item["width"]
        found.append((day, x))
    return found


def infer_day_centers(
    clusters: list[list[dict[str, Any]]],
    image_width: int,
    manual_day1_pct: float,
    manual_day31_pct: float,
    auto_detect: bool,
) -> tuple[np.ndarray, int | None, str, int]:
    """日付見出し位置から31列のX中心を推定。検出できなければ手動指定を使用。"""
    header_index: int | None = None
    centers = None
    anchor_count = 0
    candidates: list[tuple[int, dict[int, list[float]]]] = []

    # 日付見出し行の特定は手動位置モードでも実行する。
    # これにより手動調整に切り替えた場合でも「1日～31日」の見出しを職員行と誤認しない。
    for index, cluster in enumerate(clusters):
        per_day: dict[int, list[float]] = {}
        for item in cluster:
            for day, x in header_day_mentions(item):
                per_day.setdefault(day, []).append(x)
        if len(per_day) >= 5:
            candidates.append((index, per_day))

    if candidates:
        candidates.sort(
            key=lambda pair: (
                int(min(pair[1]) <= 4 and max(pair[1]) >= 28),
                len(pair[1]),
            ),
            reverse=True,
        )
        candidate_index, per_day = candidates[0]
        header_index = candidate_index
        days = np.asarray(sorted(per_day), dtype=float)
        xs = np.asarray([statistics.median(per_day[int(d)]) for d in days], dtype=float)
        anchor_count = len(days)
        if auto_detect and len(days) >= 5 and float(days.max() - days.min()) >= 10:
            slope, intercept = np.polyfit(days, xs, 1)
            if slope > 0:
                estimated = intercept + slope * np.arange(1, 32, dtype=float)
                # 左右端から大きく外れる推定は採用しない。
                if estimated[0] >= -0.05 * image_width and estimated[-1] <= 1.05 * image_width:
                    centers = np.clip(estimated, 0, image_width - 1)

    if centers is not None:
        method = f"OCRで日付見出しを自動検出（{anchor_count}日分の見出し）"
    else:
        x1 = image_width * manual_day1_pct / 100.0
        x31 = image_width * manual_day31_pct / 100.0
        if x31 <= x1:
            x31 = min(image_width - 1, x1 + image_width * 0.75)
        centers = np.linspace(x1, x31, 31)
        method = "手動指定の位置から31列を等間隔に推定"
    return centers, header_index, method, anchor_count


def parse_shift_tokens(text: str, allowed_tokens: list[str]) -> list[tuple[str, float, float]]:
    """勤務記号を最長一致で分割し、OCR文字列中の相対位置も返す。

    戻り値: [(勤務記号, start_ratio, end_ratio), ...]
    未知語は一塊として残し、勝手に文字単位で分解しない。
    """
    raw = normalized_text(text)
    if not raw:
        return []
    tokens = sorted({normalized_text(t) for t in allowed_tokens if normalized_text(t)}, key=len, reverse=True)
    if not tokens:
        return [(raw, 0.0, 1.0)]

    spans: list[tuple[str, float, float]] = []
    # OCR領域に空白がある場合、空白ごとの塊をそれぞれ最長一致で解析する。
    pieces = list(re.finditer(r"\S+", raw))
    for piece in pieces:
        segment = piece.group(0)
        pos = 0
        segment_tokens: list[tuple[str, int, int]] = []
        fully_parsed = True
        while pos < len(segment):
            match_token = next((token for token in tokens if segment.startswith(token, pos)), None)
            if match_token is None:
                fully_parsed = False
                break
            segment_tokens.append((match_token, pos, pos + len(match_token)))
            pos += len(match_token)

        if fully_parsed and segment_tokens:
            for token, start, end in segment_tokens:
                absolute_start = piece.start() + start
                absolute_end = piece.start() + end
                spans.append((token, absolute_start / len(raw), absolute_end / len(raw)))
        else:
            # 未知の記号・文字列は1つの値として扱い、誤った文字分割を避ける。
            spans.append((segment, piece.start() / len(raw), piece.end() / len(raw)))

    return spans


def _is_header_or_metadata(text: str) -> bool:
    value = normalized_text(text).lower().replace(" ", "")
    if value in {x.lower() for x in HEADER_NAMES}:
        return True
    if value in {"対象年月", "年月", "施設名", "勤務表", "シフト表", "合計", "備考", "部署名"}:
        return True
    return False


def _looks_like_shift_token(text: str, allowed_tokens: list[str]) -> bool:
    normalized = normalized_text(text)
    if not normalized:
        return False
    return normalized in {normalized_text(t) for t in allowed_tokens}


def _get_name_candidate(left_items: list[dict[str, Any]], allowed_tokens: list[str]) -> str:
    candidates = []
    for item in sorted(left_items, key=lambda d: d["center_x"]):
        txt = normalized_text(item["text"])
        if not txt or _is_header_or_metadata(txt):
            continue
        if re.fullmatch(r"[0-9.\-]+", txt):
            continue
        if _looks_like_shift_token(txt, allowed_tokens):
            continue
        candidates.append(item)

    if not candidates:
        return ""
    # 左側に番号列がある場合は、日付列に近い最後の非数値テキストを氏名候補にする。
    # 氏名が複数の領域に分かれている場合は、隣接候補を連結する。
    chosen = [candidates[-1]]
    for item in reversed(candidates[:-1]):
        last = chosen[0]
        gap = last["x_min"] - item["x_max"]
        if gap <= max(last["height"], item["height"]) * 0.5:
            chosen.insert(0, item)
        else:
            break
    return "".join(normalized_text(item["text"]) for item in chosen).strip()


def build_schedule(
    detections: list[dict[str, Any]],
    manual_day1_pct: float,
    manual_day31_pct: float,
    auto_detect: bool,
    allowed_tokens: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str], list[dict[str, Any]]]:
    """OCR領域を職員行と1～31日列に変換する。位置推定が曖昧な箇所は確認表に残す。"""
    schedule_rows: list[dict[str, Any]] = []
    review_records: list[dict[str, Any]] = []
    page_layouts: dict[int, str] = {}
    line_summaries: list[dict[str, Any]] = []
    pages = sorted({int(d["page"]) for d in detections})

    for page_no in pages:
        page_items = [d for d in detections if int(d["page"]) == page_no]
        if not page_items:
            continue
        width = int(page_items[0]["image_width"])
        height = int(page_items[0]["image_height"])
        clusters = cluster_page_rows(page_items)
        day_centers, header_index, method, anchor_count = infer_day_centers(
            clusters, width, manual_day1_pct, manual_day31_pct, auto_detect
        )
        page_layouts[page_no] = method
        steps = np.diff(day_centers)
        cell_width = float(np.median(steps)) if len(steps) else width / 31
        name_boundary = float(day_centers[0] - cell_width * 0.5)
        page_row_number = 0

        for cluster_index, cluster in enumerate(clusters):
            cluster = sorted(cluster, key=lambda d: d["center_x"])
            y_center = statistics.median(d["center_y"] for d in cluster)
            y_pct = y_center / max(1, height) * 100
            is_header_row = header_index is not None and cluster_index == header_index

            if is_header_row:
                for item in cluster:
                    review_records.append(_review_record(item, 0, "日付見出し", "", "", ""))
                continue

            left_items = [d for d in cluster if d["center_x"] < name_boundary and d["x_max"] < day_centers[0] + cell_width * 0.05]
            # 日付列のセルが、列中心から離れている場合の許容幅。
            right_items = [d for d in cluster if d["center_x"] >= name_boundary]

            # 一つのOCR領域が氏名列から日付列までまたいでいる場合の補助処理。
            crossing_items = [d for d in cluster if d["x_min"] < name_boundary and d["x_max"] > day_centers[0]]
            name = _get_name_candidate(left_items, allowed_tokens)
            shift_items = right_items[:]
            special_tokens: list[tuple[dict[str, Any], list[tuple[str, float, float]]]] = []

            if not name and crossing_items:
                # OCRが行全体を一つの領域にまとめた場合は、まず空白で氏名と勤務欄を分離する。
                crossing = max(crossing_items, key=lambda d: d["width"])
                full_text = normalized_text(crossing["text"])
                if re.search(r"\s", crossing["text"]):
                    parts = list(re.finditer(r"\S+", normalized_text(crossing["text"])))
                    if len(parts) >= 2:
                        name = parts[0].group(0)
                        tail_start = parts[1].start()
                        tail = full_text[tail_start:]
                        parsed = parse_shift_tokens(tail, allowed_tokens)
                        # tailに対する相対位置を元領域の相対位置へ戻す。
                        adjusted = []
                        for token, start, end in parsed:
                            adjusted.append((token, (tail_start + start * len(tail)) / max(1, len(full_text)), (tail_start + end * len(tail)) / max(1, len(full_text)))
                            )
                        special_tokens.append((crossing, adjusted))
                        shift_items = [d for d in right_items if d is not crossing]
                else:
                    # 空白がなく、既知の勤務記号で末尾を分割できる場合のみ暫定分離する。
                    remainder = full_text
                    rev_tokens: list[tuple[str, int, int]] = []
                    sorted_tokens = sorted({normalized_text(t) for t in allowed_tokens if normalized_text(t)}, key=len, reverse=True)
                    while remainder:
                        match_token = next((t for t in sorted_tokens if remainder.endswith(t)), None)
                        if match_token is None:
                            break
                        rev_tokens.insert(0, (match_token, len(remainder) - len(match_token), len(remainder)))
                        remainder = remainder[:-len(match_token)]
                        if len(rev_tokens) >= 31:
                            break
                    if remainder and len(rev_tokens) >= 2:
                        name = remainder
                        special_tokens.append((crossing, [(t, s / len(full_text), e / len(full_text)) for t, s, e in rev_tokens]))
                        shift_items = [d for d in right_items if d is not crossing]

            page_row_number += 1
            is_metadata_row = all(_is_header_or_metadata(d["text"]) for d in cluster)
            if is_metadata_row:
                for item in cluster:
                    review_records.append(_review_record(item, page_row_number, "見出し/メタ情報", "", "", ""))
                continue

            cell_values: dict[int, tuple[str, float, bool]] = {}
            item_to_days: dict[int, list[int]] = {}
            item_to_class: dict[int, str] = {}

            def assign_detection(item: dict[str, Any], text_for_parse: str | None = None,
                                 position_start: float | None = None,
                                 position_end: float | None = None) -> None:
                txt = normalized_text(text_for_parse if text_for_parse is not None else item["text"])
                if not txt or _is_header_or_metadata(txt):
                    item_to_class[id(item)] = "見出し/メタ情報"
                    return
                parse_text = txt
                spans = parse_shift_tokens(parse_text, allowed_tokens)
                # 名前が一つのOCR領域に混在するケース以外では、unknown textも一つのセルへ保持。
                local_start = item["x_min"] if position_start is None else position_start
                local_end = item["x_max"] if position_end is None else position_end
                span_width = max(1.0, local_end - local_start)
                assigned_days: list[int] = []
                unresolved = False
                for token, start_ratio, end_ratio in spans:
                    token_x = local_start + ((start_ratio + end_ratio) / 2) * span_width
                    day_index = int(np.argmin(np.abs(day_centers - token_x))) + 1
                    distance = abs(float(day_centers[day_index - 1]) - token_x)
                    if distance > cell_width * 0.72:
                        unresolved = True
                        continue
                    current = cell_values.get(day_index)
                    value = token
                    needs_review = float(item["score"]) < LOW_CONFIDENCE or unresolved
                    if current is None or float(item["score"]) > current[1]:
                        cell_values[day_index] = (value, float(item["score"]), needs_review)
                    elif current[0] != value:
                        cell_values[day_index] = (current[0] + " / " + value, current[1], True)
                        needs_review = True
                    assigned_days.append(day_index)
                item_to_days[id(item)] = sorted(set(assigned_days))
                item_to_class[id(item)] = "勤務記号" if assigned_days else "未割当/要確認"

            for item in shift_items:
                if item in crossing_items and not name:
                    # 行全体を分けられなかった領域を日付列に無理に割り当てない。
                    item_to_class[id(item)] = "氏名・勤務欄の分離失敗"
                    item_to_days[id(item)] = []
                    continue
                assign_detection(item)

            for item, spans in special_tokens:
                # crossing領域の相対位置を用いて、末尾の勤務記号を日付位置に割り当てる。
                txt = normalized_text(item["text"])
                local = []
                for token, start_ratio, end_ratio in spans:
                    token_x = item["x_min"] + (start_ratio + end_ratio) / 2 * item["width"]
                    day_index = int(np.argmin(np.abs(day_centers - token_x))) + 1
                    if abs(float(day_centers[day_index - 1]) - token_x) <= cell_width * 0.85:
                        current = cell_values.get(day_index)
                        # 行全体を分割した場合は文字位置を比例推定しているため、必ず要確認にする。
                        needs_review = True
                        if current is None or float(item["score"]) > current[1]:
                            cell_values[day_index] = (token, float(item["score"]), needs_review)
                        elif current[0] != token:
                            cell_values[day_index] = (current[0] + " / " + token, current[1], True)
                        local.append(day_index)
                item_to_days[id(item)] = sorted(set(local))
                item_to_class[id(item)] = "行全体から分割（要確認）"

            nonempty_cells = len(cell_values)
            if name and nonempty_cells == 0:
                # 名前だけの行/見出しを勤務データとして出力しない。
                for item in cluster:
                    review_records.append(_review_record(item, page_row_number, "氏名候補/勤務欄なし", name, "", ""))
                continue
            if not name and nonempty_cells == 0:
                for item in cluster:
                    review_records.append(_review_record(item, page_row_number, item_to_class.get(id(item), "未割当"), "", "", ""))
                continue
            if not name:
                name = f"要確認（p{page_no}-r{page_row_number}）"

            # 同じページに同名が複数行あれば、区別できるようページ内行番号を付ける。
            name_values = [r["氏名"] for r in schedule_rows if r.get("ページ") == page_no]
            if name in name_values:
                name = f"{name}（行{page_row_number}）"

            output_row: dict[str, Any] = {"氏名": name, "ページ": page_no, "OCR行": page_row_number}
            for day in range(1, 32):
                output_row[f"{day}日"] = cell_values.get(day, ("", 0.0, False))[0]
            output_row["要確認セル数"] = sum(1 for _day, (_value, _score, needs) in cell_values.items() if needs)
            schedule_rows.append(output_row)

            for item in cluster:
                cls = item_to_class.get(id(item), "氏名候補" if item in left_items else "未割当")
                days_for_item = item_to_days.get(id(item), [])
                mapped_days = ",".join(str(d) for d in days_for_item)
                review_records.append(_review_record(item, page_row_number, cls, name, mapped_days, ""))

            line_summaries.append({
                "ページ": page_no,
                "OCR行": page_row_number,
                "推定氏名": name,
                "検出セル数": nonempty_cells,
                "要確認セル数": output_row["要確認セル数"],
                "Y位置(%)": round(y_pct, 2),
            })

    # Excel本体はご指定の形式（氏名＋1日～31日）のみにする。
    if schedule_rows:
        schedule_df = pd.DataFrame(schedule_rows)
        # ページと解析補助情報は本表から除外。重複氏名などの確認はOCR確認シート側に残す。
        schedule_df = schedule_df[SCHEDULE_COLUMNS].copy()
    else:
        schedule_df = pd.DataFrame(columns=SCHEDULE_COLUMNS)

    review_df = pd.DataFrame(review_records, columns=REVIEW_COLUMNS)
    return schedule_df, review_df, page_layouts, line_summaries


def _review_record(
    item: dict[str, Any], row_no: int, classification: str,
    employee_name: str, day_text: str, _unused: str,
) -> dict[str, Any]:
    width = max(1, int(item.get("image_width", 1)))
    height = max(1, int(item.get("image_height", 1)))
    score = float(item.get("score", 0.0))
    needs_review = score < LOW_CONFIDENCE or classification in {"未割当", "未割当/要確認", "氏名・勤務欄の分離失敗", "行全体から分割（要確認）"}
    return {
        "ページ": item.get("page", ""),
        "OCR行": row_no,
        "分類": classification,
        "推定氏名": employee_name,
        "推定日": day_text,
        "認識テキスト": item.get("text", ""),
        "信頼度": round(score, 4),
        "要確認": "要確認" if needs_review else "",
        "X中心(%)": round(float(item.get("center_x", 0)) / width * 100, 2),
        "Y中心(%)": round(float(item.get("center_y", 0)) / height * 100, 2),
        "左端(px)": round(float(item.get("x_min", 0)), 1),
        "右端(px)": round(float(item.get("x_max", 0)), 1),
    }


def make_excel(schedule_df: pd.DataFrame, review_df: pd.DataFrame, metadata: dict[str, Any]) -> bytes:
    """メインシートを氏名×1日～31日に整形し、OCR確認情報を別シートへ保存。"""
    output = io.BytesIO()
    meta_df = pd.DataFrame([{"項目": k, "値": v} for k, v in metadata.items()], columns=META_COLUMNS)

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        schedule_df[SCHEDULE_COLUMNS].to_excel(writer, sheet_name="勤務シフト表", index=False)
        review_df.to_excel(writer, sheet_name="OCR確認", index=False)
        meta_df.to_excel(writer, sheet_name="処理情報", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": "#1F4E78",
            "align": "center", "valign": "vcenter", "border": 1,
        })
        name_fmt = workbook.add_format({"border": 1, "valign": "vcenter"})
        cell_fmt = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter"})

        ws = writer.sheets["勤務シフト表"]
        ws.freeze_panes(1, 1)
        ws.autofilter(0, 0, max(0, len(schedule_df)), len(SCHEDULE_COLUMNS) - 1)
        ws.set_row(0, 24)
        ws.set_column(0, 0, 22, name_fmt)
        ws.set_column(1, 31, 6, cell_fmt)
        for col_index, column_name in enumerate(SCHEDULE_COLUMNS):
            ws.write(0, col_index, column_name, header_fmt)

        review_ws = writer.sheets["OCR確認"]
        review_ws.freeze_panes(1, 0)
        if len(REVIEW_COLUMNS):
            review_ws.autofilter(0, 0, max(0, len(review_df)), len(REVIEW_COLUMNS) - 1)
        review_ws.set_column(0, 4, 14)
        review_ws.set_column(5, 5, 34)
        review_ws.set_column(6, 11, 14)
        for col_index, column_name in enumerate(REVIEW_COLUMNS):
            review_ws.write(0, col_index, column_name, header_fmt)

        meta_ws = writer.sheets["処理情報"]
        meta_ws.set_column(0, 0, 24)
        meta_ws.set_column(1, 1, 72)
        for col_index, column_name in enumerate(META_COLUMNS):
            meta_ws.write(0, col_index, column_name, header_fmt)

    return output.getvalue()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title("📄 手書き勤務シフト表 OCR")
    st.write(
        "PDF・画像から文字を読み取り、Excelの1枚目を「氏名 × 1日～31日」の表に整形します。"
        "OCRが間違えたセルは画面上で修正してからダウンロードできます。"
    )
    st.warning(
        "OCRは勤務表の罫線や文字の書き方に影響されます。列位置が推定できない場合は手動設定を調整してください。"
        "給与・勤怠に使用する前に必ず原本と照合してください。"
    )
    st.info(
        "Streamlit Community Cloudではアップロードした書類がクラウド上で処理されます。"
        "実際の職員名・勤務情報を使う前に、会社の情報管理ルールを確認してください。"
    )

    col1, col2 = st.columns(2)
    with col1:
        target_ym = st.text_input("対象年月（処理情報シート用）", value=dt.datetime.now().strftime("%Y/%m"))
    with col2:
        facility_name = st.text_input("施設名（処理情報シート用・任意）", value="")

    uploaded = st.file_uploader(
        "勤務シフト表をアップロード（PDF / PNG / JPG / TIFF）",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
        help=f"PDFは先頭から最大{MAX_PDF_PAGES}ページまで処理します。",
    )
    dpi = st.slider("PDF画像化の解像度（DPI）", 150, 350, 250, 50)

    st.subheader("列位置の設定")
    auto_detect = st.checkbox("OCRで日付見出し（1日～31日）を自動検出する", value=True)
    c1, c2 = st.columns(2)
    with c1:
        manual_day1_pct = st.slider("手動設定：1日列の中心（画像の左から%）", 1, 90, 18)
    with c2:
        manual_day31_pct = st.slider("手動設定：31日列の中心（画像の左から%）", 10, 100, 98)
    st.caption("日付見出しを自動検出できなかった場合に、上記の1日列・31日列を使って列位置を等間隔に推定します。")

    token_text = st.text_input(
        "勤務記号（長い記号を先に認識します。カンマ区切りで追加・変更できます）",
        value=DEFAULT_SHIFT_TOKENS,
    )
    allowed_tokens = [token.strip() for token in token_text.split(",") if token.strip()]

    file_bytes = uploaded.getvalue() if uploaded is not None else b""
    signature = hashlib.sha256(file_bytes).hexdigest() if file_bytes else ""
    dpi_signature = f"{signature}|{dpi}"
    if "ocr_result_payload" not in st.session_state:
        st.session_state["ocr_result_payload"] = None
    old_payload = st.session_state.get("ocr_result_payload")
    if old_payload and old_payload.get("signature") != dpi_signature:
        st.session_state["ocr_result_payload"] = None

    run_ocr = st.button("OCRを実行してシフト表を作成", type="primary", disabled=uploaded is None)
    if run_ocr and uploaded is not None:
        try:
            with st.spinner("PDF/画像を読み込んでいます..."):
                images, total_pages, processed_pages = load_images(file_bytes, uploaded.name, dpi)
            ocr = get_ocr_engine()
            detections: list[dict[str, Any]] = []
            progress = st.progress(0, text="OCRを準備しています...")
            with st.spinner("PaddleOCRで文字認識中です。初回はモデルのダウンロードに時間がかかります..."):
                for page_no, image in enumerate(images, start=1):
                    result = ocr.ocr(np.asarray(image.convert("RGB")), cls=True)
                    detections.extend(extract_detections(result, page_no, image.width, image.height))
                    progress.progress(page_no / max(1, len(images)), text=f"{page_no}/{len(images)}ページ処理済み")
            progress.empty()
            st.session_state["ocr_result_payload"] = {
                "signature": dpi_signature,
                "detections": detections,
                "preview_image": images[0].copy() if images else None,
                "filename": uploaded.name,
                "total_pages": total_pages,
                "processed_pages": processed_pages,
            }
        except Exception as exc:
            st.error("OCR処理に失敗しました。Streamlit CloudのLogsで完全なエラー内容を確認してください。")
            st.exception(exc)
            st.session_state["ocr_result_payload"] = None

    payload = st.session_state.get("ocr_result_payload")
    if payload and payload.get("signature") == dpi_signature:
        if payload.get("total_pages", 1) > payload.get("processed_pages", 1):
            st.warning(
                f"全{payload['total_pages']}ページ中、先頭{payload['processed_pages']}ページのみ処理しました。"
            )
        with st.expander("原本プレビュー（先頭ページ）", expanded=False):
            if payload.get("preview_image") is not None:
                st.image(payload["preview_image"], use_container_width=True)

        schedule_df, review_df, page_layouts, line_summaries = build_schedule(
            payload["detections"],
            manual_day1_pct=float(manual_day1_pct),
            manual_day31_pct=float(manual_day31_pct),
            auto_detect=auto_detect,
            allowed_tokens=allowed_tokens,
        )
        st.subheader("日付列の推定結果")
        for page_no, method in page_layouts.items():
            st.write(f"ページ {page_no}: {method}")
        if auto_detect and any("手動指定" in method for method in page_layouts.values()):
            st.warning("日付見出しを十分に検出できなかったため、手動設定の1日列・31日列から列位置を推定しました。必要なら上の%を調整し、再確認してください。")

        if schedule_df.empty:
            st.error("勤務行を抽出できませんでした。原本プレビューとOCR確認をチェックし、列位置を調整してから再実行してください。")
        else:
            st.subheader("勤務シフト表（編集可能）")
            st.caption("列は氏名と1日～31日です。誤認識や空欄は、ダウンロード前に表内で修正できます。")
            token_signature = hashlib.sha1(token_text.encode("utf-8")).hexdigest()[:6]
            editor_key = (
                f"shift_editor_{signature[:12]}_{int(manual_day1_pct)}_"
                f"{int(manual_day31_pct)}_{int(auto_detect)}_{token_signature}"
            )
            edited_schedule = st.data_editor(
                schedule_df,
                key=editor_key,
                num_rows="dynamic",
                hide_index=True,
                use_container_width=True,
                column_config={
                    "氏名": st.column_config.TextColumn("氏名", required=True, width="medium"),
                    **{f"{d}日": st.column_config.TextColumn(f"{d}日", width="small") for d in range(1, 32)},
                },
            )

            low_count = 0
            if not review_df.empty:
                low_count = int((review_df["要確認"] == "要確認").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("抽出した職員行", len(edited_schedule))
            m2.metric("OCR文字領域", len(payload["detections"]))
            m3.metric("確認推奨のOCR領域", low_count)

            if not review_df.empty:
                with st.expander("OCR確認情報（信頼度・座標・推定分類）", expanded=False):
                    st.dataframe(review_df, use_container_width=True, hide_index=True)

            metadata = {
                "対象年月": target_ym,
                "施設名": facility_name.strip() or "（未入力）",
                "元ファイル名": payload["filename"],
                "処理日時": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "総ページ数": payload["total_pages"],
                "処理ページ数": payload["processed_pages"],
                "OCR方式": "PaddleOCR 2.10 / lang=japan",
                "備考": "勤務シフト表シートは氏名＋1日～31日。OCR確認シートの結果は原本照合が必要です。",
            }
            excel_bytes = make_excel(edited_schedule[SCHEDULE_COLUMNS], review_df, metadata)
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "📥 氏名×1日～31日のExcelをダウンロード",
                data=excel_bytes,
                file_name=f"勤務シフト表_OCR_{stamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )

        if not review_df.empty:
            with st.expander("OCRが検出したテキスト一覧", expanded=False):
                st.dataframe(review_df[["ページ", "OCR行", "分類", "推定氏名", "推定日", "認識テキスト", "信頼度", "要確認"]], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
