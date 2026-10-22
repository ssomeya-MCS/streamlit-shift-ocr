from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import math
import numbers
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
from paddleocr import PaddleOCR

APP_TITLE = "手書き勤務シフト表 OCR（固定レイアウト）"
MAX_PDF_PAGES = 5
LOW_CONFIDENCE = 0.85
SCHEDULE_COLUMNS = ["氏名"] + [f"{day}日" for day in range(1, 32)]
REVIEW_COLUMNS = ["ページ", "行番号", "項目", "日付", "OCR結果", "信頼度", "判定", "セル位置"]
META_COLUMNS = ["項目", "値"]
DEFAULT_SHIFT_TOKENS = "D1,D2,休,M,A,B,C,夜,明,早,遅,日,有休,有,公,欠,研,代,半休,○,〇,×,-"
HEADER_NAMES = {"氏名", "職員名", "従業員名", "名前", "職員", "従業員", "スタッフ", "name", "employee"}

# UI widgets use these keys so template settings can be loaded from/saved to JSON.
CONFIG_DEFAULTS = {
    "day1_center_pct": 18.0,
    "day31_center_pct": 98.0,
    "name_left_pct": 1.0,
    "name_right_pct": 15.0,
    "first_row_center_pct": 24.0,
    "row_step_pct": 2.5,
    "row_count": 25,
    "row_crop_ratio": 0.72,
    "upscale": 3,
    "enhance_contrast": True,
    "binarize": False,
    "dpi": 300,
}
CONFIG_LABELS = {
    "day1_center_pct": "1日列の中心X位置（画像の左から%）",
    "day31_center_pct": "31日列の中心X位置（画像の左から%）",
    "name_left_pct": "氏名欄の左端X位置（%）",
    "name_right_pct": "氏名欄の右端X位置（%）",
    "first_row_center_pct": "最初の職員行の中心Y位置（上から%）",
    "row_step_pct": "職員行の間隔（画像高さに対する%）",
    "row_count": "読み取る職員行数",
    "row_crop_ratio": "行セルの縦方向の切り出し幅（行間隔に対する比率）",
    "upscale": "文字画像の拡大倍率",
    "enhance_contrast": "文字のコントラストを補正",
    "binarize": "白黒2値化する（文字が薄くなる場合はオフ）",
    "dpi": "PDFの画像化解像度（DPI）",
}


@st.cache_resource(show_spinner=False)
def get_ocr_engine() -> PaddleOCR:
    """CPU版PaddleOCR 2.10の日本語モデルを初期化。"""
    return PaddleOCR(use_angle_cls=False, lang="japan", use_gpu=False, show_log=False)


def load_images(file_bytes: bytes, filename: str, dpi: int) -> tuple[list[Image.Image], int, int]:
    """PDF/画像をRGB画像へ変換。PDFは最大MAX_PDF_PAGESページまで処理。"""
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix == "pdf":
        info = pdfinfo_from_bytes(file_bytes)
        total_pages = int(info.get("Pages", 0))
        if total_pages < 1:
            raise ValueError("PDFからページ数を取得できません。PDFが破損していないか確認してください。")
        count = min(total_pages, MAX_PDF_PAGES)
        pages = convert_from_bytes(file_bytes, dpi=dpi, first_page=1, last_page=count, fmt="png", thread_count=1)
        return [ImageOps.exif_transpose(im).convert("RGB") for im in pages], total_pages, len(pages)

    try:
        with Image.open(io.BytesIO(file_bytes)) as im:
            return [ImageOps.exif_transpose(im).convert("RGB")], 1, 1
    except Exception as exc:
        raise ValueError(f"画像を読み込めませんでした: {exc}") from exc


def get_config() -> dict[str, Any]:
    return {key: st.session_state.get(f"cfg_{key}", default) for key, default in CONFIG_DEFAULTS.items()}


def draw_template_overlay(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
    """Calibration preview: draw name/date columns and expected employee row bands."""
    preview = image.copy().convert("RGBA")
    width, height = preview.size
    overlay = Image.new("RGBA", preview.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    x1 = width * float(cfg["day1_center_pct"]) / 100.0
    x31 = width * float(cfg["day31_center_pct"]) / 100.0
    if x31 <= x1:
        return image.copy()
    centers = np.linspace(x1, x31, 31)
    step = float(centers[1] - centers[0])
    name_left = int(width * float(cfg["name_left_pct"]) / 100.0)
    name_right = int(width * float(cfg["name_right_pct"]) / 100.0)
    row_step = height * float(cfg["row_step_pct"]) / 100.0
    first_center = height * float(cfg["first_row_center_pct"]) / 100.0
    band_half = row_step * float(cfg["row_crop_ratio"]) / 2.0

    # Name field and per-day crop cells.
    draw.rectangle([name_left, 0, name_right, height - 1], outline=(0, 120, 255, 230), width=max(2, width // 1200))
    for i, center in enumerate(centers):
        left = int(center - step / 2)
        right = int(center + step / 2)
        draw.line([int(center), 0, int(center), height], fill=(255, 80, 40, 190), width=max(1, width // 1800))
        if i in (0, 30):
            draw.text((int(center) + 2, max(2, int(height * 0.015))), str(i + 1), fill=(255, 40, 20, 255), stroke_width=1, stroke_fill=(255, 255, 255, 230))
        # Subtle fill only for the first/last day to help tune endpoints.
        if i in (0, 30):
            draw.rectangle([left, 0, right, height - 1], fill=(255, 120, 0, 18))

    for row_idx in range(int(cfg["row_count"])):
        center_y = first_center + row_idx * row_step
        top = int(max(0, center_y - band_half))
        bottom = int(min(height - 1, center_y + band_half))
        if top >= height or center_y >= height:
            break
        color = (0, 160, 90, 135) if row_idx == 0 else (0, 160, 90, 70)
        draw.rectangle([name_left, top, width - 1, bottom], outline=color, width=max(1, width // 1800))
        draw.line([0, int(center_y), width - 1, int(center_y)], fill=(0, 160, 90, 110), width=max(1, width // 2200))
        if row_idx < 5:
            draw.text((2, top + 2), f"row {row_idx + 1}", fill=(0, 100, 30, 255), stroke_width=1, stroke_fill=(255, 255, 255, 220))

    return Image.alpha_composite(preview, overlay).convert("RGB")


def _cell_is_blank(gray_array: np.ndarray) -> bool:
    """Avoid running the recognizer on genuinely empty cells."""
    if gray_array.size == 0:
        return True
    # Treat only very sparse dark pixels as blank; faint writing should remain eligible.
    dark_ratio = float(np.mean(gray_array < 205))
    return dark_ratio < 0.0012


def prepare_crop(crop: Image.Image, upscale: int, enhance_contrast: bool, binarize: bool) -> tuple[np.ndarray | None, bool]:
    """Trim blank margins around handwriting, upscale, and add a white border.

    Returns (RGB ndarray or None if blank, is_blank).
    """
    gray = crop.convert("L")
    if enhance_contrast:
        gray = ImageOps.autocontrast(gray, cutoff=1)
    arr = np.asarray(gray)
    if _cell_is_blank(arr):
        return None, True

    # Tight crop around dark strokes so a one-letter symbol is not shrunk by large cell margins.
    ink = arr < 225
    ys, xs = np.where(ink)
    if len(xs) < 3:
        return None, True

    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    char_w = max(1, x_max - x_min)
    char_h = max(1, y_max - y_min)
    pad_x = max(3, int(char_w * 0.18))
    pad_y = max(3, int(char_h * 0.18))
    x_min = max(0, x_min - pad_x)
    x_max = min(arr.shape[1], x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(arr.shape[0], y_max + pad_y)
    gray = gray.crop((x_min, y_min, x_max, y_max))

    if binarize:
        a = np.asarray(gray)
        # Otsu-like fixed threshold after autocontrast; preserve black text on white.
        gray = Image.fromarray(np.where(a < 190, 0, 255).astype(np.uint8), mode="L")

    scale = int(upscale)
    gray = gray.resize((max(24, gray.width * scale), max(24, gray.height * scale)), Image.Resampling.LANCZOS)
    pad = max(8, 8 * scale)
    canvas = Image.new("L", (gray.width + pad * 2, gray.height + pad * 2), color=255)
    canvas.paste(gray, (pad, pad))
    rgb = canvas.convert("RGB")
    return np.asarray(rgb), False


def _find_text_score(obj: Any) -> tuple[str, float] | None:
    """Find a PaddleOCR recognition pair in its version-specific nested return value."""
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        if isinstance(obj[0], str) and isinstance(obj[1], (numbers.Real, np.number)):
            try:
                return obj[0], float(obj[1])
            except (TypeError, ValueError):
                pass
    if isinstance(obj, (list, tuple)):
        for child in obj:
            found = _find_text_score(child)
            if found is not None:
                return found
    return None


def recognize_prepared_crops(ocr: PaddleOCR, crop_arrays: list[np.ndarray], batch_size: int = 64) -> list[tuple[str, float]]:
    """Use recognition-only mode on a batch of already-cropped cells.

    Recognition-only is important here: the cell's known location replaces full-page
    text detection, which often misses one-character shift codes.
    """
    if not crop_arrays:
        return []
    outputs: list[tuple[str, float]] = []

    # PaddleOCR 2.10 exposes the recognizer. Batch calls are much faster than launching
    # the full detector separately for every one of the 31 cells in each employee row.
    recognizer = getattr(ocr, "text_recognizer", None)
    if callable(recognizer):
        try:
            for start in range(0, len(crop_arrays), batch_size):
                batch = crop_arrays[start:start + batch_size]
                raw = recognizer(batch)
                if not isinstance(raw, (list, tuple)) or len(raw) != len(batch):
                    raise RuntimeError("PaddleOCR text recognizer returned an unexpected result shape.")
                for item in raw:
                    pair = _find_text_score(item)
                    outputs.append(pair if pair is not None else ("", 0.0))
            return outputs
        except Exception:
            # Fall through to the public OCR method for compatibility with builds
            # whose recognizer wrapper differs from the pinned 2.10 environment.
            outputs = []

    for crop_array in crop_arrays:
        try:
            raw = ocr.ocr(crop_array, det=False, rec=True, cls=False)
            pair = _find_text_score(raw)
            outputs.append(pair if pair is not None else ("", 0.0))
        except Exception:
            outputs.append(("", 0.0))
    return outputs


def normalize_ocr_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"\s+", "", value).strip()


def is_metadata_name(name: str) -> bool:
    normalized = normalize_ocr_text(name).lower()
    return normalized in {normalize_ocr_text(x).lower() for x in HEADER_NAMES} or normalized in {
        "部署", "所属", "氏名", "職員名", "従業員名", "勤務表", "シフト表", "備考", "合計", "対象年月", "施設名", "番号"
    }


def cell_box_for_day(width: int, day_index: int, x1_pct: float, x31_pct: float, inset_ratio: float = 0.12) -> tuple[int, int]:
    centers = np.linspace(width * x1_pct / 100, width * x31_pct / 100, 31)
    step = float(centers[1] - centers[0])
    center = float(centers[day_index - 1])
    inset = step * float(inset_ratio)
    left = int(max(0, center - step / 2 + inset))
    right = int(min(width, center + step / 2 - inset))
    if right <= left:
        right = min(width, left + 1)
    return left, right


def crop_schedule_cells(image: Image.Image, cfg: dict[str, Any], page_no: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cut every employee-name and daily cell using the calibrated fixed template."""
    width, height = image.size
    x1 = float(cfg["day1_center_pct"])
    x31 = float(cfg["day31_center_pct"])
    if x31 <= x1:
        raise ValueError("1日列の中心は31日列の中心より左に指定してください。")
    if float(cfg["name_right_pct"]) <= float(cfg["name_left_pct"]):
        raise ValueError("氏名欄の右端は左端より右に指定してください。")

    row_step_px = height * float(cfg["row_step_pct"]) / 100
    if row_step_px <= 1:
        raise ValueError("職員行の間隔が小さすぎます。テンプレート設定を確認してください。")
    band_h = max(4, row_step_px * float(cfg["row_crop_ratio"]))
    first_y = height * float(cfg["first_row_center_pct"]) / 100
    crop_items: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []

    for row_idx in range(int(cfg["row_count"])):
        center_y = first_y + row_idx * row_step_px
        y_top = int(max(0, center_y - band_h / 2))
        y_bottom = int(min(height, center_y + band_h / 2))
        if center_y >= height or y_bottom <= y_top + 1:
            break

        # Name column crop. The user adjusts the left/right bounds once for the template.
        name_left = int(width * float(cfg["name_left_pct"]) / 100)
        name_right = int(width * float(cfg["name_right_pct"]) / 100)
        name_crop = image.crop((max(0, name_left), y_top, min(width, name_right), y_bottom))
        name_arr, name_blank = prepare_crop(name_crop, cfg["upscale"], cfg["enhance_contrast"], cfg["binarize"])
        cells.append({
            "page": page_no, "row": row_idx + 1, "kind": "name", "day": None,
            "box": (name_left, y_top, name_right, y_bottom), "blank": name_blank,
            "array": name_arr,
        })

        for day in range(1, 32):
            day_left, day_right = cell_box_for_day(width, day, x1, x31)
            day_crop = image.crop((day_left, y_top, day_right, y_bottom))
            arr, blank = prepare_crop(day_crop, cfg["upscale"], cfg["enhance_contrast"], cfg["binarize"])
            cells.append({
                "page": page_no, "row": row_idx + 1, "kind": "shift", "day": day,
                "box": (day_left, y_top, day_right, y_bottom), "blank": blank,
                "array": arr,
            })

    nonblank = [item for item in cells if not item["blank"] and item["array"] is not None]
    return cells, nonblank


def run_template_ocr(images: list[Image.Image], cfg: dict[str, Any], allowed_tokens: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read each known cell, build the schedule table, and create a cell-level review log."""
    ocr = get_ocr_engine()
    all_cells: list[dict[str, Any]] = []
    work_items: list[dict[str, Any]] = []
    progress_total = len(images) * int(cfg["row_count"])

    progress = st.progress(0, text="テンプレートに沿ってセルを切り出しています...")
    for page_no, image in enumerate(images, start=1):
        cells, nonblank = crop_schedule_cells(image, cfg, page_no)
        all_cells.extend(cells)
        work_items.extend(nonblank)
        progress.progress(min(1.0, page_no / max(1, len(images))), text=f"{page_no}/{len(images)}ページのセル位置を計算しました")

    crop_arrays = [item["array"] for item in work_items]
    st.write(f"認識対象のセル数（空欄を除く）: **{len(crop_arrays)}**")
    if crop_arrays:
        with st.spinner("切り出したセルを認識しています。セルの数に応じて時間がかかります..."):
            recognized = recognize_prepared_crops(ocr, crop_arrays, batch_size=64)
        for item, (text, score) in zip(work_items, recognized):
            item["text"] = normalize_ocr_text(text)
            item["score"] = float(score)
            item["blank"] = not bool(item["text"])
    progress.empty()

    # Attach blank-cell defaults and prepare the staff-by-day table.
    for item in all_cells:
        item.setdefault("text", "")
        item.setdefault("score", 0.0)

    by_page_row: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for item in all_cells:
        by_page_row.setdefault((int(item["page"]), int(item["row"])), []).append(item)

    schedule_records: list[dict[str, Any]] = []
    review_records: list[dict[str, Any]] = []
    allowed = {normalize_ocr_text(x) for x in allowed_tokens if normalize_ocr_text(x)}

    for (page_no, row_no), row_cells in sorted(by_page_row.items()):
        name_cell = next(item for item in row_cells if item["kind"] == "name")
        name = normalize_ocr_text(name_cell.get("text", ""))
        shift_cells = [item for item in row_cells if item["kind"] == "shift"]
        shift_text_count = sum(bool(item.get("text")) for item in shift_cells)

        # Ignore rows that are fully blank. Keep rows with shift marks but no name for manual repair.
        if not name and shift_text_count == 0:
            continue
        if is_metadata_name(name):
            continue
        if not name:
            name = f"要確認（P{page_no}-行{row_no}）"
            name_cell["needs_review"] = True
        else:
            name_cell["needs_review"] = float(name_cell.get("score", 0.0)) < LOW_CONFIDENCE

        out: dict[str, Any] = {"氏名": name}
        for item in shift_cells:
            day = int(item["day"])
            value = normalize_ocr_text(item.get("text", ""))
            score = float(item.get("score", 0.0))
            item["needs_review"] = bool(value) and (score < LOW_CONFIDENCE or value not in allowed)
            out[f"{day}日"] = value

        schedule_records.append(out)

        # The review sheet lists recognized cells plus low-confidence/unknown values;
        # it does not fill the log with hundreds of ordinary empty cells.
        for item in [name_cell] + shift_cells:
            value = normalize_ocr_text(item.get("text", ""))
            if item["kind"] == "name":
                needs_review = bool(item.get("needs_review", False))
                field = "氏名"
                day_label = ""
                cell_position = f"X:{item['box'][0]}-{item['box'][2]} / Y:{item['box'][1]}-{item['box'][3]}"
                score_value: float | str = round(float(item.get("score", 0.0)), 4) if value else ""
                status = "要確認" if needs_review else ("認識" if value else "空欄")
            else:
                if not value:
                    continue
                needs_review = bool(item.get("needs_review", False))
                field = "勤務記号"
                day_label = f"{item['day']}日"
                cell_position = f"X:{item['box'][0]}-{item['box'][2]} / Y:{item['box'][1]}-{item['box'][3]}"
                score_value = round(float(item.get("score", 0.0)), 4)
                status = "要確認" if needs_review else "認識"
            review_records.append({
                "ページ": item["page"], "行番号": item["row"], "項目": field,
                "日付": day_label, "OCR結果": value, "信頼度": score_value,
                "判定": status, "セル位置": cell_position,
            })

    schedule_df = pd.DataFrame(schedule_records, columns=SCHEDULE_COLUMNS)
    review_df = pd.DataFrame(review_records, columns=REVIEW_COLUMNS)
    if not schedule_df.empty:
        # Keep input order: pages then row positions. Same names are allowed when repeated across pages.
        schedule_df = schedule_df[SCHEDULE_COLUMNS]
    return schedule_df, review_df


def make_excel(schedule_df: pd.DataFrame, review_df: pd.DataFrame, metadata: dict[str, Any]) -> bytes:
    output = io.BytesIO()
    meta_df = pd.DataFrame([{"項目": key, "値": value} for key, value in metadata.items()], columns=META_COLUMNS)
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        schedule_df[SCHEDULE_COLUMNS].to_excel(writer, sheet_name="勤務シフト表", index=False)
        review_df.to_excel(writer, sheet_name="OCR確認", index=False)
        meta_df.to_excel(writer, sheet_name="処理情報", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": "#1F4E78",
            "align": "center", "valign": "vcenter", "border": 1,
        })
        body_name_fmt = workbook.add_format({"border": 1, "valign": "vcenter"})
        body_cell_fmt = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter"})

        ws = writer.sheets["勤務シフト表"]
        ws.freeze_panes(1, 1)
        ws.autofilter(0, 0, max(0, len(schedule_df)), len(SCHEDULE_COLUMNS) - 1)
        ws.set_row(0, 24)
        ws.set_column(0, 0, 22, body_name_fmt)
        ws.set_column(1, 31, 6, body_cell_fmt)
        for col_idx, col_name in enumerate(SCHEDULE_COLUMNS):
            ws.write(0, col_idx, col_name, header_fmt)

        review_ws = writer.sheets["OCR確認"]
        review_ws.freeze_panes(1, 0)
        if len(REVIEW_COLUMNS):
            review_ws.autofilter(0, 0, max(0, len(review_df)), len(REVIEW_COLUMNS) - 1)
        review_ws.set_column(0, 3, 13)
        review_ws.set_column(4, 4, 28)
        review_ws.set_column(5, 6, 12)
        review_ws.set_column(7, 7, 32)
        for col_idx, col_name in enumerate(REVIEW_COLUMNS):
            review_ws.write(0, col_idx, col_name, header_fmt)

        meta_ws = writer.sheets["処理情報"]
        meta_ws.set_column(0, 0, 24)
        meta_ws.set_column(1, 1, 74)
        for col_idx, col_name in enumerate(META_COLUMNS):
            meta_ws.write(0, col_idx, col_name, header_fmt)

    return output.getvalue()


def config_json_bytes(cfg: dict[str, Any]) -> bytes:
    return json.dumps(cfg, ensure_ascii=False, indent=2).encode("utf-8")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title("📄 手書き勤務シフト表 OCR（固定レイアウト版）")
    st.write(
        "毎月ほぼ同じ位置に職員名欄と1日～31日の列があるシフト表向けです。"
        "テンプレートの座標から各セルを切り出し、文字検出ではなく文字認識中心で読み取ります。"
    )
    st.warning(
        "固定レイアウトを利用しても、手書きの似た文字・空欄・薄い筆記は誤認識することがあります。"
        "勤務データとして使用する前に原本と照合してください。"
    )
    st.info(
        "Streamlit Community Cloudではアップロードした書類がクラウド上で処理されます。"
        "実際の職員名や勤務データを使う際は、会社の情報管理ルールを確認してください。"
    )

    col1, col2 = st.columns(2)
    with col1:
        target_ym = st.text_input("対象年月（処理情報シート）", value=dt.datetime.now().strftime("%Y/%m"))
    with col2:
        facility_name = st.text_input("施設名（処理情報シート・任意）", value="")

    uploaded = st.file_uploader(
        "勤務シフト表（PDF / PNG / JPG / TIFF）",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
    )
    config_upload = st.file_uploader(
        "以前保存したテンプレート設定JSON（任意）",
        type=["json"],
        help="同じ形式のシフト表で座標設定を再利用できます。初回は未指定で構いません。",
        key="template_config_upload",
    )

    # Apply uploaded JSON once, before the related sliders are instantiated.
    if config_upload is not None:
        config_raw = config_upload.getvalue()
        config_hash = hashlib.sha256(config_raw).hexdigest()
        if st.session_state.get("_loaded_config_hash") != config_hash:
            try:
                loaded = json.loads(config_raw.decode("utf-8-sig"))
                for key, default in CONFIG_DEFAULTS.items():
                    if key in loaded:
                        value = loaded[key]
                        if isinstance(default, bool):
                            value = bool(value)
                        elif isinstance(default, int) and not isinstance(default, bool):
                            value = int(value)
                        elif isinstance(default, float):
                            value = float(value)
                        st.session_state[f"cfg_{key}"] = value
                st.session_state["_loaded_config_hash"] = config_hash
                st.success("テンプレート設定を読み込みました。プレビューの位置を確認してください。")
            except Exception as exc:
                st.error(f"テンプレート設定JSONを読み込めませんでした: {exc}")

    # Seed widget state with defaults. Values from an uploaded JSON take precedence.
    for key, default in CONFIG_DEFAULTS.items():
        st.session_state.setdefault(f"cfg_{key}", default)

    st.subheader("1. 固定レイアウトの設定")
    st.caption(
        "数値は画像全体に対するパーセントです。1日・31日は各列の中心、最初の職員行は氏名欄の最初の人の中心に合わせます。"
    )
    with st.expander("テンプレート設定（初回に調整し、JSONとして保存できます）", expanded=True):
        st.markdown("**列の位置**")
        c1, c2 = st.columns(2)
        with c1:
            st.slider("1日列の中心X位置（左から%）", 1.0, 98.0, step=0.1, key="cfg_day1_center_pct")
            st.slider("氏名欄の左端X位置（%）", 0.0, 30.0, step=0.1, key="cfg_name_left_pct")
        with c2:
            st.slider("31日列の中心X位置（左から%）", 2.0, 100.0, step=0.1, key="cfg_day31_center_pct")
            st.slider("氏名欄の右端X位置（%）", 1.0, 50.0, step=0.1, key="cfg_name_right_pct")

        st.markdown("**職員行の位置と間隔**")
        c3, c4 = st.columns(2)
        with c3:
            st.slider("最初の職員行の中心Y位置（上から%）", 1.0, 99.0, step=0.1, key="cfg_first_row_center_pct")
            st.slider("職員行の間隔（画像高さに対する%）", 0.4, 8.0, step=0.1, key="cfg_row_step_pct")
        with c4:
            st.slider("読み取る職員行数", 1, 60, step=1, key="cfg_row_count")
            st.slider("行セルの縦方向の切り出し幅（行間隔の比率）", 0.40, 0.95, step=0.01, key="cfg_row_crop_ratio")

        st.markdown("**文字画像の前処理**")
        c5, c6, c7 = st.columns(3)
        with c5:
            st.slider("文字画像の拡大倍率", 1, 5, step=1, key="cfg_upscale")
        with c6:
            st.checkbox("コントラストを補正", key="cfg_enhance_contrast")
        with c7:
            st.checkbox("白黒2値化", key="cfg_binarize", help="薄い文字が消える場合はオフのままにしてください。")
        st.slider("PDFの画像化解像度（DPI）", 150, 400, step=25, key="cfg_dpi")

    cfg = get_config()
    if float(cfg["day31_center_pct"]) <= float(cfg["day1_center_pct"]):
        st.error("1日列の中心が31日列の中心より右です。列位置を修正してください。")
    if float(cfg["name_right_pct"]) <= float(cfg["name_left_pct"]):
        st.error("氏名欄の右端が左端より左です。氏名欄の位置を修正してください。")

    st.download_button(
        "⚙️ 現在のテンプレート設定をJSONで保存",
        data=config_json_bytes(cfg),
        file_name="shift_template_config.json",
        mime="application/json",
    )

    token_text = st.text_input(
        "勤務記号（カンマ区切り。既存の候補を残し、必要に応じて追加してください）",
        value=DEFAULT_SHIFT_TOKENS,
    )
    allowed_tokens = [token.strip() for token in token_text.split(",") if token.strip()]

    if uploaded is None:
        st.caption("ファイルをアップロードすると、テンプレートの位置合わせ用プレビューを表示します。")
        return

    file_bytes = uploaded.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    preview_signature = f"{file_hash}|{cfg['dpi']}"
    if st.session_state.get("preview_signature") != preview_signature:
        try:
            with st.spinner("プレビュー画像を準備しています..."):
                images, total_pages, processed_pages = load_images(file_bytes, uploaded.name, int(cfg["dpi"]))
            st.session_state["preview_signature"] = preview_signature
            st.session_state["preview_images"] = images
            st.session_state["total_pages"] = total_pages
            st.session_state["processed_pages"] = processed_pages
            st.session_state["ocr_output"] = None
        except Exception as exc:
            st.error(f"PDF/画像の読み込みに失敗しました: {exc}")
            return

    images = st.session_state.get("preview_images", [])
    if not images:
        st.error("プレビュー画像がありません。ファイルを選び直してください。")
        return

    st.subheader("2. テンプレートの位置を確認")
    st.caption(
        "青枠が氏名欄、赤線が日付列の中心、緑枠が各職員行の切り出し範囲です。"
        "色の線が原本の列中心・職員行の中央に重なるように調整してください。"
    )
    preview = draw_template_overlay(images[0], cfg)
    st.image(preview, caption="テンプレートオーバーレイ（先頭ページ）", use_container_width=True)

    row_end_pct = float(cfg["first_row_center_pct"]) + (int(cfg["row_count"]) - 1) * float(cfg["row_step_pct"])
    if row_end_pct + float(cfg["row_step_pct"]) * float(cfg["row_crop_ratio"]) / 2 > 100:
        st.warning("設定された職員行数では、最終行の切り出し範囲がページ下端を越えます。行数・間隔を確認してください。")
    if st.session_state.get("total_pages", 1) > MAX_PDF_PAGES:
        st.warning(f"PDFは全{st.session_state['total_pages']}ページです。先頭{MAX_PDF_PAGES}ページまでを処理します。")

    config_signature = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    run_signature = hashlib.sha256((file_hash + config_signature + token_text).encode("utf-8")).hexdigest()

    if st.button("固定レイアウトでセルごとにOCRを実行", type="primary"):
        try:
            schedule_df, review_df = run_template_ocr(
                images,
                cfg,
                allowed_tokens,
            )
            st.session_state["ocr_output"] = {
                "signature": run_signature,
                "schedule_df": schedule_df,
                "review_df": review_df,
                "filename": uploaded.name,
                "total_pages": st.session_state.get("total_pages", len(images)),
                "processed_pages": st.session_state.get("processed_pages", len(images)),
            }
        except Exception as exc:
            st.error("OCRに失敗しました。Streamlit CloudのLogsで詳細を確認してください。")
            st.exception(exc)
            st.session_state["ocr_output"] = None

    result = st.session_state.get("ocr_output")
    if result is None or result.get("signature") != run_signature:
        if result is not None:
            st.info("テンプレート設定またはファイルが変わっています。最新の設定でOCRを再実行してください。")
        return

    schedule_df = result["schedule_df"].copy()
    review_df = result["review_df"].copy()
    st.subheader("3. 読み取った勤務シフト表（編集できます）")
    st.caption("各日のセルを直接修正できます。誤認識、空欄、氏名の誤りを原本と照合してください。")

    editor_key = f"schedule_editor_{run_signature[:16]}"
    edited_schedule = st.data_editor(
        schedule_df,
        key=editor_key,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "氏名": st.column_config.TextColumn("氏名", required=True, width="medium"),
            **{f"{day}日": st.column_config.TextColumn(f"{day}日", width="small") for day in range(1, 32)},
        },
    )

    review_count = 0
    if not review_df.empty:
        review_count = int((review_df["判定"] == "要確認").sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("抽出した職員行", len(edited_schedule))
    m2.metric("OCR確認セル", len(review_df))
    m3.metric("要確認セル", review_count)

    if not review_df.empty:
        with st.expander("OCR確認（低信頼度・認識結果・セル位置）", expanded=True):
            st.dataframe(review_df, use_container_width=True, hide_index=True)

    metadata = {
        "対象年月": target_ym,
        "施設名": facility_name.strip() or "（未入力）",
        "元ファイル名": result["filename"],
        "処理日時": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "総ページ数": result["total_pages"],
        "処理ページ数": result["processed_pages"],
        "OCR方式": "PaddleOCR 2.10 認識専用 + 固定テンプレートセル切り出し",
        "テンプレート設定": json.dumps(cfg, ensure_ascii=False),
        "勤務記号候補": ",".join(allowed_tokens),
        "注意": "勤務シフト表は氏名＋1日～31日。必ず原本と照合してください。",
    }
    excel_bytes = make_excel(edited_schedule[SCHEDULE_COLUMNS], review_df, metadata)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "📥 氏名×1日～31日のExcelをダウンロード",
        data=excel_bytes,
        file_name=f"勤務シフト表_OCR_固定レイアウト_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )


if __name__ == "__main__":
    main()
