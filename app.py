from __future__ import annotations

import datetime as dt
import hashlib
import io
import statistics
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
from paddleocr import PaddleOCR


APP_TITLE = "手書き勤務シフト表 OCR 検証ツール"
MAX_PDF_PAGES = 10

DETAIL_COLUMNS = [
    "ページ", "行番号", "文字領域番号", "認識テキスト", "信頼度", "要確認",
    "X位置(%)", "Y位置(%)", "左端(px)", "上端(px)", "右端(px)", "下端(px)"
]
ROW_COLUMNS = ["ページ", "行番号", "Y位置(%)", "領域数", "認識テキスト", "要確認件数"]


@st.cache_resource(show_spinner=False)
def get_ocr_engine() -> PaddleOCR:
    """PaddleOCR 2.10系の日本語OCRエンジンを初期化してキャッシュする。"""
    return PaddleOCR(
        use_angle_cls=True,
        lang="japan",
        use_gpu=False,
        show_log=False,
    )


def load_images(file_bytes: bytes, file_name: str, dpi: int) -> tuple[list[Image.Image], int, int]:
    """アップロードされたPDFまたは画像をRGB画像として読み込む。

    PDFはPopplerを利用し、先頭から最大MAX_PDF_PAGESページまでを変換する。
    戻り値は (画像リスト, PDF等の総ページ数, 実際に変換したページ数)。
    """
    suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""

    if suffix == "pdf":
        info = pdfinfo_from_bytes(file_bytes)
        total_pages = int(info.get("Pages", 0))
        if total_pages < 1:
            raise ValueError("PDFからページ数を取得できませんでした。PDFが破損していないか確認してください。")

        pages_to_process = min(total_pages, MAX_PDF_PAGES)
        images = convert_from_bytes(
            file_bytes,
            dpi=dpi,
            first_page=1,
            last_page=pages_to_process,
            fmt="png",
            thread_count=1,
        )
        return [im.convert("RGB") for im in images], total_pages, len(images)

    try:
        with Image.open(io.BytesIO(file_bytes)) as source_image:
            image = ImageOps.exif_transpose(source_image).convert("RGB")
    except Exception as exc:
        raise ValueError(f"画像ファイルを読み込めませんでした: {exc}") from exc
    return [image], 1, 1


def _extract_page_detections(
    ocr_result: Any,
    page_no: int,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """PaddleOCR 2.xの戻り値から文字、信頼度、位置を抽出する。"""
    detections: list[dict[str, Any]] = []
    if not ocr_result:
        return detections

    # PaddleOCR 2.xの通常の戻り値は [ [ [box, (text, score)], ... ] ]。
    page_items = ocr_result[0] if isinstance(ocr_result, list) and len(ocr_result) > 0 else None
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

        x_min = float(np.min(points[:, 0]))
        y_min = float(np.min(points[:, 1]))
        x_max = float(np.max(points[:, 0]))
        y_max = float(np.max(points[:, 1]))
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        detections.append({
            "page": page_no,
            "text": text,
            "score": score,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "center_x": center_x,
            "center_y": center_y,
            "width": max(1.0, x_max - x_min),
            "height": max(1.0, y_max - y_min),
            "image_width": image_width,
            "image_height": image_height,
        })
    return detections


def group_detections_into_rows(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """文字領域のY座標から、同じ高さにある領域を暫定的に行へまとめる。

    これは表のセルを確定する処理ではなく、OCR結果確認のための行グルーピング。
    """
    grouped_rows: list[dict[str, Any]] = []
    pages = sorted({int(item["page"]) for item in detections})

    for page_no in pages:
        page_items = [item for item in detections if int(item["page"]) == page_no]
        page_items.sort(key=lambda item: (item["center_y"], item["center_x"]))
        if not page_items:
            continue

        typical_height = statistics.median(item["height"] for item in page_items)
        y_tolerance = max(8.0, typical_height * 0.65)
        clusters: list[list[dict[str, Any]]] = []

        for item in page_items:
            best_cluster = None
            best_distance = None
            for cluster in clusters:
                cluster_y = statistics.median(member["center_y"] for member in cluster)
                distance = abs(item["center_y"] - cluster_y)
                if distance <= y_tolerance and (best_distance is None or distance < best_distance):
                    best_cluster = cluster
                    best_distance = distance
            if best_cluster is None:
                clusters.append([item])
            else:
                best_cluster.append(item)

        clusters.sort(key=lambda cluster: statistics.median(item["center_y"] for item in cluster))
        for row_no, cluster in enumerate(clusters, start=1):
            cluster.sort(key=lambda item: item["center_x"])
            text = "  |  ".join(item["text"] for item in cluster)
            image_height = max(1, int(cluster[0]["image_height"]))
            low_confidence_count = sum(item["score"] < 0.85 for item in cluster)
            grouped_rows.append({
                "ページ": page_no,
                "行番号": row_no,
                "Y位置(%)": round(statistics.median(item["center_y"] for item in cluster) / image_height * 100, 2),
                "領域数": len(cluster),
                "認識テキスト": text,
                "要確認件数": low_confidence_count,
            })

    return grouped_rows


def make_dataframes(detections: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rows = group_detections_into_rows(detections)
    row_df = pd.DataFrame(rows, columns=ROW_COLUMNS)

    # 明細ごとに、Y座標が最も近い暫定行を割り当てる。
    detail_records: list[dict[str, Any]] = []
    for page_no in sorted({int(item["page"]) for item in detections}):
        page_detections = [item for item in detections if int(item["page"]) == page_no]
        page_detections.sort(key=lambda item: (item["center_y"], item["center_x"]))
        # 1行内の領域をまとめるため、行グルーピング結果から中心Yが近い行を探す。
        page_rows = [r for r in rows if int(r["ページ"]) == page_no]
        for item in page_detections:
            image_height = max(1, int(item["image_height"]))
            image_width = max(1, int(item["image_width"]))
            nearest_row_no = 0
            nearest_distance = None
            for row in page_rows:
                # Y位置は%で格納されているためpxへ戻す。
                row_y_px = float(row["Y位置(%)"]) / 100 * image_height
                distance = abs(item["center_y"] - row_y_px)
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_row_no = int(row["行番号"])

            score = float(item["score"])
            detail_records.append({
                "ページ": page_no,
                "行番号": nearest_row_no,
                "文字領域番号": len(detail_records) + 1,
                "認識テキスト": item["text"],
                "信頼度": round(score, 4),
                "要確認": "要確認" if score < 0.85 else "",
                "X位置(%)": round(item["center_x"] / image_width * 100, 2),
                "Y位置(%)": round(item["center_y"] / image_height * 100, 2),
                "左端(px)": round(item["x_min"], 1),
                "上端(px)": round(item["y_min"], 1),
                "右端(px)": round(item["x_max"], 1),
                "下端(px)": round(item["y_max"], 1),
            })

    detail_df = pd.DataFrame(detail_records, columns=DETAIL_COLUMNS)
    raw_text = "\n".join(
        str(row["認識テキスト"])
        for row in rows
    )
    return row_df, detail_df, raw_text


def generate_excel(
    meta_info: dict[str, Any],
    row_df: pd.DataFrame,
    detail_df: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()
    meta_df = pd.DataFrame(
        [{"項目": key, "値": value} for key, value in meta_info.items()],
        columns=["項目", "値"],
    )

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        meta_df.to_excel(writer, sheet_name="処理情報", index=False)
        row_df.to_excel(writer, sheet_name="OCR行", index=False)
        detail_df.to_excel(writer, sheet_name="OCR明細", index=False)

        workbook = writer.book
        header_format = workbook.add_format({
            "bold": True,
            "text_wrap": True,
            "valign": "top",
            "border": 1,
        })
        for sheet_name, dataframe in [
            ("処理情報", meta_df),
            ("OCR行", row_df),
            ("OCR明細", detail_df),
        ]:
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            if len(dataframe.columns) > 0:
                worksheet.autofilter(0, 0, max(0, len(dataframe)), len(dataframe.columns) - 1)
                for col_index, col_name in enumerate(dataframe.columns):
                    worksheet.write(0, col_index, col_name, header_format)
                    # 長文の認識結果列は広くしすぎない。
                    if col_name == "認識テキスト":
                        worksheet.set_column(col_index, col_index, 55)
                    else:
                        worksheet.set_column(col_index, col_index, min(max(len(str(col_name)) + 2, 12), 24))

    return output.getvalue()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title("📄 手書き勤務シフト表 OCR 検証ツール")
    st.write(
        "PDFまたは画像から日本語・数字・アルファベットの文字を認識し、"
        "認識位置と信頼度を付けて表示・Excel出力します。"
    )
    st.warning(
        "この版はOCR結果を確認するための検証用です。"
        "勤務記号を『職員名 × 日付』のセルへ自動で確定転記する機能はまだありません。"
        "手書きの表はレイアウトにより認識が変わるため、出力を必ず原本と照合してください。"
    )
    st.info(
        "Streamlit Community Cloudへアップロードしたファイルはクラウド環境で処理されます。"
        "個人情報・勤務情報を含む実データは、会社の利用規定を確認してからアップロードしてください。"
    )

    left_col, right_col = st.columns(2)
    with left_col:
        target_ym = st.text_input("対象年月", value=dt.datetime.now().strftime("%Y/%m"))
    with right_col:
        facility_name = st.text_input("施設名（任意）", value="")

    uploaded_file = st.file_uploader(
        "勤務シフト表（PDFまたは画像）",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
        help=f"PDFは先頭から最大{MAX_PDF_PAGES}ページまで処理します。",
    )
    dpi = st.slider("PDFの画像化解像度（DPI）", min_value=150, max_value=350, value=250, step=50)

    if "ocr_payload" not in st.session_state:
        st.session_state["ocr_payload"] = None

    file_bytes = uploaded_file.getvalue() if uploaded_file is not None else b""
    if file_bytes:
        signature_source = (
            hashlib.sha256(file_bytes).hexdigest()
            + "|" + target_ym
            + "|" + facility_name
            + "|" + str(dpi)
        )
        signature = hashlib.sha256(signature_source.encode("utf-8")).hexdigest()
    else:
        signature = ""
    existing_payload = st.session_state.get("ocr_payload")
    if existing_payload and existing_payload.get("signature") != signature:
        st.session_state["ocr_payload"] = None

    run_clicked = st.button("OCRを実行", type="primary", disabled=uploaded_file is None)
    if run_clicked and uploaded_file is not None:
        try:
            with st.spinner("PDF/画像を読み込んでいます..."):
                images, total_pages, processed_pages = load_images(file_bytes, uploaded_file.name, dpi)

            with st.spinner("PaddleOCRで文字を認識しています。初回はモデルのダウンロードに時間がかかる場合があります..."):
                ocr = get_ocr_engine()
                all_detections: list[dict[str, Any]] = []
                progress = st.progress(0, text="OCRを準備しています...")
                for page_no, image in enumerate(images, start=1):
                    image_array = np.asarray(image.convert("RGB"))
                    result = ocr.ocr(image_array, cls=True)
                    page_detections = _extract_page_detections(
                        result,
                        page_no,
                        image.width,
                        image.height,
                    )
                    all_detections.extend(page_detections)
                    progress.progress(page_no / max(1, len(images)), text=f"{page_no}/{len(images)}ページを処理しました")
                progress.empty()

            row_df, detail_df, raw_text = make_dataframes(all_detections)
            processed_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            meta_info = {
                "対象年月": target_ym,
                "施設名": facility_name if facility_name.strip() else "（未入力）",
                "ファイル名": uploaded_file.name,
                "処理日時": processed_at,
                "総ページ数": total_pages,
                "処理ページ数": processed_pages,
                "PDF変換DPI": dpi if uploaded_file.name.lower().endswith(".pdf") else "画像ファイルのため対象外",
                "OCR明細数": len(detail_df),
                "注意": "OCR行・OCR明細は確認用であり、勤務シフトの確定データではありません。",
            }
            excel_bytes = generate_excel(meta_info, row_df, detail_df)
            st.session_state["ocr_payload"] = {
                "signature": signature,
                "row_df": row_df,
                "detail_df": detail_df,
                "raw_text": raw_text,
                "excel_bytes": excel_bytes,
                "meta_info": meta_info,
                "processed_pages": processed_pages,
                "total_pages": total_pages,
                "preview_image": images[0].copy() if images else None,
            }
        except Exception as exc:
            st.session_state["ocr_payload"] = None
            st.error("OCR処理に失敗しました。エラー内容を確認してください。")
            st.exception(exc)

    payload = st.session_state.get("ocr_payload")
    if payload and payload.get("signature") == signature:
        st.success(
            f"OCR処理が完了しました。認識した文字領域数: {len(payload['detail_df'])} "
            f"（{payload['processed_pages']}/{payload['total_pages']}ページを処理）"
        )
        if payload["total_pages"] > payload["processed_pages"]:
            st.warning(
                f"PDFは全{payload['total_pages']}ページありますが、リソース節約のため"
                f"先頭{payload['processed_pages']}ページのみ処理しました。"
            )

        with st.expander("原本プレビュー（先頭ページ）", expanded=False):
            if payload.get("preview_image") is not None:
                st.image(payload["preview_image"], use_container_width=True)

        st.subheader("1. OCR行（Y座標が近い文字領域をまとめた確認用表示）")
        st.caption("同じ高さに検出された文字を左から右に並べています。日付・勤務セルへの対応付けは未確定です。")
        st.dataframe(payload["row_df"], use_container_width=True, hide_index=True)

        st.subheader("2. OCR明細（位置・信頼度）")
        if not payload["detail_df"].empty:
            low_count = int((payload["detail_df"]["要確認"] == "要確認").sum())
            st.metric("信頼度0.85未満（要確認）の領域", low_count)
        st.dataframe(payload["detail_df"], use_container_width=True, hide_index=True)

        with st.expander("抽出されたテキスト", expanded=False):
            st.text(payload["raw_text"] or "文字を検出できませんでした。")

        file_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            label="📥 OCR結果をExcelでダウンロード",
            data=payload["excel_bytes"],
            file_name=f"ShiftOCR_{file_stamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.caption(
            "次の段階で勤務表を職員名・日付・勤務記号の表へ変換するには、"
            "実際のシフト表のレイアウトに合わせたセル割り当て処理が必要です。"
        )


if __name__ == "__main__":
    main()
