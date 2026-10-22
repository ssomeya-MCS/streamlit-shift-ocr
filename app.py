import streamlit as st
import pandas as pd
import io
import time
import datetime

# ==========================================
# 1. OCRエンジンのモック関数
# （実際はここにGoogle Cloud Vision APIやGemini/GPT-4Vの処理を組み込みます）
# ==========================================
def extract_shift_data_mock(uploaded_file):
    """
    アップロードされたPDF/画像から手書きシフトを読み取ったと仮定し、
    ダミーの構造化データを返すモック関数。
    ※ソースコード上に実際の個人名や施設名はハードコードしない。
    """
    # 疑似的なOCR処理の待機時間
    time.sleep(2)
    
    # 抽出された設定情報（対象年月や施設名は外部入力またはOCRから取得）
    meta_info = {
        "target_ym": "2026/07",
        "facility_name": "[Facility_X]" # プレースホルダー
    }
    
    # 抽出されたシフトデータ（従業員名と1日〜31日までのシフト記号）
    # ※個人名はダミー（Employee_A等）を使用
    extracted_records = [
        {"emp_name": "Employee_A", "shifts": ["M", "M", "休", "M", "M", "休", "M", "A", "B", "C", "休", "M", "M", "休", "A", "B", "C", "休", "M", "M", "休", "A", "B", "C", "休", "M", "M", "休", "A", "B", "C"]},
        {"emp_name": "Employee_B", "shifts": ["A", "B", "C", "休", "A", "B", "C", "休", "A", "B", "C", "休", "A", "B", "C", "休", "A", "B", "C", "休", "A", "B", "C", "休", "A", "B", "C", "休", "A", "B", "C"]},
        {"emp_name": "Employee_C", "shifts": ["休", "A", "B", "C", "休", "A", "B", "C", "休", "A", "B", "C", "休", "A", "B", "C", "休", "A", "B", "C", "休", "A", "B", "C", "休", "A", "B", "C", "休", "A", "B"]},
        {"emp_name": "Employee_D", "shifts": ["D1", "D1", "休", "D1", "D1", "休", "D1", "D1", "休", "D1", "D1", "休", "D1", "D1", "休", "D1", "D1", "休", "D1", "D1", "休", "D1", "D1", "休", "D1", "D1", "休", "D1", "D1", "休", "D1"]},
    ]
    
    return meta_info, extracted_records

# ==========================================
# 2. Excelファイル（出力用フォーマット）の生成関数
# ==========================================
def generate_excel_file(meta_info, records):
    """
    OCRで抽出したデータをPandasを用いてExcel（.xlsx）のバイナリデータに変換する。
    """
    # Excelの列定義（勤怠システムの取り込みフォーマットに合わせる）
    columns = ["対象年月", "施設名", "従業員名"] + [f"{i}日" for i in range(1, 32)]
    
    data_rows = []
    for rec in records:
        # 月や施設名は行ごとにセット、またはヘッダ部に持たせる仕様に合わせる
        row = [meta_info["target_ym"], meta_info["facility_name"], rec["emp_name"]]
        # 抽出したシフト（最大31日分）を結合。足りない日数は空文字で埋める
        shifts = rec["shifts"]
        shifts += [""] * (31 - len(shifts)) 
        row.extend(shifts[:31])
        data_rows.append(row)
        
    df = pd.DataFrame(data_rows, columns=columns)
    
    # メモリ上でExcelファイルを生成
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='ShiftData')
    
    return output.getvalue()

# ==========================================
# 3. Streamlit UI 構築
# ==========================================
def main():
    st.set_page_config(page_title="手書きシフトOCRデモ", layout="wide")
    
    st.title("📄 手書きシフト表 OCR自動転記デモ")
    st.markdown("""
    手書きのシフト表（PDF/画像）をアップロードすると、AIが内容を読み取り、
    勤怠システムへコピペできるExcelフォーマットに自動変換します。
    ※本環境は検証用のデモ（モック）です。入力データは保存されません。
    """)
    
    # ファイルアップロードUI
    uploaded_file = st.file_uploader("シフト表のファイル（PDF）をアップロードしてください", type=["pdf"])
    
    if uploaded_file is not None:
        st.success(f"ファイル '{uploaded_file.name}' を読み込みました。")
        
        if st.button("OCRでテキスト抽出を実行"):
            with st.spinner("AIが手書き文字を解析・抽出しています..."):
                # モック関数を呼び出し
                meta_info, records = extract_shift_data_mock(uploaded_file)
            
            st.success("抽出が完了しました！")
            
            # 抽出結果のプレビュー表示
            st.subheader("🔍 抽出結果プレビュー")
            st.write(f"**対象年月:** {meta_info['target_ym']} | **施設名:** {meta_info['facility_name']}")
            
            # プレビュー用のDataFrameを作成
            preview_columns = ["従業員名"] + [f"{i}" for i in range(1, 32)]
            preview_data = []
            for r in records:
                row = [r["emp_name"]] + (r["shifts"] + [""] * 31)[:31]
                preview_data.append(row)
            
            df_preview = pd.DataFrame(preview_data, columns=preview_columns)
            st.dataframe(df_preview, use_container_width=True)
            
            # Excelファイルの生成とダウンロードボタン
            excel_data = generate_excel_file(meta_info, records)
            
            st.markdown("---")
            st.subheader("💾 データのエクスポート")
            st.markdown("抽出したデータを勤怠システム入力用のExcelフォーマットでダウンロードします。")
            
            # ファイル名を動的に生成（例: ShiftData_20261021_1234.xlsx）
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            file_name = f"ShiftData_{timestamp}.xlsx"
            
            st.download_button(
                label="📥 Excelファイルを出力してダウンロード",
                data=excel_data,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()