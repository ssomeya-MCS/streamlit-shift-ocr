勤務シフト表 PaddleOCR版（Streamlit Community Cloud向け）

ファイル:
- app.py: OCR、氏名×1日～31日の表変換、編集可能プレビュー、Excel出力
- requirements.txt: Python依存ライブラリ
- packages.txt: PDF変換のPopplerとOpenCV用のLinux依存ライブラリ

推奨設定:
- Streamlit Community CloudのPython: 3.11
- Branch: リポジトリで使用しているブランチ名（例: PDFadd）
- Main file path: app.py

注意:
- PDF/画像の行列レイアウトは表ごとに異なります。日付見出しを自動検出できない場合、アプリ上の「1日列の中心」「31日列の中心」を調整してください。
- 主シートは「氏名」「1日」～「31日」です。OCR確認シートには検出文字、信頼度、位置を残します。
- OCR認識結果は勤務確定データではありません。原本と照合してから業務利用してください。
- 実行中にImportErrorが残る場合は、Streamlit Cloud Logsの完全な最終例外行を確認してください。現在の依存関係の組み合わせで、Cloudでの実動作までは保証していません。
