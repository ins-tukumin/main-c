import os
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# CSVファイルの読み込み
file_path = 'csvs/mainfiles/1001_selected_file.csv'
df = pd.read_csv(file_path)

# 不要な列を削除
# df = df.drop(columns=['タイムスタンプ', '学籍番号'])

# 不要な改行やスペースを取り除く関数
def clean_text(text):
    if isinstance(text, str):
        text = text.replace('\r\n', '')  # Windowsの改行をスペースに置換
        text = text.replace('\n', '')  # 改行をスペースに置換
    return text

# テキスト列をクリーンアップ
df['Q1'] = df['Q1'].apply(clean_text)

# 出力フォルダーのパスを指定
output_folder = 'pdfs'

# 出力フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# PDFを作成する関数
def create_pdf(student_id, diary_entries):
    pdf_file_name = os.path.join(output_folder, f"{student_id}.pdf")
    doc = SimpleDocTemplate(pdf_file_name, pagesize=A4)
    
    # フォントの登録
    pdfmetrics.registerFont(TTFont('IPAexGothic', 'fonts/ipaexg.ttf'))
    
    # スタイルの取得とフォントの設定
    styles = getSampleStyleSheet()
    styleN = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontName='IPAexGothic',
        fontSize=14,
        leading=20,
    )
    
    # PDF要素のリスト
    elements = []
    
    elements.append(Spacer(1, 12))
    
    # 各エントリの出力
    for index, row in diary_entries.iterrows():
        elements.append(Paragraph(row['StartDate_month_day'], styleN))
        elements.append(Paragraph(row['Q1'], styleN))
        elements.append(Spacer(1, 12))
    
    # PDFの生成
    doc.build(elements)
    print(f"PDF saved as {pdf_file_name}")

# 学籍番号ごとにデータをグループ化し、PDFを生成
grouped = df.groupby('user_id')

for student_id, group in grouped:
    create_pdf(student_id, group)

# 学籍番号を"ID_Number.pdf"に出力
id_number_pdf_path = os.path.join(output_folder, "ID_Number.pdf")
doc = SimpleDocTemplate(id_number_pdf_path, pagesize=A4)
styles = getSampleStyleSheet()
styleN = ParagraphStyle(
    'Normal',
    parent=styles['Normal'],
    fontName='IPAexGothic',
    fontSize=14,
    leading=20,
)
pdfmetrics.registerFont(TTFont('IPAexGothic', 'fonts/ipaexg.ttf'))

elements = []

for student_id, group in grouped:
    elements.append(Paragraph(f"{student_id}", styleN))
    elements.append(Spacer(1, 12))

doc.build(elements)
print(f"ID list saved as {id_number_pdf_path}")
