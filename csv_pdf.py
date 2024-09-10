import os
import pandas as pd
import random
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# CSVファイルの読み込み
file_path = 'csvs/test0704.csv'
df = pd.read_csv(file_path)

# 不要な列を削除
df = df.drop(columns=['タイムスタンプ', '学籍番号'])

# 不要な改行やスペースを取り除く関数
def clean_text(text):
    if isinstance(text, str):
        text = text.replace('\r\n', '')  # Windowsの改行をスペースに置換
        text = text.replace('\n', '')  # 改行をスペースに置換
    return text

# テキスト列をクリーンアップ
df['日記'] = df['日記'].apply(clean_text)

# 出力フォルダーのパスを指定
output_folder = 'vector_lab_pdfs'

# 出力フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 学籍番号のリストを取得
student_ids = df['名前'].unique()

# 6桁の乱数を生成する関数
def generate_unique_random_numbers(n, length=6):
    random_numbers = set()
    while len(random_numbers) < n:
        random_number = ''.join(random.choices('0123456789', k=length))
        random_numbers.add(random_number)
    return list(random_numbers)

# 学籍番号の数だけ6桁の乱数を生成
random_numbers = generate_unique_random_numbers(len(student_ids))

# 学籍番号と乱数の対応を保存する辞書
student_id_to_random = dict(zip(student_ids, random_numbers))

# PDFを作成する関数
def create_pdf(random_number, diary_entries):
    pdf_file_name = os.path.join(output_folder, f"{random_number}.pdf")
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
        elements.append(Paragraph(row['日付'], styleN))
        elements.append(Paragraph(row['日記'], styleN))
        elements.append(Spacer(1, 12))
    
    # PDFの生成
    doc.build(elements)
    print(f"PDF saved as {pdf_file_name}")

# 学籍番号ごとにデータをグループ化し、PDFを生成
grouped = df.groupby('名前')

for student_id, group in grouped:
    random_number = student_id_to_random[student_id]
    create_pdf(random_number, group)

# 学籍番号と乱数の対応を"ID_Number.pdf"に出力
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

for student_id, random_number in student_id_to_random.items():
    elements.append(Paragraph(f"{student_id} : {random_number}", styleN))
    elements.append(Spacer(1, 12))

doc.build(elements)
print(f"ID and random number mapping saved as {id_number_pdf_path}")
