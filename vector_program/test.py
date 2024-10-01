import os
print("hi")
pdf_directory = "./pdfs"
for pdf_file in os.listdir(pdf_directory):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        print(f"Processing file: {pdf_path}")