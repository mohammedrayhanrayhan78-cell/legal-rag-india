import pdfplumber

with pdfplumber.open("CONSTITUTION OF INDIA.pdf") as pdf:
    for i in range(5):
        text = pdf.pages[i].extract_text()
        print(f"--- Page {i+1} ---")
        print(text[:300] if text else "EMPTY")
        print()