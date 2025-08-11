from pdf2docx import Converter
import os

PDF_PATH = '/home/gyasis/Downloads/Contratto.pdf'
OUTPUT_DOCX = 'Contratto_high_fidelity.docx'

def main():
    if not os.path.exists(PDF_PATH):
        print(f"PDF not found: {PDF_PATH}")
        return
    print("Converting PDF to Word with high fidelity (layout, headers, footers, etc)...")
    cv = Converter(PDF_PATH)
    cv.convert(OUTPUT_DOCX, start=0, end=None)
    cv.close()
    print(f"Saved as {OUTPUT_DOCX}")

if __name__ == '__main__':
    main() 