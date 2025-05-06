import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os
import datetime

def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"An error occurred: {e}"
    
def extract_text_from_pdf(pdf_file):
    try:
        filename = datetime.datetime.now()
        temp_pdf_path = f"temp/{filename}.pdf"
        pdf_file.save(temp_pdf_path)

        pages = convert_from_path(temp_pdf_path, 350)
        pages_ocr = []

        for i, page in enumerate(pages, start=1):
            image_name = f"temp/Page_{i}.png"
            page.save(image_name, "PNG") 
            pages_ocr.append(extract_text_from_image(image_name))
            os.remove(image_name) 

        os.remove(temp_pdf_path)

        return pages_ocr

    except Exception as e:
        return f"An error occurred: {e}"