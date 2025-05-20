# Portfolio AI

Portfolio AI is a web application built with Flask that allows you to upload PDF files and process their contents using OCR (Optical Character Recognition). Then AI suggests best job offer.

## Requirements

- Python 3.8 or higher
- Python virtual environment (`venv`)
- Tesseract OCR library installed

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/portfolio-ai.git
cd portfolio-ai
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR on your machine:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install poppler-utils
```

5. Run the Flask application:

```bash
python app.py
```

6. Go to the following address:

```bash
http://127.0.0.1:5000/
```
