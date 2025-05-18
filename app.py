from flask import Flask, render_template, redirect, url_for, flash
from forms import FileForm
from ocr import extract_text_from_pdf
from job_matching import CVJobMatchingSystem

app = Flask(__name__)
app.secret_key = '@( * O * )@'

processed_pages = []
job_offer = "We are looking for a JavaScript and Python developer with experience in machine learning and data analysis."

@app.route('/', methods=['GET', 'POST'])
def main():
    global processed_pages
    global job_offer
    form = FileForm()
    if form.validate_on_submit():
        uploaded_file = form.file.data
        job_offer = form.job_text.data
        print(f"Uploaded File Name: {uploaded_file.filename}")
        processed_pages = extract_text_from_pdf(uploaded_file)
        print(f"OCR finished successfully")
        return redirect(url_for('outcome'))
    return render_template('main.html', form=form)

@app.route('/outcome', methods=['GET', 'POST'])
def outcome():
    global processed_pages
    matcher = CVJobMatchingSystem()
    result = round(matcher.compare_cv_with_job(" ".join(processed_pages), job_offer) * 100, 2)
    return render_template('outcome.html', pages=processed_pages, result=result)


if __name__ == '__main__':
    with app.app_context():
        app.run(debug=True)