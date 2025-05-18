from flask import Flask, render_template, redirect, url_for, flash
from forms import FileForm
from ocr import extract_text_from_pdf
from job_matching import CVJobMatchingSystem

app = Flask(__name__)
app.secret_key = '@( * O * )@'

processed_pages = []

@app.route('/', methods=['GET', 'POST'])
def main():
    global processed_pages
    form = FileForm()
    if form.validate_on_submit():
        uploaded_file = form.file.data
        print(f"Uploaded File Name: {uploaded_file.filename}")
        processed_pages = extract_text_from_pdf(uploaded_file)
        print(f"OCR finished successfully")
        return redirect(url_for('outcome'))
    return render_template('main.html', form=form)

@app.route('/outcome', methods=['GET', 'POST'])
def outcome():
    global processed_pages
    matcher = CVJobMatchingSystem()
    result = matcher.compare_cv_with_job(processed_pages[0])
    print(result)
    return render_template('outcome.html', pages=processed_pages)


if __name__ == '__main__':
    with app.app_context():
        app.run(debug=True)