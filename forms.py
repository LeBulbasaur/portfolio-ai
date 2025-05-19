from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms.validators import DataRequired, Length

class FileForm(FlaskForm):
    file = FileField('Upload your portfolio', validators=[
        FileRequired(),
        FileAllowed(['pdf'], 'Only PDF files are allowed!')
    ])
    job_text = TextAreaField('Paste your job offer', validators=[DataRequired(), Length(min=4, max=1000)])
    submit = SubmitField('Submit')