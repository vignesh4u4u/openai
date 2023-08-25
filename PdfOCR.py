from flask import  Flask,request,render_template
from pdfminer.high_level import extract_text,extract_pages,extract_text_to_fp
from camelot import read_pdf
from tabula import read_pdf
import numpy as np
import scipy as scy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import json
from PIL import Image,ImageDraw
import os
import re
app=Flask(__name__,template_folder="template")
@app.route("/yorosis")
def home():
    return render_template("new1.html")
@app.route("/pre" ,methods="POST")
def extract_text_information_pdf():
    if request.method == 'POST':
        file = request.files
        selected_options = request.form["extractOptions"]
        file_path = "temp.pdf"
        file.save(file_path)
        with open(file_path, 'rb') as f:
            text = extract_text(f)
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)