from flask import  Flask,request,render_template
from paddleocr import PaddleOCR,draw_ocr
from PIL import Image,ImageDraw
import  numpy as np
import  matplotlib.pyplot as plt
import pytesseract as pytes
import requests
import torch
import easyocr
import cv2
import re
import os
import json
reader=easyocr.Reader(lang_list=["en"])
ocr = PaddleOCR(use_angle_cls=True, lang='en')
pytes.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
app=Flask(__name__,template_folder="template")
@app.route("/")
@app.route("/yorosis")
def home():
    return render_template("new.html")
@app.route("/predict", methods=["POST"])
def image_text_conversion():
    if request.method == "POST":
        image_file = request.files['image']
        image_data = image_file.read()
        fields_input = request.form.get("fields")
        fields = json.loads(fields_input)
        gray = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        result = ocr.ocr(gray)
        detected_text_list = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                detected_text_list.append(line[1][0])
        text = ' '.join(detected_text_list)
        #print(text)
        data={}
        for field in fields:
            key = field.get("key")
            pattern = field.get("pattern")
            repeatable = field.get("repeatable", True)
            if pattern:
                matches = re.findall(pattern, text,flags=re.IGNORECASE)
                if matches:
                    if repeatable:
                        data[key] = matches
                    else:
                        data[key] = matches[0]
        # create the countors to find the table data inside table.
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = 1000
        table_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        table_boundaries = [cv2.boundingRect(cnt) for cnt in table_contours]
        table_data = []
        for x, y, w, h in table_boundaries:
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0,0, 255), 2)
            table_image = gray[y:y + h, x:x + w]
            text = pytes.image_to_string(table_image)
            rows = text.strip().split("\n" or "\t")
            table_data = [row.split() for row in rows]
        plt.figure(figsize=(20, 20))
        c = plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
        plt.title('Table Detection')
        plt.show()
    return render_template("new.html",**locals())
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
