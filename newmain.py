from flask import Flask,request,render_template,jsonify,json
from paddleocr import PaddleOCR,draw_ocr
import pytesseract as pytes
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import easyocr
import os
import re
import cv2
from PIL import Image
import requests
import json
ocr = PaddleOCR(use_angle_cls=True,lang='en')
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
        gray = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        #text = pytes.image_to_string(gray, lang="eng+fra", config='--oem 3 --psm 6')
        fields_input = request.form.get("fields")
        fields = json.loads(fields_input)
        result = ocr.ocr(gray)
        detected_text_list = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                detected_text_list.append(line[1][0])
        text = ' '.join(detected_text_list)
        #print(text)
        #print(fields)
        data ={}
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
        tables = []
        for x, y, w, h in table_boundaries:
            table_image = gray[y:y + h, x:x + w]
            text = pytes.image_to_string(table_image)
            rows = text.strip().split("\n")
            table_data = [row.split() for row in rows]
            table = {
                'table_data': table_data,
                'x': x,
                'y': y,
                'width': w,
                'height': h
                }
            tables.append(table)
        for field in fields:
            key = field.get("key")
            repeatable = field.get("repeatable", True)
            header_row = table_data[0]
            headers = header_row[0:]
            if key in headers:
                column_data = {header: [] for header in headers}
                for row in table_data[1:]:
                    for header, value in zip(headers, row[1:]):
                        column_data[header].append(value)
                json_data = []
                for i in range(len(column_data[headers[0]])):
                    data_item = {}
                    for header in headers:
                        data_item[header] = column_data[header][i]
                    json_data.append(data_item)
                    json_output = json.dumps(json_data)
                data[key] = json_data
        print(data)
        # return jsonify(data)
    return render_template("new.html",**locals())
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)