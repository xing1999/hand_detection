# helper
import os
import numpy as np
from io import BytesIO
import base64

from interface import SSD_Interface

# main app
from flask import Flask, render_template, request,jsonify
from PIL import Image
import PIL

weight_path = "./weights/xing_weight.pth"
model = SSD_Interface(
    weight_path,
    list_classes=None, # Use default - [None, "hand"]
    conf_thres=0.3,
)

    
app = Flask(__name__)


# Ignore error 15 omp
os.environ['KMP_DUPLICATE_LIB_OK']='True'

@app.route('/')
def hello_world():
    return 'Hello World'


import json
@app.route('/', methods=['POST'])
def results():
    
    if request.method == 'POST':
        # convert bytes to dict if not python (send file instead data)
        # rdata = request.json
        # print(rdata)
        cur_lang = 'python'
        try: 
            # dict_str = request.decode("UTF-8")            
            # request.data = json.loads(dict_str)
            cur_lang = request.json['lang']
        except:
            pass    
        print("Client language:", cur_lang)        
        # each type of lang has difference way to read img, cpp case is decode base64, python send file in request instead data as cpp
        # but current i dont know how to wrap lang into data field.
        if cur_lang == 'python':
            img_file = request.files['image']
            img = Image.open(img_file.stream)
        elif cur_lang == 'cpp':
            img = Image.open(BytesIO(base64.b64decode(request.json['img_encoded'])))
        else: 
            img = Image.open(BytesIO(base64.b64decode(request.json['img_encoded'])))
        
        img = img.convert("RGB") 
        state = request.json['type']  

        model_result = model.process(img)
        return jsonify(model_result)

# app.run("localhost", "3000", debug=True)
app.run("0.0.0.0", "3000", debug=True)
