import os
from flask import Flask, request, jsonify, make_response
from werkzeug.utils import secure_filename
from moduls.preprocess import *
from moduls.process import *

UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = {'tif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.post('/upload')
def upload_crop():
    if 'file' not in request.files:
        return make_response(jsonify(success = False), 400)
    file = request.files['file']
    if file.filename == '':
        return make_response(jsonify(success = False), 400)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    resp = jsonify(success=True)
    return resp

@app.post('/fix')
def fix():
    if not request.is_json:
        return make_response(jsonify(success=False), 400)
    data = request.json
    name = data.get('filename')
    if not name:
        return jsonify(success = False)
    try:
        process_image(app.config['UPLOAD_FOLDER'], name)
    except:
        return make_response(jsonify(success = False), 500)
    return jsonify(success=True)
    
@app.get('/predict')
def predict():
    layout_name = request.args.get('layout')
    filename = request.args.get('filename')
    if not layout_name or not filename:
        return make_response(jsonify(success = False), 400)
    try:
        result = process_images('./layouts', app.config['UPLOAD_FOLDER'], layout_name, filename, './geotiff', './geotiff')
    except:
        return make_response(jsonify(success = False), 500)
    return result
    
    

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 80)