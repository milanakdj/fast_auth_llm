import os
from flask import Flask, render_template, request, jsonify
from engine import Engine


from engine_registrations import get_factory
factory = get_factory()
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    engines=factory.registered_keys()
    return render_template('index.html',engines=engines)

@app.route('/upload', methods=['POST'])
def upload():

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    engine_name = request.form["engine"]
    engine:Engine = factory.get_engine(engine_name)
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if(not os.path.exists(app.config['UPLOAD_FOLDER'])):
            os.mkdir(app.config["UPLOAD_FOLDER"])
           
     
        file.save(file_path)
        
        # Placeholder text for simulation
        extracted_text = engine.process(file_path)
        
        return extracted_text
    else:
        return "This file format is not supported"

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5500)
