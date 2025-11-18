from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
from flask_cors import CORS 

app = Flask(__name__)
CORS(app) 

# Route for the detection app
# @app.route('/')
def index():
    return redirect(url_for('detection_page'))  # Default route redirects to detection page

@app.route('/detection')
def detection_page():
    return render_template('indexDetection.html')

@app.route("/resultDetection", methods=["POST"])
def predict_deficiency_detection():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        interpreter = load_tflite_model_detection()
        class_index, prediction_probabilities = predict_with_tflite(image, interpreter)
        deficiency_type = get_deficiency_detection(class_index)

        accuracy = prediction_probabilities[class_index] * 100  # Convert to percentage
        response_data = {'deficiency_type': deficiency_type}

        if accuracy < 50.0 or class_index == 1: # 1 corresponds to "Invalid" class
            response_data = {'deficiency_type': "Invalid"}
        else:
            response_data['accuracy'] = f"{accuracy:.2f}%"

        image = image.convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_data = base64.b64encode(buffered.getvalue()).decode()
        

        return jsonify({
            'deficiency_type': response_data['deficiency_type'],
            'accuracy': response_data.get('accuracy', None),
            'img_data': img_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Route for the monitoring app
@app.route('/monitor')
def monitor_page():
    return render_template('indexMonitor.html')

@app.route("/resultMonitor", methods=["POST"])
def predict_deficiency_monitor():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        interpreter = load_tflite_model_monitor()
        class_index, prediction_probabilities = predict_with_tflite(image, interpreter)
        deficiency_type = get_deficiency_monitor(class_index)

        accuracy = prediction_probabilities[class_index] * 100  # Convert to percentage
        response_data = {'deficiency_type': deficiency_type}

        if accuracy < 50.0 or class_index == 1:  # 1 corresponds to "Invalid" class
            response_data = {'deficiency_type': "Invalid"}
        else:
            response_data['accuracy'] = f"{accuracy:.2f}%"

        image = image.convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_data = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'deficiency_type': response_data['deficiency_type'],
            'accuracy': response_data.get('accuracy', None),
            'img_data': img_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/resultDetection/batch", methods=["POST"])
def predict_batch_deficiency_detection():
    if "files" not in request.files:
        return jsonify({"error": "No files in the request"}), 400

    files = request.files.getlist("files")
    
    if len(files) == 0:
        return jsonify({"error": "No files selected"}), 400

    try:
        predictions = []
        interpreter = load_tflite_model_detection()
        
        for file in files:
            image = Image.open(io.BytesIO(file.read()))
            class_index, prediction_probabilities = predict_with_tflite(image, interpreter)
            deficiency_type = get_deficiency_detection(class_index)

            accuracy = prediction_probabilities[class_index] * 100
            
            if accuracy < 50.0 or class_index == 1:
                prediction_data = {'deficiency_type': "Invalid", 'accuracy': None}
            else:
                prediction_data = {
                    'deficiency_type': deficiency_type,
                    'accuracy': f"{accuracy:.2f}%"
                }
            
            predictions.append(prediction_data)

        return jsonify({"predictions": predictions})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Load TFLite model and initialize interpreter for Detection
def load_tflite_model_detection():
    interpreter = tf.lite.Interpreter(model_path="cornModel.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Load TFLite model and initialize interpreter for Monitoring
def load_tflite_model_monitor():
    interpreter = tf.lite.Interpreter(model_path="Monitor.tflite")
    interpreter.allocate_tensors()
    return interpreter


# Preprocess image for TFLite model
def preprocess_image(image):
    image = image.convert('RGB')
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction using TFLite model
def predict_with_tflite(image, interpreter):
    img_array = preprocess_image(image)
    
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    
    interpreter.set_tensor(input_index, img_array)
    interpreter.invoke()
    
    predictions = interpreter.get_tensor(output_index)
    
    class_index = np.argmax(predictions)
        # 🔍 Debug print logs
    print("Raw model output:", predictions)
    class_index = np.argmax(predictions)
    print("Class index:", class_index)
    accuracy = predictions[0][class_index]
    print("Accuracy:", accuracy)
    prediction_probabilities = predictions[0]
    return class_index, prediction_probabilities

# Deficiency mapping for Detection
def get_deficiency_detection(class_index):
    deficiency_mapping = {
        0: 'Healthy',
        1: 'Invalid',
     2: 'Nitrogen',
     3: 'Phosphorus',
      4: 'Potassium',
       
    }
    return deficiency_mapping.get(class_index, 'Unknown')

# Deficiency mapping for Monitoring
def get_deficiency_monitor(class_index):
    deficiency_mapping = {
        0: 'Healthy',
        1: 'Invalid',
        2: 'Mild',
        3: 'Moderate',
        4: 'Severe',
    }
    return deficiency_mapping.get(class_index, 'Unknown')


if __name__ == "__main__":
    # app.run(host='192.168.100.105',port=8000, debug=True)
     app.run()