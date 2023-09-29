import tensorflow as tf
import numpy as np
import cv2
from flask import request, jsonify, Flask
from werkzeug.utils import secure_filename
import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from flask_bcrypt import Bcrypt
import os


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.environ.get("DB_URI", f"sqlite:///{os.path.join(BASE_DIR, 'app.db')}")


def fk_connect(conn, conn_record):
    conn.execute("PRAGMA foreign_keys='ON'")


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = "True"
# Define the directory where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure the uploaded files settings
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}

flask_bcrypt = Bcrypt(app)
CORS(app, supports_credentials=True)
# app.json.compact = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

SESSION_TYPE = "sqlalchemy"


# Load the TensorFlow Lite model during app initialization
interpreter = tf.lite.Interpreter(model_path="best_model.tflite")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# # input details
# print(input_details)
# # output details
# print(output_details)

@app.route("/")
def home():
    return "The One Piece is real"

@app.route("/classify", methods=["POST"])
def classify_image():
    image = request.files['image']
    print(f"Image:{image}")
    # image = dictData
    # print(image.keys['image'])


    if image:
        filename = secure_filename(image.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)
        

        # Load and preprocess the image
        img = cv2.imread(filepath)
        new_img = cv2.resize(img, (224, 224))  # Resize to match the model input size
        new_img = new_img / 255.0  # Normalize the image 
        new_img = np.moveaxis(new_img, -1, 0) # make the channel the first dimension


        # Prepare input tensor
        input_shape = input_details[0]['shape']
        # input_tensor = tf.convert_to_tensor(np.expand_dims(new_img, 0), dtype=tf.float32)
        input_data = np.array([new_img], dtype=np.float32)
        # print(input_data)
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        print(input_tensor)
 
        interpreter.resize_tensor_input(input_details[0]['index'], input_shape, True)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], input_tensor)


        # Perform inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Process the output and find the predicted label
        predicted_label = np.argmax(output_data)
        highest_confidence = output_data[0][predicted_label]
        flowerName = ""
        all_labels = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen', 'watercress', 'canna lily', 'hippeastrum', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']

        if 0 <= predicted_label < len(all_labels):
            mapped_label = all_labels[predicted_label]
        

        return jsonify({
            "predicted_label": int(predicted_label),
            "confidence_score": float(highest_confidence),
            "flowerName": mapped_label
        })

# host='0.0.0.0'
if __name__ == "__main__":
    app.run(port=8000, debug=True)

