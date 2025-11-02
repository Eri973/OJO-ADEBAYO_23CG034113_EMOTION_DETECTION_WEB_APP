from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)

model = load_model('emotion_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/', methods=['GET', 'POST'])
def index():
    emotion = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_path = os.path.join('static', file.filename)
            file.save(image_path)

            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi = roi_gray.astype('float')/255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)

                preds = model.predict(roi)[0]
                emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                emotion = emotion_labels[np.argmax(preds)]

            return render_template('index.html', emotion=emotion, image_path=image_path)
    return render_template('index.html', emotion=emotion)

if __name__ == '__main__':
    app.run(debug=True)
