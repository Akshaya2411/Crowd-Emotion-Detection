from flask import Flask, render_template, Response
import cv2
import os
from deepface import DeepFace

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
app = Flask(__name__)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:  
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                crop_image = frame[y:y+h, x:x+w]
                results = DeepFace.analyze(crop_image, actions=['emotion'], enforce_detection=False)
                for result in results:
                    cv2.putText(frame, result['dominant_emotion'], (x, y), font, 1, (0, 0, 255), 2, cv2.LINE_4)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('app.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(port="0.0.0.0", debug=True)
