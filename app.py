from flask import Flask, render_template, Response
import cv2
import torch
import numpy as np

import time


#Initialize the Flask app
app = Flask(__name__)


model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

camera = cv2.VideoCapture(0)

def gen_frames():  
    counter = 0
    text = "Recording and looking for a bottle to save a picture"
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:

            results = model(frame)

            # print(type(results.crop())) # results.crop() saves the whole image as well as the crops of the objects in the image

            # print(type(results)) # results is of type <class 'models.common.Detections'>

            #time.sleep(1)

            if results.pandas().xyxy[0]['name'].str.contains('bottle').any():
                counter += 1
                print("bottle is here", counter)
                if counter >= 40:
                    print("the bottle cropped is saved")
                    # We could make many more pictures, and than do a majority vote for the lassification task
                    results.crop() 
                    counter = 0
            else:
                counter = 0

            ret, buffer = cv2.imencode('.jpg', np.squeeze(results.render()))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)