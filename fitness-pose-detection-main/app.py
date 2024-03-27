from flask import Flask,render_template,Response
import mediapipe as mp
import cv2
import numpy as np
from angle import *

import json
global curls
f = open("assets/exercise.json", "r")  # Open the file in read
LOAD = json.load(f)
curls = dict(LOAD["curls"])
print(curls)

mp_pose = mp.solutions.pose

app=Flask(__name__)
camera=cv2.VideoCapture(0)
video = cv2.VideoCapture("assets\11.mp4")

mp_drawings = mp.solutions.drawing_utils

global ind
ind = 0
print(curls.get(str(ind))[-1] +  ' is your next exercise')
print(float(curls.get(str(ind))[0]))

guide_image = cv2.imread(curls.get(str(ind))[-1])

def compare_angles(angles1, angles2, angle_threshold=5):
    """
    Compare two arrays of corresponding pose angles.

    Parameters:
    - angles1: List or NumPy array containing the first set of pose angles.
    - angles2: List or NumPy array containing the second set of pose angles.
    - angle_threshold: Threshold for considering an angle match (default is 5 degrees).

    Returns:
    - accuracy: Percentage accuracy based on the number of matching angles.
    """
    if len(angles1) != len(angles2):
        raise ValueError("Input arrays must have the same length.")

    num_angles = len(angles1)
    correct_matches = sum(abs(angle1 - angle2) <= angle_threshold for angle1, angle2 in zip(angles1, angles2))

    accuracy = (correct_matches / num_angles) * 100
    return accuracy

def compare_angle(angle):
    global ind

    f = open("assets/exercise.json", "r")  # Open the file in read
    LOAD = json.load(f)
    curls = dict(LOAD["curls"])
    result = 0

    acc = angle
    pro = [
        float(curls.get(str(ind))[0]),
        float(curls.get(str(ind))[1]),
        float(curls.get(str(ind))[2]),
        float(curls.get(str(ind))[3])]

    result = compare_angles(acc,pro)

    if result <70:
        ind = 0
        # print("Try Again")
        return result
    else: 
        ind += 1
        return result

def generate_frames():
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
                
            ## read the camera frame
            success,frame=camera.read()
            if not success:
                break
            else:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
                landmarks = results.pose_landmarks.landmark
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Render detections
                mp_drawings.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawings.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawings.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )

                # body_parts = {part: [landmarks[getattr(mp_pose.PoseLandmark, f'LEFT_{part}').value].x,
                #              landmarks[getattr(mp_pose.PoseLandmark, f'LEFT_{part}').value].y]
                #       for part in body_part_names}
                
                # for part_name in i:
                #     part_angle = getAngles(body_parts[f'LEFT_{part_name[0]}'], body_parts[f'LEFT_{part_name[1]}'], body_parts[f'LEFT_{part_name[2]}'])
                #     print(part_angle)
                #     cv2.putText(image, f"{part_name.capitalize()}: {round(part_angle, 2)}", (50, 50 + 30 * i.index(part_name)),
                #         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                ANGLE = []
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    # Example: Get the angles between shoulder-elbow and elbow-wrist
                angle_shoulder_elbow = getAngles(shoulder, elbow, wrist)
                angle_elbow_hip = getAngles(elbow, shoulder, hip)
                angle_shoulder_knee = getAngles(shoulder, hip, knee)
                angle_hip_ankle = getAngles(hip, knee, ankle)

                ANGLE = [angle_shoulder_elbow, angle_elbow_hip,angle_shoulder_knee,angle_hip_ankle]
                Accuracy = compare_angle(ANGLE)
                print(Accuracy)

            # Display the resulting image on the
                ret,buffer=cv2.imencode('.jpg',image)
                image=buffer.tobytes()

            yield(
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')



@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/gui')
def gui():
    return render_template('gui.html')


@app.route('/faq')
def faq():
    return render_template('faq.html')


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_image_frames():

    ret,buffer2=cv2.imencode('.jpg',guide_image)
    image_bytes=buffer2.tobytes()
    yield(
        b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n\r\n'
    )

@app.route('/video2')
def video2():
    return Response(generate_image_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(debug=True)