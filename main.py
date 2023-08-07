from tqdm import tqdm
from aioconsole import ainput
import cv2, time
import pync
import mediapipe as mp
import numpy as np
from datetime import datetime
from utils import *
import math as m
import pygame.camera

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)
font = cv2.FONT_HERSHEY_SIMPLEX

# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def sendWarning(x):
    pync.notify(x, sound='ping')

# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

pose = mp_pose.Pose()

def main():
  good_frames = 0
  bad_frames = 0
  global init
  
  # open HD Web cam – not facetime cam
  pygame.camera.init()
  cams = pygame.camera.list_cameras()
  cam_index = [i for i, s in enumerate(cams) if 'USB Cam' in s][0] # TODO: modify based on what the cam idx names are; you don't want the facetime one
  cap = cv2.VideoCapture(cam_index)

  if args.debug:
    # Loop until the camera is closed or 'q' is pressed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything done, release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
    return

  pose = mp_pose.Pose()
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # get high level image reading & pose stuff (need to convert to and from RGB for pose stuff)
    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        # Get landmarks
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        ## Calculate alignment
        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        if offset < args.alignment_offset:
            cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
        else:
            cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

        ## Calculate posture
        # Calculate angles
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))
        in_good_posture = neck_inclination < args.neck_inclination and torso_inclination < args.torso_inclination

        if in_good_posture: 
            bad_frames = 0
            good_frames += 1
            
            indicator_color = light_green
            landmark_color = green

        else: # BAD
            good_frames = 0
            bad_frames += 1
            indicator_color = red
            landmark_color = red
        
        # Write all information to screen
        # add indicator text
        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, indicator_color, 2)
        cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, indicator_color, 2)
        cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, indicator_color, 2)

        # Draw landmarks.
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)

        # Let's take y - coordinate of P3 100px above x1,  for display elegance.
        cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
        cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

        # Join landmarks.
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), landmark_color, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), landmark_color, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), landmark_color, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), landmark_color, 4)

        # Calculate the time of remaining in a particular posture.
        good_time = (1 / fps) * good_frames
        bad_time =  (1 / fps) * bad_frames

        # Pose time.
        if good_time > 0:
            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
            cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
        else:
            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
            cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

        # If you stay in bad posture for more than args.threshold seconds send an alert.
        if bad_time > args.threshold:
            sendWarning("out of alignment")
    except:
        cv2.putText(image, 'ERROR: Cannot identify keypoints!', (w - 150, 30), font, 0.9, green, 2)
    
    # Display.
    # resize image
    height, width = image.shape[:2]
    # new_dimensions = (int(height*args.resize), int(width*args.resize))
    new_dimensions =  (int(width*args.resize), int(height*args.resize))
    resized_image = cv2.resize(image, new_dimensions)

    cv2.imshow('Posture', resized_image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
  cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=int, default=10, help='Number of seconds in bad posture before notification')
    parser.add_argument('--debug', type=int, default=0, help='If 1, will display the camera feed and exit.  Useful for debugging camera issues.')

    # thresholds for alignment
    parser.add_argument('--alignment_offset', type=int, default=30, help='Maximum distance between left and right shoulders to qualify for "alignment" – "can the camera see you ok"') #TODO: 100?

    # thresholds for good and bad posture
    parser.add_argument('--neck_inclination', type=int, default=30, help='Maximum angle of neck inclination (degrees) before notification')
    parser.add_argument('--torso_inclination', type=int, default=10, help='Maximum angle of torso inclination (degrees) before notification')

    # how big should the window be? width and height
    parser.add_argument('--resize', type=float, default=.5, help='Size of the window to display the camera feed.  1.0 is full size, 0.5 is half size, etc.')

    args = parser.parse_args()

    main()