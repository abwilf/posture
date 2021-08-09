import asyncio
from tqdm import tqdm
from aioconsole import ainput
import cv2, time, pync
import mediapipe as mp
from datetime import datetime
from utils import *

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


# TODO: time tolerance (in seconds)
# confidence threshold
time_threshold = 5
time_print_threshold = 2 # time in state before starts printing msgs to console
min_visibility_threshold = .7
capture_frequency = .01
show_image = True

paused = True

# angles format:
# (pt1, pt2, pt3, optimal angle, angle tolerance)

angles = {
  'left_elbow': ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST', 90, 10],
  'right_elbow': ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST', 90, 10],
}

class Logger():
  def __init__(self):
    self.last_msg = ''
    self.last_notification = ''
  def print(self, msg):
    if msg == self.last_msg:
      return
    else:
      self.last_msg = msg
      print(msg)
  def notify(self, msg):
    if msg == self.last_notification:
      return
    else:
      self.last_notification = msg
      pync.notify(msg, sound='ping')

class TestMonitor():
  def __init__(self, test_name, time_threshold, min_visibility_threshold):
    self.test_failed = ''
    self.optimal_angle = None
    self.current_angle = None
    self.time_in_fail = 0
    self.started_failing = None
    self.time_threshold = time_threshold
    self.min_visibility_threshold = min_visibility_threshold
    self.test_name = test_name
    self.logger = Logger()

  def watch(self, angle, opt_angle, angle_tolerance, min_visibility):
      lower_bound, upper_bound, within_range = interpret_angle(angle, opt_angle, angle_tolerance)
      
      # failing
      if not within_range and min_visibility > self.min_visibility_threshold and not paused:
        
        if self.started_failing is None:
          self.started_failing = datetime.utcnow() 
          self.time_in_fail = 0
        
        else:
          self.time_in_fail = (datetime.utcnow() - self.started_failing).total_seconds()
        
        if self.time_in_fail > time_print_threshold:
          self.logger.print(f'{self.test_name} is out of alignment!')
          if self.time_in_fail > self.time_threshold:
            self.logger.notify(f'{self.test_name} is out of alignment!')
      
      else:
        was_not_all_good = max([monitor.time_in_fail for monitor in monitors.values()]) != 0
        self.started_failing = None
        self.time_in_fail = 0
        
        all_good_now = max([monitor.time_in_fail for monitor in monitors.values()]) == 0

        if all_good_now and was_not_all_good:
          self.logger.print('Good posture!')

        self.logger.last_msg = ''
        self.logger.last_notification = ''
  
  def reset(self):
    self.started_failing = None

monitors = {k: TestMonitor(k,time_threshold,min_visibility_threshold) for k in angles.keys()}

def interpret_angle(angle, opt_angle, angle_tolerance):
  lower_bound = opt_angle-angle_tolerance
  upper_bound = opt_angle+angle_tolerance
  within_range = lower_bound <= angle <= upper_bound
  return lower_bound, upper_bound, within_range

async def input_loop():
  global paused
  while True:
    line = await ainput('')
    if line == ' ':
      paused = flip_bool(paused)
      print('Monitors ' + ('paused' if paused else 'unpaused'))

async def main_loop():
  cap = cv2.VideoCapture(1)
  with mp_pose.Pose( min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
      await asyncio.sleep(capture_frequency)
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        continue

      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False
      image_height, image_width, image_z = image.shape

      results = pose.process(image)

      if results.pose_landmarks is None:
        if not paused:
          print('No landmarks detected, resetting all monitors')
        [monitor.reset() for monitor in monitors.values()]

      else:
          landmark = results.pose_landmarks.landmark

          for test_name,v in angles.items():
            monitor = monitors[test_name]

            name1, name2, name3, opt_angle, angle_tolerance = v
            
            coords1 = landmark[getattr(mp_holistic.PoseLandmark, name1)]
            x1, v1 = np.array([coords1.x, coords1.y, coords1.z]), coords1.visibility

            coords2 = landmark[getattr(mp_holistic.PoseLandmark, name2)]
            x2, v2 = np.array([coords2.x, coords2.y, coords2.z]), coords2.visibility

            coords3 = landmark[getattr(mp_holistic.PoseLandmark, name3)]
            x3, v3 = np.array([coords3.x, coords3.y, coords3.z]), coords3.visibility

            x1x2 = x1-x2
            x2x3 = x3-x2
            angle = np.degrees(np.arccos(np.dot(x1x2,x2x3) / (np.linalg.norm(x1x2)*np.linalg.norm(x2x3))))
            min_visibility = np.mean([v1,v2,v3])

            monitor.watch(angle, opt_angle, angle_tolerance, min_visibility)

      # Draw the pose annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      mp_drawing.draw_landmarks(
          image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
      
      image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
      cv2.imshow('MediaPipe Pose', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()

async def main():
  main_task = asyncio.create_task(main_loop())
  input_task = asyncio.create_task(input_loop())

  await input_task
  await main_task

asyncio.run(main())
