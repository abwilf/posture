# posture

![alt text](https://github.com/abwilf/posture/blob/main/calibration.png)

## Approach
We all know roughly what good posture feels like, but it can be difficult to keep that in mind when we're working. **posture** uses an external webcam and an [open source computer vision library](https://google.github.io/mediapipe/) to track the angles between joints in the human body and notify the user when those angles stray too far from what they should be for too long.  The joints the user wishes to monitor, the optimal angle for that joint, the tolerance of how far off the angle can be before notifying, and the tolerance of how long the user can be in a "bad" position are all configurable through a simple json file.

## Requirements
- `python 3.7+`
- Packages in `requirements.txt`.  I would recommend creating a `conda` environment:
```
conda create -n posture python=3.7 -y
conda activate posture
pip install -r requirements.txt
```
- (recommended) An external webcam.  You may be able to do it from your laptop, but I've found that the angle is much more accurate from the side. [This one](https://www.amazon.com/gp/product/B082X91MPP/ref=ppx_yo_dt_b_asin_title_o04_s00?ie=UTF8&psc=1) has worked great for me.
- (recommended) A `mac`, for the notification support.  If you do not have a mac, comment out all lines involving `pync` in `main.py` and uncomment the `os.system` call to give yourself an audio notification instead.

## Usage
1. Start the program (for advanced usage, see the arguments subsection below).
```
python3 main.py
```
2. Calibrate: position the camera and move your body around until the program can clearly identify the relevant joints. You may need to roll up your sleeves.

3. "Unpause" the program by typing space + enter into the terminal once you've finished calibrating, and **posture** will begin monitoring the relevant angles.

4. End the program with ctrl+c as usual.

The program is currently set to monitor the right and left elbows to be close to 90 degrees.  If you would like to change this or add joints to monitor, simply do so in `angles.json`.  A description of how to do so is in `angles_path` below. 

### Arguments
Below is the output of`python3 main.py --help`

```
usage: main.py [-h] [--time_threshold TIME_THRESHOLD]
               [--cam_number CAM_NUMBER]
               [--time_print_threshold TIME_PRINT_THRESHOLD]
               [--angles_path ANGLES_PATH]
               [--min_visibility_threshold MIN_VISIBILITY_THRESHOLD]
               [--capture_frequency CAPTURE_FREQUENCY]

optional arguments:
  -h, --help            show this help message and exit
  --time_threshold TIME_THRESHOLD
                        Number of seconds in bad posture before notification
  --cam_number CAM_NUMBER
                        The index of the camera attached to this system. 0
                        indicates automatically selected, but does not always
                        work with external webcams. Try 1,2...etc.
  --time_print_threshold TIME_PRINT_THRESHOLD
                        Number of seconds in bad posture before program starts
                        to print messages to console
  --angles_path ANGLES_PATH
                        Path to where the angles json is stored, containing
                        all angles you would like the program to monitor. The
                        format is a dictionary with name of the angle as the
                        key mapping to an array of [first joint, middle joint,
                        last joint, optimal angle, angle tolerance before
                        notification] e.g.: "left_elbow": ["LEFT_SHOULDER",
                        "LEFT_ELBOW", "LEFT_WRIST", 90, 10] means that I named
                        this angle "left_elbow" (arbitrary), and it consists
                        of the angle between "LEFT_SHOULDER", "LEFT_ELBOW",
                        and "LEFT_WRIST" (not arbitrary; these names must
                        align with those in the mediapipe api detailed in fig
                        4 at https://google.github.io/mediapipe/solutions/pose
                        .html). The optimal angle is 90 degrees, and I'm
                        allowing for being 10 degrees off in either direction
                        before notification.
  --min_visibility_threshold MIN_VISIBILITY_THRESHOLD
                        Threshold of how confident model must be in the
                        visibility of the least visible joint in an angle
                        triad
  --capture_frequency CAPTURE_FREQUENCY
                        Number of seconds between capturing frames for
                        processing. The smaller this number, the smoother the
                        video, but the more processing power required.
```

## Future Work
- Compatibility with windows notifications
- Support for multiple simultaneous camera streams: multiple cameras may be able to better capture groups of angles that a single camera would have difficulty with, for example the elbow and the neck