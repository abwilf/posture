# posture

![alt text](https://github.com/abwilf/posture/blob/main/good_posture.png)

## Approach
We all know roughly what good posture feels like, but it can be difficult to keep that in mind when we're working. **posture** uses an external webcam and an [open source computer vision library](https://google.github.io/mediapipe/) to track the angles between joints in the human body and notify the user when those angles stray too far from what they should be for too long.  The joints the user wishes to monitor, the optimal angle for that joint, the tolerance of how far off the angle can be before notifying, and the tolerance of how long the user can be in a "bad" position are all configurable through a simple json file. Previous commits in this repo focused on my elbow angles, but I've refactored it substantially to be focused on neck and torso posture, drawing heavily on this [repo](https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/).

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
```
python3 main.py
```

Make sure you're "aligned" – the camera should be roughly parallel with the line between your two shoulders, so it's seeing your profile.

### Arguments
Below is the output of`python3 main.py --help`

```
usage: main.py [-h] [--threshold THRESHOLD] [--debug DEBUG]
               [--alignment_offset ALIGNMENT_OFFSET]
               [--neck_inclination NECK_INCLINATION]
               [--torso_inclination TORSO_INCLINATION]

optional arguments:
  -h, --help            show this help message and exit
  --threshold THRESHOLD
                        Number of seconds in bad posture before notification
  --debug DEBUG         If 1, will display the camera feed and exit. Useful
                        for debugging camera issues.
  --alignment_offset ALIGNMENT_OFFSET
                        Maximum distance between left and right shoulders to
                        qualify for "alignment" – "can the camera see you ok"
  --neck_inclination NECK_INCLINATION
                        Maximum angle of neck inclination (degrees) before
                        notification
  --torso_inclination TORSO_INCLINATION
                        Maximum angle of torso inclination (degrees) before
                        notification
```

## Future Work
- Compatibility with windows notifications
- Support for multiple simultaneous camera streams: multiple cameras may be able to better capture groups of angles that a single camera would have difficulty with, for example the elbow and the neck