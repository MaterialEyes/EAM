import imageio
import numpy as np
import cv2
from global_defs import *

try:
    assert os.path.exists(video_filename)
except:
    print("Video file path does not exist")

vid = imageio.get_reader(filename, 'ffmpeg')
metadata = vid._meta
try:
    assert ((type(metadata['fps']) == int) or (type(metadata['fps']) == float))
except:
    print("The fps field in the metadata is not of an appropriate datatype.")
try:
    assert ((type(metadata['duration']) == int) or (type(metadata['duration']) == float))
except:
    print("The duration field in metadata is not of an appropriate datatype.")

if str(metadata['nframes']) == 'inf':
    num_frames = int(metadata['fps'])*int(metadata['duration'])
else:
    num_frames = metadata['nframes']

try:
    assert ((type(num_frames) == int) or (type(num_frames) == float))
except:
    print("The num_frames argument is not a numerical value")

imgshape = metadata['source_size']
try:
    assert ((type(imgshape[0]) == int) and (type(imgshape[1]) == int))
except:
    print("Image shape is not in integer")

num_frames_per_stack = 25
num_stacks = num_frames//num_frames_per_stack
try:
    assert num_stacks >= 1
except:
    print("Number of stacks does not come out to be greater than 1")

for s in range(num_stacks):
    temp_array = np.zeros((imgshape[1], imgshape[0]))
    for n in range(s*num_frames_per_stack, (s+1)*num_frames_per_stack):
        image = vid.get_data(n)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        temp_array = np.add(temp_array, np.array(image))
    temp_array = temp_array/num_frames_per_stack
    cv2.imwrite(root + f"data/stack_{s+1}.png", temp_array)