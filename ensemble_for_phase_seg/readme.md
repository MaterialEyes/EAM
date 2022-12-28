## Instructions

### Frame stacking
* Set the 'video_filename' variable to the file path for the video.
* Set the 'num_frames_per_stack' to the number of image frames required to form one stack.
* Run image_extracter.py 

### Single model training
* Assemble the image stacks for training in the training directory. 
* Set the 'new_training_ option on global_defs.py to 'True'.
* If the backbone is not meant to be a run-time input command, define the name of the backbone in the 'single_backbone_for_training' option.
* If the backbone is meant to be a run-time command, uncomment the lines on the top of the file to assert the validity of the input.
* Set the 'start' and 'end' variables in global_defs.py to the initial and final indices of the image stacks that will be used for training.


### Ensemble training
* Assemble the image stacks for training in the training directory. 
* Set the 'new_training_ option on global_defs.py to 'True'.
* Define the list of backbones that forms the part of the ensemble in the 'backbones_for_training' option in global_defs.py
* Set the 'start' and 'end' variables in global_defs.py to the initial and final indices of the image stacks that will be used for training.

### Ensemble re-training
* Assemble the image stacks for re-training in the training directory. 
* Set the 'new_training_ option on global_defs.py to 'False'. This will make sure that the pre-trained weights are loaded for each base member in the ensemble during the model definition step. 
* Verify the list of backbones to retrain in the 'backbones_for_training' option in global_defs.py
* Set the 'start' and 'end' variables in global_defs.py to the initial and final indices of the image stacks that will be used for re-training.


### Extracting prediction mean and standard deviation

### Ensemble inference
