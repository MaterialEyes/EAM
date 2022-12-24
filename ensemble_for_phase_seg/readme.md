## Instructions

### Frame stacking

### Single model training
* Assemble the image stacks for training in the training sub-repository. 
* Set the 'new_training_ option on global_defs.py to 'True'.
* If the backbone is not meant to be a run-time input command, define the name of the backbone in the 'single_backbone_for_training' option.
* If the backbone is meant to be a run-time command, uncomment the lines on the top of the file to assert the validity of the input.
* Set the 'start' and 'end' variables in global_defs.py to the initial and final indices of the image stacks that will be used for training.


### Ensemble training
* Assemble the image stacks for training in the training sub-repository. 
* Set the 'new_training_ option on global_defs.py to 'True'.
* Define the list of backbones that forms the part of the ensemble in the 'backbones_for_training' option in global_defs.py
* Set the 'start' and 'end' variables in global_defs.py to the initial and final indices of the image stacks that will be used for training.

### Ensemble re-training

### Extracting prediction mean and standard deviation

### Ensemble inference
