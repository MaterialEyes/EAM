h, w = 512, 512
start = 1
end = 7
num_train = end-start
aug_per_img = 4
N=1
num_instances_e2 = 5

lr_per_replica = 0.1
batch_size_per_replica = 8
epochs = 20
verbose = 1
val_split = 0.1
arch = "unet"
single_backbone_for_training = 'inceptionv3'
backbones_for_inference = ['densenet201', 'densenet169', 'densenet121']
new_training = False

available_backbones = ['vgg16','vgg19','resnet18','resnet34','resnet50','resnet101','resnet152','seresnet18','seresnet34','seresnet50', 'seresnet101','seresnet152','resnext50','resnext101','seresnext50','seresnext101','senet154','densenet121','densenet169', 'densenet201','inceptionv3','inceptionresnetv2','mobilenet','mobilenetv2','efficientnetb0','efficientnetb1','efficientnetb2','efficientnetb3' 'efficientnetb4','efficientnetb5','efficientnetb6','efficientnetb7']

root = "../" 
training_dir = root + "data/training/"
inference_dir = root + "data/inference/" 
weights_dir = root + "trained_weights/"
metadata_dir = root
