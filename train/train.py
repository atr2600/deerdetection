from imageai.Detection.Custom import DetectionModelTrainer
#from tensorflow import tf.compat.v1.assign_add as tf.assign_add

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="Deer")
trainer.setTrainConfig(object_names_array=["Deer"], batch_size=4, num_experiments=100, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()
