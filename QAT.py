import tempfile
import os
import q_models
from backbones import q_resnet

import tensorflow as tf
from tensorflow import keras
import q_evals

from tensorflow import keras
import losses, train, models
import tensorflow_addons as tfa
import os
import tensorflow as tf


#path = 'checkpoints/8_14_21H2_r50se/test_emore.h5'
#path = 'checkpoints/8_14_21H_r50prelu/r50serelu_emore.h5'
#path = 'checkpoints/8_10_20H_origin/r50serelu_emore.h5'
#path = 'checkpoints/8_10_20H2_resnet/test_emore.h5'
#path = 'checkpoints/8_30_15H/r50_emore.h5'
#save_path = '/'.join(path.split('/')[:-1]) + '/q_model.tflite'
#train_path = '../Datasets/faces_emore_112x112_folders'
#
#
#model = tf.keras.models.load_model(path, compile=False, custom_objects={"NormDense": q_models.NormDense})
#
#keras.utils.plot_model(model, to_file='b.png', show_shapes=True)

# r50 

with tf.distribute.MirroredStrategy().scope():
# basic_model = models.buildin_models("ResNet101V2", dropout=0.4, emb_shape=512, output_layer="E")
    basic_model = q_models.buildin_models("r50", dropout=0.4, emb_shape=512, output_layer="E",  activation='relu', use_se=True)
    data_path = '../Datasets/faces_emore_112x112_folders'
    eval_paths = ['../Datasets/faces_emore/lfw.bin', '../Datasets/faces_emore/cfp_fp.bin', '../Datasets/faces_emore/agedb_30.bin']
    save_path = 'q_r50serelu_emore.h5'
    tt = train.Train(data_path, save_path=save_path, eval_paths=eval_paths,
                    basic_model=basic_model, batch_size=128, random_status=0,
                    lr_base=0.1, lr_decay=0.1, lr_decay_steps=[9, 15,], lr_min=1e-5)
    optimizer = tfa.optimizers.SGDW(learning_rate=0.1, momentum=0.9, weight_decay=5e-4)
#    optimizer = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=5e-4)
    sch = [
        {"loss": losses.ArcfaceLoss(), "epoch": 1, "optimizer": optimizer},
#      {"loss": losses.ArcfaceLoss(scale=64), "epoch": 10, "optimizer": optimizer},
  #{"loss": losses.ArcfaceLoss(scale=32), "epoch": 5},
  #{"loss": losses.ArcfaceLoss(scale=64), "epoch": 40},
  # {"loss": losses.ArcfaceLoss(), "epoch": 20, "triplet": 64, "alpha": 0.35},
    ]
    tt.train(sch, 0)



exit()





model = q_models.buildin_models("r50", dropout=0.4, emb_shape=512, output_layer="E")


model.summary()


import tensorflow_model_optimization as tfmot

quantize_scope = tfmot.quantization.keras.quantize_scope
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model

model = quantize_annotate_model(model)

with quantize_scope(
    {'DefaultDenseQuantizeConfig': q_resnet.DefaultDenseQuantizeConfig,
     'batchnorm_with_activation': q_resnet.batchnorm_with_activation,
     'DefaultBNQuantizeConfig': q_resnet.DefaultBNQuantizeConfig}):
    q_aware_model = tfmot.quantization.keras.quantize_apply(model)

q_aware_model.summary()


converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

qat_tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=qat_tflite_model)

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(input_details)
print(output_details)

ee = q_evals.eval_callback(interpreter, '../Datasets/faces_emore/lfw.bin')
ee.on_epoch_end(0)