from tensorflow import keras
import losses, train, models
import tensorflow_addons as tfa
import os
import tensorflow as tf

with tf.distribute.MirroredStrategy().scope():
# basic_model = models.buildin_models("ResNet101V2", dropout=0.4, emb_shape=512, output_layer="E")
    basic_model = models.buildin_models("r50", dropout=0.4, emb_shape=512, output_layer="E", use_se=True, activation='relu')
    data_path = '../Datasets/faces_emore_112x112_folders'
    eval_paths = ['../Datasets/faces_emore/lfw.bin', '../Datasets/faces_emore/cfp_fp.bin', '../Datasets/faces_emore/agedb_30.bin']
    save_path = 'r50serelu_emore.h5'
    tt = train.Train(data_path, save_path=save_path, eval_paths=eval_paths,
                    basic_model=basic_model, batch_size=128, random_status=0,
                    lr_base=0.1, lr_decay=0.1, lr_decay_steps=[4, 7,], lr_min=1e-5)
    optimizer = tfa.optimizers.SGDW(learning_rate=0.1, momentum=0.9, weight_decay=5e-4)
#    optimizer = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=5e-4)
    sch = [
        {"loss": losses.ArcfaceLoss(), "epoch": 8, "optimizer": optimizer},
#      {"loss": losses.ArcfaceLoss(scale=64), "epoch": 10, "optimizer": optimizer},
  #{"loss": losses.ArcfaceLoss(scale=32), "epoch": 5},
  #{"loss": losses.ArcfaceLoss(scale=64), "epoch": 40},
  # {"loss": losses.ArcfaceLoss(), "epoch": 20, "triplet": 64, "alpha": 0.35},
    ]
    tt.train(sch, 0)
