import tensorflow as tf
import numpy as np
import models
import evals
import os
import pathlib
import data
import keras
import q_evals

path = 'checkpoints/8_10_20H/r50serelu_emore.h5'
save_path = '/'.join(path.split('/')[:-1]) + '/q_model.tflite'
train_path = '../Datasets/faces_emore_112x112_folders'


# model = tf.keras.models.load_model(path, compile=False, custom_objects={"NormDense": models.NormDense})

if not os.path.exists(save_path):

    model = tf.keras.models.load_model(path, compile=False,)

    layer = model.layers[:-1]
    model = keras.Model(model.inputs,layer[-1].output)
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        
        ds, steps_per_epoch = data.prepare_dataset(train_path)
        x = None

        for d in ds:
            x = d
            break

        yield [np.array(x[0], dtype=np.float32)]

    converter.representative_dataset = representative_data_gen

    converter.target_spec.supported_pos = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

#    print(input_details[0]['dtype'])
#    print(output_details[0]['dtype'])

    saves_path = pathlib.Path(save_path)
    saves_path.write_bytes(tflite_model)

else:
    interpreter = tf.lite.Interpreter(model_path=save_path)

#interpreter.allocate_tensors()

model = tf.keras.models.load_model(path, compile=False,)

layer = model.layers[:-1]
model = keras.Model(model.inputs,layer[-1].output)
model.save('/'.join(path.split('/')[:-1]) + '/cutting.h5')


ee = q_evals.eval_callback(interpreter, '../Datasets/faces_emore/lfw.bin')
ee.on_epoch_end(0)
