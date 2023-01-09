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
import evals
import data
import models

import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt

from weight import weight_check



def trains():
    with tf.distribute.MirroredStrategy().scope():
    # basic_model = models.buildin_models("ResNet101V2", dropout=0.4, emb_shape=512, output_layer="E")
        basic_model = models.buildin_models("r18", dropout=0.4, emb_shape=512, output_layer="E",  activation='relu')
        data_path = '/mnt/hdd0/hjyoo/Datasets/faces_emore_112x112_folders'
        eval_paths = ['/mnt/hdd0/hjyoo/Datasets/faces_emore/lfw.bin', '/mnt/hdd0/hjyoo/Datasets/faces_emore/cfp_fp.bin', '/mnt/hdd0/hjyoo/Datasets/faces_emore/agedb_30.bin']
#        data_path = '../Datasets/faces_emore_112x112_folders'
#        eval_paths = ['../Datasets/faces_emore/lfw.bin', '../Datasets/faces_emore/cfp_fp.bin', '../Datasets/faces_emore/agedb_30.bin']
        save_path = 'q_r18relu_emore.h5'
        tt = train.Train(data_path, save_path=save_path, eval_paths=eval_paths,
                        basic_model=basic_model, batch_size=256, random_status=0,
                        lr_base=0.1, lr_decay=0.1, lr_decay_steps=[9, 15,], lr_min=1e-5, )#quantization=True)
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

def QAT():

    #default_model = models.buildin_models("r18", dropout=0.4, emb_shape=512, output_layer="E",  activation='relu')
    path = 'checkpoints/9_27_0H/q_r18relu_emore_basic_model_latest.h5'
    #save_path = '/'.join(path.split('/')[:-1]) + '/q_model.tflite'
    #train_path = '../Datasets/faces_emore_112x112_folders'
    

    
    
    quantize_scope = tfmot.quantization.keras.quantize_scope
    quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
    
    #print('init model')
    #print(model.layers[2].weights)
    
    with quantize_scope(
        {'DefaultDenseQuantizeConfig': q_resnet.DefaultDenseQuantizeConfig,
         'batchnorm_with_activation': q_resnet.batchnorm_with_activation,
         'DefaultBNQuantizeConfig': q_resnet.DefaultBNQuantizeConfig}):
         
        original_model = tf.keras.models.load_model(path, compile=False, )
        
        #layer = model.layers[:-1] 
        #model = keras.Model(model.inputs,layer[-1].output)
        #print(layer)
        
#        print('----------------------original model---------------------------------')
#        ee = evals.eval_callback(original_model, '/mnt/hdd0/hjyoo/Datasets/faces_emore/lfw.bin')
#        ee.on_epoch_end(0)
#        
        #del val_model

        original_model.summary()
        
        
        keras.utils.plot_model(original_model, to_file='original.png')
#        for l in model.layers:
#            print(l)
        
        q_model = quantize_annotate_model(original_model)
        
        
        from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit  import default_n_bit_quantize_scheme
        
        scheme = default_n_bit_quantize_scheme.DefaultNBitQuantizeScheme(num_bits_weight=8, num_bits_activation=8)
        
        q_aware_model = tfmot.quantization.keras.quantize_apply(q_model, scheme=scheme)
    
    q_aware_model.summary()
    
    keras.utils.plot_model(q_aware_model, to_file='QAT.png')
    
    
    print('original weight')
    print(len(original_model.weights))
    print(original_model.weights[0:5])
    
    print('QAT weight')
    print(len(q_aware_model.weights))
    print(q_aware_model.weights[:10])

    
    
#    print('-----------------------qat_model_before_finetuning---------------------')
#    ee = evals.eval_callback(q_aware_model, '/mnt/hdd0/hjyoo/Datasets/faces_emore/lfw.bin')
#    ee.on_epoch_end(0)
    arc_kwargs = {'loss_top_k': 1, 'append_norm': False, 'partial_fc_split': 0, 'name': 'arcface'}

    arcface_logits = models.NormDense(85742, None, **arc_kwargs, dtype="float32")
    
    inputs = q_aware_model.inputs[0]
    embedding = q_aware_model.outputs[0]
    
    arcface_logits.build(embedding.shape)
    output_fp32 = arcface_logits(embedding)
    
    q_aware_model = keras.models.Model(inputs, output_fp32)
    
    # fine-tuning
    data_path = '/mnt/hdd0/hjyoo/Datasets/faces_emore_112x112_folders'
    train_ds, steps_per_epoch = data.prepare_dataset(data_path, batch_size=100, fine_tuning=True)
    print('train_ds', train_ds)
    tt = train_ds.take(10)
    print('tt', tt)
    
    # compiler
    q_aware_model.compile(optimizer = tfa.optimizers.SGDW(learning_rate=0.1, momentum=0.9, weight_decay=5e-4),
    loss = [losses.ArcfaceLoss()],
    metrics = {'arcface': 'accuracy'},
    loss_weights= {'arcface': 1}) 
    
    # model.fit
    q_aware_model.fit(tt, epochs=1,)
                    
    
    # q_aware_model
    print('-----------------------QAT model after Finetuning---------------------')
    ee = evals.eval_callback(q_aware_model, '/mnt/hdd0/hjyoo/Datasets/faces_emore/lfw.bin')
    ee.on_epoch_end(0)
    
    
LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

class DefaultBNQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  """QuantizeConfig which only quantizes the output from a layer."""

  def __init__(self, quantize_output: bool = True, bit_width=8) -> None:
    self.quantize_output = quantize_output
    self.bit_width=bit_width

  def get_weights_and_quantizers(self, layer):
    return []

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    if self.quantize_output:
      return [MovingAverageQuantizer(
          num_bits=self.bit_width, per_axis=False, symmetric=False, narrow_range=False)]
    return []

  def get_config(self):
    return {'quantize_output': self.quantize_output}
    
class DefaultPreluQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  """QuantizeConfig which only quantizes the output from a layer."""

  def __init__(self, quantize_output: bool = True, bit_width=8) -> None:
    self.quantize_output = quantize_output
    self.bit_width=bit_width

  def get_weights_and_quantizers(self, layer):
    return []

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    if self.quantize_output:
      return [MovingAverageQuantizer(
          num_bits=self.bit_width, per_axis=False, symmetric=False, narrow_range=False)]
    return []

  def get_config(self):
    return {'quantize_output': self.quantize_output}
    

# Partially-QAT
def QAT2(bit, eval_path):
    print('----------------------conv/fc====================')

    #default_model = models.buildin_models("r18", dropout=0.4, emb_shape=512, output_layer="E",  activation='relu')
    path = 'checkpoints/9_28_17H_r50prelu/r50prelu_emore.h5'
    #eval_path = '../Datasets/faces_emore/cplfw.bin'#'/mnt/hdd0/hjyoo/Datasets/faces_emore/cplfw.bin'#'../Datasets/faces_emore/agedb_30.bin' #'/mnt/hdd0/hjyoo/Datasets/faces_emore/cfp_fp.bin' # '/mnt/hdd0/hjyoo/Datasets/faces_emore/lfw.bin'

    default_model = tf.keras.models.load_model(path, compile=False, )
    
    bit_width = int(bit)
    save_weight = False
    original_eval = True
    model_eval = True
    weight_path = 'weight'
    summary = False
    lr = 0.001
    fine_tuning_data = 10000
    fc_qat = 0
    store = 'quantization/conv_fc'
    
    print('----------------------{}---------------------------------'.format(str(bit_width)))
    
    if summary:
        print('original model')
        default_model.summary()
    
    
    if original_eval:
        print('----------------------original model---------------------------------')
        layer = default_model.layers[:-1]
        modela = keras.Model(default_model.inputs, layer[-1].output)
        ee = evals.eval_callback(modela, eval_path)
        ee.on_epoch_end(0)



    # origianl model check weight
    if save_weight:
        path = weight_path + '/before'
        os.mkdir(path)
        weight_check(default_model, path)
#        writer = tf.summary.create_file_writer(path)
#        with writer.as_default():
#            for w in default_model.weights:
#                tf.summary.histogram(str(w.name), w.numpy(), step=0)

    store_path = store + '/{}bit.h5'.format(bit)
    if not os.path.exists(store_path):
    
        config1 = DefaultBNQuantizeConfig(bit_width)
        config2 = DefaultPreluQuantizeConfig(bit_width)

        layer = default_model.layers[:-1]
        defaults_model = keras.models.Model(default_model.inputs, layer[-1].output)

        def apply_f(layer):
            if isinstance(layer, keras.layers.BatchNormalization) and ('1' in layer.name.split('_') or layer.name == 'E_batchnorm'):
                return tfmot.quantization.keras.quantize_annotate_layer(layer, config1)
            elif isinstance(layer, keras.layers.PReLU):
                return tfmot.quantization.keras.quantize_annotate_layer(layer, config2)
            elif isinstance(layer, models.NormDense):
                return tfmot.quantization.keras.quantize_annotate_layer(layer, config1)
            return layer
        keras.utils.plot_model(defaults_model, show_shapes=True, to_file='original.png')
        cloning = tf.keras.models.clone_model(defaults_model, clone_function= apply_f)

        if summary:
            print('cloning model')
            cloning.summary()

        quantize_scope = tfmot.quantization.keras.quantize_scope
        quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
        with quantize_scope({'DefaultBNQuantizeConfig': DefaultBNQuantizeConfig,
        'DefaultPreluQuantizeConfig': DefaultPreluQuantizeConfig}):

            q_model = quantize_annotate_model(cloning)
            from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit        import default_n_bit_quantize_scheme

            scheme = default_n_bit_quantize_scheme.DefaultNBitQuantizeScheme(num_bits_weight=bit_width, num_bits_activation=bit_width, disable_per_axis=False)

            q_aware_model = tfmot.quantization.keras.quantize_apply(q_model, scheme=scheme)

        #import numpy as np
        # if model_eval:
        #     print('-----------------------qat_model_before_finetuning---------------------')
        #     ee = evals.eval_callback(q_aware_model, eval_path)
        #     ee.on_epoch_end(0)

        arc_kwargs = {'loss_top_k': 1, 'append_norm': False, 'partial_fc_split': 0, 'name': 'arcface'}

        arcface_logits = models.NormDense(85742, None, **arc_kwargs, dtype="float32")

        inputs = q_aware_model.inputs[0]
        embedding = q_aware_model.outputs[0]

        arcface_logits.build(embedding.shape)
        output_fp32 = arcface_logits(embedding)

        if fc_qat:
            output_fp32 = quantize_annotate_model(output_fp32)
            tfmot.quantization.keras.quantize_apply(output_fp32, scheme=scheme)

        q_aware_model = keras.models.Model(inputs, output_fp32)

        # weight load
        weights = default_model.layers[-1].get_weights()
        q_aware_model.layers[-1].set_weights(weights)

        if summary:
            print('QAT model')
            q_aware_model.summary()
        # fine-tuning
        data_path = '/mnt/hdd0/hjyoo/Datasets/faces_emore_112x112_folders'
        train_ds, steps_per_epoch = data.prepare_dataset(data_path, batch_size=100, fine_tuning=True)
        print('train_ds', train_ds)
        tt = train_ds.take(int(fine_tuning_data / 100))
        print('tt', tt)

        # compiler
        q_aware_model.compile(optimizer = tfa.optimizers.SGDW(learning_rate=lr, momentum=0.9, weight_decay=5e-4),
        loss = [losses.ArcfaceLoss()],
        metrics = {'arcface': 'accuracy'},
        loss_weights= {'arcface': 1})

        # model.fit
        q_aware_model.fit(tt, epochs=1,)

        q_aware_model.save(store_path)

    else:
        config1 = DefaultBNQuantizeConfig(bit_width)
        config2 = DefaultPreluQuantizeConfig(bit_width)

        quantize_scope = tfmot.quantization.keras.quantize_scope
        with quantize_scope({'DefaultBNQuantizeConfig': DefaultBNQuantizeConfig,
                             'DefaultPreluQuantizeConfig': DefaultPreluQuantizeConfig}):
            q_aware_model = tf.keras.models.load_model(store_path, compile=False, )
            print('exist fine-tuning model')
    
    
    # remove fc normdense
    layer = q_aware_model.layers[:-1]
    q_aware_model = keras.models.Model(q_aware_model.inputs, layer[-1].output)
    
    if model_eval:
        print('-----------------------QAT model after Finetuning---------------------')
        ee = evals.eval_callback(q_aware_model, eval_path)
        acc, thresh = ee.on_epoch_end(0)
        with open('{}/result.txt'.format(store), 'a') as f:
            f.write('{}--> {}-bit acc:{:0.4f} thresh:{:0.6f}\n'.format(eval_path.split('/')[-1].split('.')[0], bit, acc,
                                                                       thresh))
    
    
    # check weight
    if save_weight:
        path = weight_path + '/after'
        os.mkdir(path)
        weight_check(q_aware_model, path)


# Fully-QAT
def QAT3(bit, eval_path):
    print('----------------------total====================')

    #default_model = models.buildin_models("r18", dropout=0.4, emb_shape=512, output_layer="E",  activation='relu')
    path = 'checkpoints/9_28_17H_r50prelu/r50prelu_emore.h5'
    #eval_path = eval_path#'/mnt/hdd0/hjyoo/Datasets/faces_emore/cplfw.bin'#'../Datasets/faces_emore/agedb_30.bin' #'/mnt/hdd0/hjyoo/Datasets/faces_emore/cfp_fp.bin' # '/mnt/hdd0/hjyoo/Datasets/faces_emore/lfw.bin'
    
    default_model = tf.keras.models.load_model(path, compile=False, )
    
    bit_width = int(bit)
    save_weight = False
    model_eval = True
    weight_path = 'weight'
    original_eval = True
    summary = False
    lr = 0.001
    fine_tuning_data = 10000
    fc_qat = 0
    store = 'quantization/total'
    
    print('----------------------{}---------------------------------'.format(str(bit_width)))
    
    if summary:
        print('original model')
        default_model.summary()
    
    
    if original_eval:
        print('----------------------original model---------------------------------')
        layer = default_model.layers[:-1] 
        modela= keras.Model(default_model.inputs,layer[-1].output)
        ee = evals.eval_callback(modela, eval_path)
        ee.on_epoch_end(0)
    

    
    # origianl model check weight
    if save_weight:
        path = weight_path + '/before'
        os.mkdir(path)
        weight_check(default_model, path)
#        writer = tf.summary.create_file_writer(path)
#        with writer.as_default():
#            for w in default_model.weights:
#                tf.summary.histogram(str(w.name), w.numpy(), step=0


    import numpy as np

    store_path = store + '/{}bit.h5'.format(bit)
    if not os.path.exists(store_path):

        config1 = DefaultBNQuantizeConfig(bit_width)
        config2 = DefaultPreluQuantizeConfig(bit_width)

        def apply_f(layer):
            if (isinstance(layer, keras.layers.BatchNormalization) and
                    '1' in layer.name.split('_')) or layer.name == 'E_batchnorm':
                return tfmot.quantization.keras.quantize_annotate_layer(layer, config1)
            elif isinstance(layer, keras.layers.PReLU):
                return tfmot.quantization.keras.quantize_annotate_layer(layer, config2)
            elif isinstance(layer, models.NormDense):
                return tfmot.quantization.keras.quantize_annotate_layer(layer, config1)
            return layer

        #keras.utils.plot_model(default_model, show_shapes=True, to_file='original.png')
        cloning = tf.keras.models.clone_model(default_model, clone_function=apply_f)

        if summary:
            print('cloning model')
            cloning.summary()

        quantize_scope = tfmot.quantization.keras.quantize_scope
        quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
        with quantize_scope({'DefaultBNQuantizeConfig': DefaultBNQuantizeConfig,
                             'DefaultPreluQuantizeConfig': DefaultPreluQuantizeConfig}):

            q_model = quantize_annotate_model(cloning)
            from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit import \
                default_n_bit_quantize_scheme

            scheme = default_n_bit_quantize_scheme.DefaultNBitQuantizeScheme(num_bits_weight=bit_width,
                                                                             num_bits_activation=bit_width,
                                                                             disable_per_axis=False)

            q_aware_model = tfmot.quantization.keras.quantize_apply(q_model, scheme=scheme)

        if summary:
            print('QAT model')
            q_aware_model.summary()

        print('Fine-tuning')
        # fine-tuning
        data_path = '/mnt/hdd0/hjyoo/Datasets/faces_emore_112x112_folders'
        train_ds, steps_per_epoch = data.prepare_dataset(data_path, batch_size=100, fine_tuning=True)
        print('train_ds', train_ds)
        tt = train_ds.take(int(fine_tuning_data / 100))
        print('tt', tt)

        # compiler
        q_aware_model.compile(optimizer = tfa.optimizers.SGDW(learning_rate=lr, momentum=0.9, weight_decay=5e-4),
        loss = [losses.ArcfaceLoss()],
        metrics = {'quant_arcface': 'accuracy'},
        loss_weights= {'quant_arcface': 1})

        # model.fit
        q_aware_model.fit(tt, epochs=1,)

        q_aware_model.save(store_path)

    else:
        config1 = DefaultBNQuantizeConfig(bit_width)
        config2 = DefaultPreluQuantizeConfig(bit_width)

        quantize_scope = tfmot.quantization.keras.quantize_scope
        with quantize_scope({'DefaultBNQuantizeConfig': DefaultBNQuantizeConfig,
                             'DefaultPreluQuantizeConfig': DefaultPreluQuantizeConfig}):
            q_aware_model = tf.keras.models.load_model(store_path, compile=False, )
            print('exist fine-tuning model')

    # # model store
    # if store:
    #     store_path = store + '/{}bit.h5'.format(bit)
    #     if not os.path.exists(store_path):
    #         q_aware_model.save(store_path)
    #     else:
    #         print('exist {}'.format(store_path))
    
    
    # remove fc normdense
    layer = q_aware_model.layers[:-1]
    q_aware_model = keras.models.Model(q_aware_model.inputs, layer[-1].output)

    keras.utils.plot_model(q_aware_model, show_shapes=True, to_file='original.png')

    if model_eval:
        print('-----------------------QAT model after Finetuning---------------------')
        ee = evals.eval_callback(q_aware_model, eval_path)
        acc, thresh = ee.on_epoch_end(0)
        with open('{}/result.txt'.format(store), 'a') as f:
            f.write('{}--> {}-bit acc:{:0.4f} thresh:{:0.6f}\n'.format(eval_path.split('/')[-1].split('.')[0], bit, acc, thresh))
    
    
    # check weight
    if save_weight:
        path = weight_path + '/after'
        os.mkdir(path)
        weight_check(q_aware_model, path)
    

def weight_view(bit_width, path, case):
    quantize_scope = tfmot.quantization.keras.quantize_scope
    with quantize_scope({'DefaultBNQuantizeConfig': DefaultBNQuantizeConfig,
                         'DefaultPreluQuantizeConfig': DefaultPreluQuantizeConfig}):
        store_model = tf.keras.models.load_model(path, compile=False,)

    bins = [0, 0, 0, 0, 100, 120, 140, 180, 200]

    for layer in store_model.layers:

        if hasattr(layer, 'quantize_config'):
            for weight, quantizer, quantizer_vars in layer._weight_vars:
                quantized_w = quantizer(weight, False, quantizer_vars)
                try:
                    quantized_w_c = quantized_w[:,:,:,0]
                except:

                    quantized_w_c = quantized_w[:,0]

                print(plt.hist(quantized_w_c.numpy().flatten(), bins=bins[bit_width], color='r' , edgecolor='black', rwidth=2.0))
                plt.xlabel('Quantized value', fontsize=16)
                plt.ylabel('Frequency',  fontsize=16)
                plt.savefig('weight/{}/{}bit/{}.png'.format(case, bit_width, layer.name))
                plt.cla()
        if 'bn' in layer.name:
            #print(layer)
            pass

def get_attribute(path):
    quantize_scope = tfmot.quantization.keras.quantize_scope
    with quantize_scope({'DefaultBNQuantizeConfig': DefaultBNQuantizeConfig,
                         'DefaultPreluQuantizeConfig': DefaultPreluQuantizeConfig}):
        store_model = tf.keras.models.load_model(path, compile=False, )

    d = dict()

    for layer in store_model.layers:

        if hasattr(layer, 'quantize_config'):
            print(layer.name)




#  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 python QAT.py
# https://www.tensorflow.org/model_optimization/guide/quantization/training_comprehensive_guide#overview

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bit', default=8)
parser.add_argument('--dataset', default='lfw')

args = parser.parse_args()

if args.dataset == 'lfw':
    eval_p = '../Datasets/faces_emore/lfw.bin'
if args.dataset == 'cfp_fp':
    eval_p = '../Datasets/faces_emore/cfp_fp.bin'
if args.dataset == 'agedb':
    eval_p = '../Datasets/faces_emore/agedb_30.bin'
if args.dataset == 'calfw':
    eval_p = '../Datasets/faces_emore/calfw.bin'
if args.dataset == 'cplfw':
    eval_p = '../Datasets/faces_emore/cplfw.bin'


#QAT2(2, eval_p) # conv_fc partial QAT
#QAT3(args.bit, eval_p) # total Fully QAT

# how = 0
# cases = ['partial_qat', 'fully_qat']
# s_path = ['conv_fc', 'total']
# bit = [5] # 8,7,6,5,4
#
# for b in bit:
#     weight_view(b, 'quantization/{}/{}bit.h5'.format(s_path[how], b), cases[how])

get_attribute('quantization/total/8bit.h5')