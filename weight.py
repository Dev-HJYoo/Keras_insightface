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
import numpy as np


def weight_check(model, save=False):
    if model == None:
        raise ValueError
        
    if save != None and not os.path.exists(save):
        raise ValueError
        
    
    for w in model.weights:
        if 'kernel' in w.name:
            name = w.name.split(':')[0].replace('/', '_')
            np_w = w.numpy()
            flat = np_w.flatten()
            
            #counts, bins = np.histogram(flat)
            #counts, bins = np.histogram(flat,)
            #print(counts)
            #print(bins)
            
#            d = dict()
#            for a in flat:
#                
#                if a not in d:
#                    d[a] = 1
#                else:
#                    d[a] += 1
#                    
#            print(name)
#            print(len(d.items()))
#            print(flat.shape[0])
#            if 'quant_0_conv' in w.name:
#                print(d.items())
            
            #plt.hist(flat, bins, weights=counts, rwidth = 0.8)
            plt.hist(flat, 10,  edgecolor='k', range=(-0.25, 0.25)) # rwidth = 0.8
            plt.title(name)
            
            if save:
                plt.savefig(save + '/' + name)
            else:
                plt.show()
        

if __name__ == '__main__':

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
    
    
    path = 'checkpoints/9_28_17H_r50prelu/r50prelu_emore_basic_model_latest.h5'
        
    default_model = tf.keras.models.load_model(path, compile=False, )
    
    bit_width = 4
    weight_path = 'weight'
    lr = 0.1
    
    #print('original model')
    #default_model.summary()
    
    #print('----------------------original model---------------------------------')
    #ee = evals.eval_callback(default_model, '/mnt/hdd0/hjyoo/Datasets/faces_emore/lfw.bin')
    #ee.on_epoch_end(0)
    
    config = DefaultBNQuantizeConfig(bit_width)
    
    
    def apply_f(layer):
        if isinstance(layer, keras.layers.BatchNormalization):
            return tfmot.quantization.keras.quantize_annotate_layer(layer, config)
        elif isinstance(layer, keras.layers.PReLU):
            return tfmot.quantization.keras.quantize_annotate_layer(layer, config)
        return layer
    keras.utils.plot_model(default_model, show_shapes=True, to_file='original.png')
    cloning = tf.keras.models.clone_model(default_model, clone_function= apply_f)
    
    print('cloning model')
    #cloning.summary()
    
    quantize_scope = tfmot.quantization.keras.quantize_scope
    quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
    with quantize_scope({'DefaultBNQuantizeConfig': DefaultBNQuantizeConfig}):
         
        q_model = quantize_annotate_model(cloning)
        from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit        import default_n_bit_quantize_scheme
        
        scheme = default_n_bit_quantize_scheme.DefaultNBitQuantizeScheme(num_bits_weight=bit_width, num_bits_activation=bit_width)
        
        q_aware_model = tfmot.quantization.keras.quantize_apply(q_model, scheme=scheme)
    
    
    #    print('original weight')
    #    print(len(default_model.weights))
    #    print(default_model.weights[:5])
    #    
    #    print('QAT weight')
    #    print(len(q_aware_model.weights))
    #    print(q_aware_model.weights[:10])
        import matplotlib.pyplot as plt
        import numpy as np
        
        for l in default_model.layers:
            print(l)
            
        exit()
        
        for w in default_model.weights:
            if 'kernel' in w.name:
                np_w = w.numpy()
                flat = np_w.flatten()
                
                counts, bins = np.histogram(flat)
                print(counts, len(counts))
                print(bins, len(bins))
                
                plt.hist(bins[:-1], bins, weights=counts)
                plt.title(w.name)
                plt.show()
        
        exit()
        
    
    import numpy as np
    
    # check weight
    os.mkdir(weight_path + '/before')
    writer = tf.summary.create_file_writer(weight_path + '/before')
    with writer.as_default():
        for w in default_model.weights:
            tf.summary.histogram(str(w.name), w.numpy(), step=0)
    
            
    
    print('QAT model')
    #    q_aware_model.summary()
    print('-----------------------qat_model_before_finetuning---------------------')
    ee = evals.eval_callback(q_aware_model, '/mnt/hdd0/hjyoo/Datasets/faces_emore/lfw.bin')
    ee.on_epoch_end(0)
    
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
    tt = train_ds.take(100)
    print('tt', tt)
    
    # compiler
    q_aware_model.compile(optimizer = tfa.optimizers.SGDW(learning_rate=lr, momentum=0.9, weight_decay=5e-4),
    loss = [losses.ArcfaceLoss()],
    metrics = {'arcface': 'accuracy'},
    loss_weights= {'arcface': 1}) 
    
    # model.fint
    q_aware_model.fit(tt, epochs=1,)
    
    
    # remove fc normdense
    layer = q_aware_model.layers[:-1]
    q_aware_model = keras.models.Model(q_aware_model.inputs, layer[-1].output)
    
    print('-----------------------QAT model after Finetuning---------------------')
    ee = evals.eval_callback(q_aware_model, '/mnt/hdd0/hjyoo/Datasets/faces_emore/lfw.bin')
    ee.on_epoch_end(0)
    
    
    # check weight
    os.mkdir(weight_path + '/after')
    writer = tf.summary.create_file_writer(weight_path + '/after')
    with writer.as_default():
        for w in q_aware_model.weights:
            tf.summary.histogram(str(w.name), w.numpy(), step=0)