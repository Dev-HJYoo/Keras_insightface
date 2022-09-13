import os


original = 'checkpoints/8_10_20H/cutting.h5'
quant = 'checkpoints/8_10_20H/q_model.tflite'

print('Origianl Model Size{:0.2f}MB'.format(os.path.getsize(original) / float(2**20)))
print('Quanti Model Size{:0.2f}MB'.format(os.path.getsize(quant) / float(2**20)))

