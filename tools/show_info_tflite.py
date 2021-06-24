# REF: https://stackoverflow.com/questions/62276989/tflite-can-i-get-graph-layer-sequence-information-directly-through-the-tf2-0-a

import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='YOLO_best_mAP.tflite')

layer_details = interpreter.get_tensor_details()
interpreter.allocate_tensors()

for layer in layer_details:
      print("\nLayer Name: {}".format(layer['name']))
      print("\tIndex: {}".format(layer['index']))
      print("\n\tShape: {}".format(layer['shape']))
      print("\tTensor: {}".format(interpreter.get_tensor(layer['index']).shape))
      print("\tTensor Type: {}".format(interpreter.get_tensor(layer['index']).dtype))
      print("\tQuantisation Parameters")
      print("\t\tScales: {}".format(layer['quantization_parameters']['scales'].shape))
      print("\t\tScales Type: {}".format(layer['quantization_parameters']['scales'].dtype))
      print("\t\tZero Points: {}".format(layer['quantization_parameters']['zero_points']))
      print("\t\tQuantized Dimension: {}".format(layer['quantization_parameters']['quantized_dimension']))
