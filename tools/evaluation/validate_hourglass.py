#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import time
import glob
import numpy as np
from PIL import Image
from operator import mul
from functools import reduce
import MNN
import onnxruntime
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.lite.python import interpreter as interpreter_wrapper
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from hourglass.data import HG_OUTPUT_STRIDE
from hourglass.postprocess import post_process_heatmap, post_process_heatmap_simple
from common.data_utils import preprocess_image
from common.utils import get_classes, get_skeleton, render_skeleton

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def process_heatmap(heatmap, image_file, image, scale, class_names, skeleton_lines, output_path):
    start = time.time()
    # parse out predicted keypoint from heatmap
    keypoints = post_process_heatmap_simple(heatmap)

    # rescale keypoints back to origin image shape
    keypoints_dict = dict()
    for i, keypoint in enumerate(keypoints):
        keypoints_dict[class_names[i]] = (keypoint[0] * scale[0] * HG_OUTPUT_STRIDE, keypoint[1] * scale[1] * HG_OUTPUT_STRIDE, keypoint[2])

    end = time.time()
    print("PostProcess time: {:.8f}ms".format((end - start) * 1000))

    print('Keypoints detection result:')
    for keypoint in keypoints_dict.items():
        print(keypoint)

    # draw the keypoint skeleton on image
    image_array = np.array(image, dtype='uint8')
    image_array = render_skeleton(image_array, keypoints_dict, skeleton_lines)

    # save or show result
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, os.path.basename(image_file))
        Image.fromarray(image_array).save(output_file)
    else:
        Image.fromarray(image_array).show()
    return


def validate_hourglass_model(model, image_file, class_names, skeleton_lines, model_input_shape, loop_count, output_path):
    img = Image.open(image_file).convert('RGB')
    image = np.array(img, dtype='uint8')
    image_data = preprocess_image(img, model_input_shape)
    image_size = img.size
    scale = (image_size[0] * 1.0 / model_input_shape[1], image_size[1] * 1.0 / model_input_shape[0])

    # predict once first to bypass the model building time
    model.predict(image_data)

    start = time.time()
    for i in range(loop_count):
        prediction = model.predict(image_data)
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    # check to handle multi-output model
    if isinstance(prediction, list):
        prediction = prediction[-1]
    heatmap = prediction[0]
    process_heatmap(heatmap, image_file, img, scale, class_names, skeleton_lines, output_path)
    return


def validate_hourglass_model_tflite(interpreter, image_file, class_names, skeleton_lines, loop_count, output_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #print(input_details)
    #print(output_details)

    # check the type of the input tensor
    if input_details[0]['dtype'] == np.float32:
        floating_model = True

    img = Image.open(image_file).convert('RGB')
    image = np.array(img, dtype='uint8')

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    model_input_shape = (height, width)

    image_data = preprocess_image(img, model_input_shape)
    image_size = img.size
    scale = (image_size[0] * 1.0 / model_input_shape[1], image_size[1] * 1.0 / model_input_shape[0])

    # predict once first to bypass the model building time
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()

    start = time.time()
    for i in range(loop_count):
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    prediction = []
    for output_detail in output_details:
        output_data = interpreter.get_tensor(output_detail['index'])
        prediction.append(output_data)

    heatmap = prediction[-1][0]
    process_heatmap(heatmap, image_file, img, scale, class_names, skeleton_lines, output_path)
    return


def validate_hourglass_model_mnn(interpreter, session, image_file, class_names, skeleton_lines, loop_count, output_path):
    # assume only 1 input tensor for image
    input_tensor = interpreter.getSessionInput(session)
    # get input shape
    input_shape = input_tensor.getShape()
    if input_tensor.getDimensionType() == MNN.Tensor_DimensionType_Tensorflow:
        batch, height, width, channel = input_shape
    elif input_tensor.getDimensionType() == MNN.Tensor_DimensionType_Caffe:
        batch, channel, height, width = input_shape
    else:
        # should be MNN.Tensor_DimensionType_Caffe_C4, unsupported now
        raise ValueError('unsupported input tensor dimension type')

    model_input_shape = (height, width)

    # prepare input image
    img = Image.open(image_file).convert('RGB')
    image = np.array(img, dtype='uint8')
    image_data = preprocess_image(img, model_input_shape)
    image_size = img.size
    scale = (image_size[0] * 1.0 / model_input_shape[1], image_size[1] * 1.0 / model_input_shape[0])

    # create a temp tensor to copy data,
    # use TF NHWC layout to align with image data array
    # TODO: currently MNN python binding have mem leak when creating MNN.Tensor
    # from numpy array, only from tuple is good. So we convert input image to tuple
    tmp_input_shape = (batch, height, width, channel)
    input_elementsize = reduce(mul, tmp_input_shape)
    tmp_input = MNN.Tensor(tmp_input_shape, input_tensor.getDataType(),\
                    tuple(image_data.reshape(input_elementsize, -1)), MNN.Tensor_DimensionType_Tensorflow)

    # predict once first to bypass the model building time
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)

    start = time.time()
    for i in range(loop_count):
        input_tensor.copyFrom(tmp_input)
        interpreter.runSession(session)
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))


    # we only handle single output model
    output_tensor = interpreter.getSessionOutput(session)
    output_shape = output_tensor.getShape()
    output_elementsize = reduce(mul, output_shape)

    assert output_tensor.getDataType() == MNN.Halide_Type_Float

    # copy output tensor to host, for further postprocess
    tmp_output = MNN.Tensor(output_shape, output_tensor.getDataType(),\
                #np.zeros(output_shape, dtype=float), output_tensor.getDimensionType())
                tuple(np.zeros(output_shape, dtype=float).reshape(output_elementsize, -1)), output_tensor.getDimensionType())

    output_tensor.copyToHostTensor(tmp_output)
    #tmp_output.printTensorData()

    output_data = np.array(tmp_output.getData(), dtype=float).reshape(output_shape)
    # our postprocess code based on TF NHWC layout, so if the output format
    # doesn't match, we need to transpose
    if output_tensor.getDimensionType() == MNN.Tensor_DimensionType_Caffe:
        output_data = output_data.transpose((0,2,3,1))
    elif output_tensor.getDimensionType() == MNN.Tensor_DimensionType_Caffe_C4:
        raise ValueError('unsupported output tensor dimension type')

    heatmap = output_data[0]
    process_heatmap(heatmap, image_file, img, scale, class_names, skeleton_lines, output_path)


def validate_hourglass_model_onnx(model, image_file, class_names, skeleton_lines, loop_count, output_path):
    input_tensors = []
    for i, input_tensor in enumerate(model.get_inputs()):
        input_tensors.append(input_tensor)
    # assume only 1 input tensor for image
    assert len(input_tensors) == 1, 'invalid input tensor number.'

    # check if input layout is NHWC or NCHW
    if input_tensors[0].shape[1] == 3:
        print("NCHW input layout")
        batch, channel, height, width = input_tensors[0].shape  #NCHW
    else:
        print("NHWC input layout")
        batch, height, width, channel = input_tensors[0].shape  #NHWC

    model_input_shape = (height, width)

    output_tensors = []
    for i, output_tensor in enumerate(model.get_outputs()):
        output_tensors.append(output_tensor)
    # assume only 1 output tensor
    assert len(output_tensors) == 1, 'invalid output tensor number.'

    # prepare input image
    img = Image.open(image_file).convert('RGB')
    image_data = preprocess_image(img, model_input_shape)
    image_size = img.size
    scale = (image_size[0] * 1.0 / model_input_shape[1], image_size[1] * 1.0 / model_input_shape[0])

    if input_tensors[0].shape[1] == 3:
        # transpose image for NCHW layout
        image_data = image_data.transpose((0,3,1,2))

    feed = {input_tensors[0].name: image_data}

    # predict once first to bypass the model building time
    prediction = model.run(None, feed)

    start = time.time()
    for i in range(loop_count):
        prediction = model.run(None, feed)

    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    # check to handle multi-output model
    if isinstance(prediction, list):
        prediction = prediction[-1]
    heatmap = prediction[0]
    process_heatmap(heatmap, image_file, img, scale, class_names, skeleton_lines, output_path)
    return


def validate_hourglass_model_pb(model, image_file, class_names, skeleton_lines, model_input_shape, loop_count, output_path):
    # check tf version to be compatible with TF 2.x
    global tf
    if tf.__version__.startswith('2'):
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()

    # NOTE: TF 1.x frozen pb graph need to specify input/output tensor name
    # so we need to hardcode the input/output tensor names here to get them from model
    output_tensor_name = 'graph/hg1_conv_1x1_predict/BiasAdd:0'

    # assume only 1 input tensor for image
    input_tensor_name = 'graph/image_input:0'

    img = Image.open(image_file).convert('RGB')
    image = np.array(img, dtype='uint8')
    image_data = preprocess_image(img, model_input_shape)
    image_size = img.size
    scale = (image_size[0] * 1.0 / model_input_shape[1], image_size[1] * 1.0 / model_input_shape[0])

    # We can list operations, op.values() gives you a list of tensors it produces
    # op.name gives you the name. These op also include input & output node
    # print output like:
    # prefix/Placeholder/inputs_placeholder
    # ...
    # prefix/Accuracy/predictions
    #
    # NOTE: prefix/Placeholder/inputs_placeholder is only op's name.
    # tensor name should be like prefix/Placeholder/inputs_placeholder:0

    #for op in model.get_operations():
        #print(op.name, op.values())

    image_input = model.get_tensor_by_name(input_tensor_name)
    output_tensor = model.get_tensor_by_name(output_tensor_name)

    # predict once first to bypass the model building time
    with tf.Session(graph=model) as sess:
        prediction = sess.run(output_tensor, feed_dict={
            image_input: image_data
        })

    start = time.time()
    for i in range(loop_count):
            with tf.Session(graph=model) as sess:
                prediction = sess.run(output_tensor, feed_dict={
                    image_input: image_data
                })
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    heatmap = prediction[0]
    process_heatmap(heatmap, image_file, img, scale, class_names, skeleton_lines, output_path)



#load TF 1.x frozen pb graph
def load_graph(model_path):
    # check tf version to be compatible with TF 2.x
    global tf
    if tf.__version__.startswith('2'):
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()

    # We parse the graph_def file
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="graph",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def load_val_model(model_path):
    # support of tflite model
    if model_path.endswith('.tflite'):
        from tensorflow.lite.python import interpreter as interpreter_wrapper
        model = interpreter_wrapper.Interpreter(model_path=model_path)
        model.allocate_tensors()

    # support of MNN model
    elif model_path.endswith('.mnn'):
        model = MNN.Interpreter(model_path)

    # support of TF 1.x frozen pb model
    elif model_path.endswith('.pb'):
        model = load_graph(model_path)

    # support of ONNX model
    elif model_path.endswith('.onnx'):
        model = onnxruntime.InferenceSession(model_path)

    # normal keras h5 model
    elif model_path.endswith('.h5'):
        model = load_model(model_path, compile=False)
        K.set_learning_phase(0)
    else:
        raise ValueError('invalid model file')

    return model


def main():
    parser = argparse.ArgumentParser(description='validate Hourglass model (h5/pb/onnx/tflite/mnn) with image')
    parser.add_argument('--model_path', help='model file to predict', type=str, required=True)
    parser.add_argument('--image_path', help='image file or directory to predict', type=str, required=True)
    parser.add_argument('--classes_path', help='path to class definitions, default=%(default)s', type=str, required=False, default='../../configs/mpii_classes.txt')
    parser.add_argument('--skeleton_path', help='path to keypoint skeleton definitions, default=%(default)s', type=str, required=False, default=None)
    parser.add_argument('--model_input_shape', help='model image input shape as <height>x<width>, default=%(default)s', type=str, default='256x256')
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)
    parser.add_argument('--output_path', help='output path to save predict result, default=%(default)s', type=str, required=False, default=None)

    args = parser.parse_args()

    # param parse
    if args.skeleton_path:
        skeleton_lines = get_skeleton(args.skeleton_path)
    else:
        skeleton_lines = None

    class_names = get_classes(args.classes_path)
    height, width = args.model_input_shape.split('x')
    model_input_shape = (int(height), int(width))


    model = load_val_model(args.model_path)
    if args.model_path.endswith('.mnn'):
        #MNN inference engine need create session
        session = model.createSession()

    # get image file list or single image
    if os.path.isdir(args.image_path):
        image_files = glob.glob(os.path.join(args.image_path, '*'))
        assert args.output_path, 'need to specify output path if you use image directory as input.'
    else:
        image_files = [args.image_path]


    # loop the sample list to predict on each image
    for image_file in image_files:
        # support of tflite model
        if args.model_path.endswith('.tflite'):
            validate_hourglass_model_tflite(model, image_file, class_names, skeleton_lines, args.loop_count, args.output_path)
        # support of MNN model
        elif args.model_path.endswith('.mnn'):
            validate_hourglass_model_mnn(model, session, image_file, class_names, skeleton_lines, args.loop_count, args.output_path)
        ## support of TF 1.x frozen pb model
        elif args.model_path.endswith('.pb'):
            validate_hourglass_model_pb(model, image_file, class_names, skeleton_lines, model_input_shape, args.loop_count, args.output_path)
        # support of ONNX model
        elif args.model_path.endswith('.onnx'):
            validate_hourglass_model_onnx(model, image_file, class_names, skeleton_lines, args.loop_count, args.output_path)
        ## normal keras h5 model
        elif args.model_path.endswith('.h5'):
            validate_hourglass_model(model, image_file, class_names, skeleton_lines, model_input_shape, args.loop_count, args.output_path)
        else:
            raise ValueError('invalid model file')


if __name__ == '__main__':
    main()
