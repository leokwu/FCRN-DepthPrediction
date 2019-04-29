# USAGE
# python3.6 predict.py --input_image input.jpg --pb_model model.pb  --output_image out.jpg

import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from imutils.video import VideoStream
import cv2
import time

def load_model(pb_model):
    with tf.gfile.GFile(pb_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    for op in graph.get_operations():
        print(op.name)
    return graph


def pb_predict_picture(input_image, output_image, graph):
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # Read image
    img = Image.open(input_image)
    img = img.resize([width, height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis=0)

    sess = tf.Session(graph=graph)
    input_node = graph.get_tensor_by_name("Placeholder:0")
    predict_node = graph.get_tensor_by_name("ConvPred/ConvPred:0")

    pred = sess.run(predict_node, feed_dict={input_node: img})


    # Plot result
    fig = plt.figure()
    ii = plt.imshow(pred[0, :, :, 0], interpolation='nearest')
    fig.colorbar(ii)
    plt.savefig(output_image)
    plt.show()


def pb_predict(graph):
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    sess = tf.Session(graph=graph)
    input_node = graph.get_tensor_by_name("Placeholder:0")
    predict_node = graph.get_tensor_by_name("ConvPred/ConvPred:0")
    # fig, ax = plt.subplots()

    fps = 0

    while True:
        start = time.time()
        frame = vs.read()
        # Read image
        img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        img = np.array(img).astype('float32')
        img = np.expand_dims(np.asarray(img), axis=0)
        pred = sess.run(predict_node, feed_dict={input_node: img})

        # img2 = cv2.cvtColor(pred[0, :, :, 0], cv2.COLOR_GRAY2RGB)
        # HSV = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        label = 'FPS: {:.4f}'.format(fps)
        cv2.putText(pred[0, :, :, 0], label, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Frame", pred[0, :, :, 0])
        
        # Plot result
        # ax.cla()
        # ii = plt.imshow(pred[0, :, :, 0], interpolation='nearest')
        # fig.colorbar(ii)
        # print(pred.shape)
        # plt.show()

        used_time = time.time() - start
        fps = (1/used_time)
        # show the output frame and wait for a key press
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    vs.stop()
    
    return pred
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_image", type=str, required=False,
                        help="path input image")
    parser.add_argument("-m", "--pb_model", type=str, required=True,
                        help="path to pb model")
    parser.add_argument("-o", "--output_image", type=str, required=False,
                        help="path output image")
    args = parser.parse_args()

    # Predict the image
    graph = load_model(args.pb_model)
    pb_pred = pb_predict_picture(args.input_image, args.output_image, graph)
    pb_pred = pb_predict(graph)
    
    os._exit(0)

if __name__ == '__main__':
    main()

        



