# This code should be run after FPGA_server.py
import socket
import numpy as np
from bitstream import BitStream
import time
import tensorflow as tf
from model import LeNet


FPGA_SERVER_IP = '127.0.0.1'    # The server's hostname or IP address
FPGA_SERVER_PORT = 65432        # The port used by the server

NUM_BITS = 16  # Should be divisible by 8
NUM_BITS_DECIMAL = 12

RECV_BUFFER_SIZE = 1024


def img_loader():
    # This function should generate images to be processed by DNN
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data('/Users/yudongtao/Research/DNN_on_FPGA/mnist.npz')
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    for i in range(x_test.shape[0]):
        yield x_test[i:i+1, :, :]


def proc_dnn_model(sess, img, op):
    # This function should take image input and generate output before last fc layer
    out = sess.run(op, feed_dict={x: img})
    return list(out[0])


def encode_packet(fc_input):
    # This function should take list of number inputs and convert it to fixed point numbers in bytes
    data = BitStream()
    for num in fc_input:
        tmp = round(num * (2 ** NUM_BITS_DECIMAL), 0)
        tmp = tmp % (2 ** NUM_BITS)
        # print('{:.8f} -> {:.8f}'.format(num, tmp / (2 ** NUM_BITS_DECIMAL)))
        tmp = int(tmp)
        scale = 2 ** NUM_BITS
        for i in range(NUM_BITS // 8):
            scale = scale // 256
            data.write(tmp // scale % 256, np.uint8)

    print(data)
    data = data.read(bytes, len(fc_input)*(NUM_BITS//8))
    print(data)
    return data


# x = encode_packet([0.5, 1, 0.004, 0.000005, 5.3, 2.7, 100, 1000])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((FPGA_SERVER_IP, FPGA_SERVER_PORT))

    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    _, fc = LeNet(x)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        st_time = time.time()
        for idx, img in enumerate(img_loader()):
            load_time = time.time()
            fc_input = proc_dnn_model(sess, img, fc)
            gpu_time = time.time()
            s.sendall(encode_packet(fc_input))
            output = s.recv(RECV_BUFFER_SIZE)
            fpga_time = time.time()
            print('Received', repr(output))
            print('Elapsed Time ({}): Loading Time = {}, GPU Time = {}, FPGA Time = {}'.format(idx, load_time - st_time,
                                                                                               gpu_time - load_time,
                                                                                               fpga_time - gpu_time))
            input("Press Enter to continue...")
            st_time = time.time()
