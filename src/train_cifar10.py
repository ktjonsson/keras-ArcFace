
import numpy as np
import pickle

from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import (GlobalAveragePooling2D, Dropout, Dense)
from keras.optimizers import Adam

from arcface_layer import ArcFace


def get_data(file):
    # Load and unpickle CIFAR-10 batch of data
    with open(file, 'rb') as fo:
        d= pickle.load(fo, encoding='bytes')
    return d[b'data'], d[b'labels']


def convert_data(raw, no_channels=3, img_size=32):
    # Convert image data
    raw_float = np.array(raw, dtype=float) / 255.0
    images = raw_float.reshape([-1, no_channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])
    return images


def load_cifar10():
    # Load individual CIFAR-10 batches
    raw_data_1, labels_1 = get_data('/home/kenneth/Data/cifar-10-batches-py/data_batch_1')
    raw_data_2, labels_2 = get_data('/home/kenneth/Data/cifar-10-batches-py/data_batch_2')
    raw_data_3, labels_3 = get_data('/home/kenneth/Data/cifar-10-batches-py/data_batch_3')
    raw_data_4, labels_4 = get_data('/home/kenneth/Data/cifar-10-batches-py/data_batch_4')
    raw_data_5, labels_5 = get_data('/home/kenneth/Data/cifar-10-batches-py/data_batch_5')
    
    # Merge batches
    raw_data = np.concatenate((raw_data_1, raw_data_2, raw_data_3, raw_data_4, raw_data_5))
    labels = np.concatenate((labels_1, labels_2, labels_3, labels_4, labels_5))

    # Convert image data
    data = convert_data(raw_data)    
    return data, labels


def get_model():
    # Construct basic MobileNet excluding top layers
    mobile_net = MobileNet(input_shape=None, alpha=0.25, depth_multiplier=1, dropout=1e-3, include_top=False, weights=None, input_tensor=None, pooling=None, classes=None)
    
    # Initialize custom ArcFace layer
    af_layer = ArcFace(output_dim=10, class_num=10, margin=0.5, scale=64.)

    # Add custom top layers to MobileNet
    x1 = GlobalAveragePooling2D()(mobile_net.output)
    x1 = Dropout(rate=0.1)(x1)
    embedding_output = Dense(64)(x1)
    arcface_output = af_layer(embedding_output)
    model = Model(inputs=mobile_net.input, outputs=arcface_output)
 
    # Compile model
    op = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=op,
        loss=af_layer.loss,
        metrics=[af_layer.accuracy])
    return model


def main():
    # Load CIFAR-10 data
    data, labels = load_cifar10()

    # Define and compile Keras model
    model = get_model()

    # Fit model to training data
    model.fit(x=data, y=labels)



if __name__ == "__main__":
    main()
