
import numpy as np
import pickle
import argparse

from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import (GlobalAveragePooling2D, Dropout, Dense)
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy

from arcface_layer import ArcFace

BATCH_SIZE=32
NO_EPOCHS = 1

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


def load_cifar10(data_path):
    # Load individual CIFAR-10 train batches
    raw_train_data_1, train_labels_1 = get_data(data_path + '/data_batch_1')
    raw_train_data_2, train_labels_2 = get_data(data_path + '/data_batch_2')
    raw_train_data_3, train_labels_3 = get_data(data_path + '/data_batch_3')
    raw_train_data_4, train_labels_4 = get_data(data_path + '/data_batch_4')
    raw_train_data_5, train_labels_5 = get_data(data_path + '/data_batch_5')
    
    # Merge train batches
    raw_train_data = np.concatenate((raw_train_data_1, raw_train_data_2, raw_train_data_3, raw_train_data_4, raw_train_data_5))
    train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3, train_labels_4, train_labels_5))

    # Convert image train data
    train_data = convert_data(raw_train_data)   

    # Load test batch
    raw_test_data, test_labels = get_data(data_path + '/test_batch')
    test_data = convert_data(raw_test_data)   

    return train_data, train_labels, test_data, test_labels


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
    parser = argparse.ArgumentParser(description='Perform basic training on CIFAR-10.')
    parser.add_argument('--path', help='Path to data.')
    args = parser.parse_args()

    # Load CIFAR-10 data
    train_data, train_labels, test_data, test_labels = load_cifar10(data_path=args.path)

    # Define and compile Keras model
    model = get_model()

    # Fit model to training data
    model.fit(x=train_data, y=train_labels, batch_size=BATCH_SIZE, epochs=NO_EPOCHS, shuffle=True)
    

if __name__ == "__main__":
    main()
