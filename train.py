import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from model import ViT


if __name__ == '__main__':
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    preprocessingModel = data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Normalization(),
        tf.keras.layers.Resizing(144, 144)
    ])

    preprocessingModel.layers[0].adapt(x_train)   

    augmentaionModel =tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(factor=0.2),
        tf.keras.layers.RandomZoom(width_factor=0.2, height_factor=0.2)
    ])

    def convert_to_dataset(data, batch_size, shuffle=False, augment=False):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.map(lambda x, y:(preprocessingModel(x)[0],y), num_parallel_calls = tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(len(dataset))
        dataset = dataset.batch(batch_size, drop_remainder=True )

        if augment:
            dataset = dataset.map(lambda x, y:(augmentaionModel(x, training=True),y), num_parallel_calls = tf.data.AUTOTUNE)
        
        return dataset.prefetch(tf.data.AUTOTUNE)

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))

    strategy = tf.distribute.TPUStrategy(resolver)


    trainingData = convert_to_dataset(data=(x_train, y_train), batch_size=1024, shuffle=True, augment=True )

    valData = convert_to_dataset(data=(x_test, y_test), batch_size=1024, shuffle=True, augment=False )

    with strategy.scope():
        ViTclassifier = ViT(
        num_classes =10,
        patch_size=16,
        num_of_patches = (144//16)**2,     
        d_model=128,
        heads=2,
        num_layers=4,        
        mlp_rate=2,
        dropout_rate=0.1
    )

        ViTclassifier.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        optimizer= 'adam',
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                    tf.keras.metrics.SparseCategoricalAccuracy(name='top_5_accuracy')
                ]
        )


    ViTclassifier.fit(x = trainingData, validation_data=valData, batch_size=1024, epochs = 100)
