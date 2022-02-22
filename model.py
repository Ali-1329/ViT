import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

class PatchEmbedding(tf.keras.layers.Layer):                   
    def __init__(self, size, num_of_patches, projection_dim):   
        super().__init__()
        self.size = size     
        self.num_of_patches = num_of_patches + 1              
        self.projection_dim = projection_dim   

        self.projection = tf.keras.layers.Dense(projection_dim)    
        self.clsToken = tf.Variable(initial_value=tf.keras.initializers.GlorotNormal()(shape=(1, 1, projection_dim)), trainable=True)   

        self.positionEmbedding = tf.keras.layers.Embedding(self.num_of_patches, projection_dim)           
       
    
    def call(self, inputs):
        patches = tf.image.extract_patches(inputs, sizes=[1, self.size, self.size, 1],                         
                            strides=[1, self.size, self.size, 1], rates=[1, 1, 1, 1], padding='VALID')
        patches = tf.reshape(patches, (tf.shape(inputs)[0], -1, self.size*self.size*3))                       

        patches = self.projection(patches)  

        clsToken = tf.repeat(self.clsToken, repeats=tf.shape(inputs)[0],axis=0)  
        patches = tf.concat((clsToken, patches), axis=1)            

        positions = tf.range(0, self.num_of_patches, 1)[tf.newaxis,...]  
        positionalEmbedding = self.positionEmbedding(positions)  
        patches = patches + positionalEmbedding        
        return patches


class TransformerLayer(tf.keras.layers.Layer):    # d_model = projection_dim
    def __init__(self, d_model, heads, mlp_rate, dropout_rate = 0.1):
        super().__init__()
        self.layernorm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim= d_model//heads, value_dim = d_model//heads, dropout=dropout_rate)
       

        self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * mlp_rate, activation='gelu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(d_model, activation='gelu'),
            tf.keras.layers.Dropout(dropout_rate)
        ])


    def call(self, inputs, training=True):   
        out_1 = self.layernorm_1(inputs)
        out_1 = self.mha(out_1,out_1, training=training)    
        out_1 = out_1 + inputs

        out_2 = self.layernorm_2(out_1)
        out_2 = self.mlp(out_2, training=training)
        out_2 = out_1 + out_2
        return out_2


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, heads, mlp_rate, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.encoders = [TransformerLayer(d_model, heads, mlp_rate, dropout_rate) for _ in range(num_layers)]
        # stacking transformer layers

    def call(self, inputs, training=True):
        x = inputs
        for layer in self.encoders:
            x = layer(x, training=training)
        return x


class ViT(tf.keras.Model):
    def __init__(self, num_classes,patch_size, num_of_patches, d_model, heads, num_layers, mlp_rate, dropout_rate=0.1):
        super().__init__()
        self.PatchEmbedding = PatchEmbedding(patch_size, num_of_patches, d_model)
        self.encoder = TransformerEncoder(d_model, heads, mlp_rate, num_layers, dropout_rate)
        self.prediction = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(mlp_rate*d_model, activation='gelu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax'),

        ])

    def call(self, inputs, training=True):
        patches = self.PatchEmbedding(inputs)
        encoderResult = self.encoder(patches, training=training)
        clsResult = encoderResult[:, 0, :] 
        prediction = self.prediction(clsResult, training=training)
        return prediction


