#from keras_resnet.models import ResNet50
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras import layers, models
import tensorflow as tf
from losses import db_loss, simpler_loss
import pydot

def db_simpler(nfilters=256):
    image_input = layers.Input(shape=(None, None, 3))
    backbone = ResNet50(include_top=False, weights='imagenet',
                        input_tensor=image_input)

    output_names = ['conv2_block3_out', 'conv3_block4_out',
                    'conv4_block6_out', 'conv5_block3_out']
    [resnet_conv2, resnet_conv3, resnet_conv4, resnet_conv5] = [x.output for x in backbone.layers if x.name in output_names]
    in2 = layers.Conv2D(nfilters, (1, 1), padding='same',
                        kernel_initializer='he_normal', name='in2')(resnet_conv2)
    in3 = layers.Conv2D(nfilters, (1, 1), padding='same',
                        kernel_initializer='he_normal', name='in3')(resnet_conv3)
    in4 = layers.Conv2D(nfilters, (1, 1), padding='same',
                        kernel_initializer='he_normal', name='in4')(resnet_conv4)
    in5 = layers.Conv2D(nfilters, (1, 1), padding='same',
                        kernel_initializer='he_normal', name='in5')(resnet_conv5)

    conv5 = layers.Conv2D(nfilters//4, (3, 3), padding='same', kernel_initializer='he_normal', name='conv2d_in5')(in5)
    P5 = layers.UpSampling2D(size=(8, 8), name='upsampling_conv5')(conv5)
    out4 = layers.Add(name='join_in4_in5')([in4, layers.UpSampling2D(size=(2, 2))(in5)])
    conv4 = layers.Conv2D(nfilters//4, (3, 3), padding='same', kernel_initializer='he_normal', name='conv2d_out4')(out4)
    P4 = layers.UpSampling2D(size=(4, 4), name='upsampling_conv4')(conv4)
    out3 = layers.Add(name='join_in3_out4')([in3, layers.UpSampling2D(size=(2, 2))(out4)])
    conv3 = layers.Conv2D(nfilters//4, (3, 3), padding='same', kernel_initializer='he_normal', name='conv2d_out3')(out3)
    P3 = layers.UpSampling2D(size=(2, 2), name='upsampling_conv3')(conv3)     
    out2 = layers.Add(name='join_in2_out3')([in2, layers.UpSampling2D(size=(2, 2))(out3)]) 
    P2 = layers.Conv2D(nfilters//4, (3, 3), padding='same', kernel_initializer='he_normal', name='conv2d_out2')(out2)
        
    fuse = layers.Concatenate(name='concatenate_P2345')([P2, P3, P4, P5])

    proba = layers.Conv2D(nfilters//4, (3, 3), padding='same',
                        kernel_initializer='he_normal', use_bias=False, name='conv2d_concatenate_P2345')(fuse)
    proba = layers.BatchNormalization(name="batchnormalization_proba_1")(proba)
    proba = layers.ReLU(name="relu_proba_1")(proba)
    proba = layers.Conv2DTranspose(nfilters//4, (2, 2), strides=(
                                    2, 2), kernel_initializer='he_normal', use_bias=False, name='conv2d_proba_1')(proba)
    proba = layers.BatchNormalization(name="batchnormalization_proba_2")(proba)
    proba = layers.ReLU(name="relu_proba_2")(proba)
    proba = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                                activation='sigmoid', name='conv2d_proba_2')(proba)

    model = models.Model(inputs=image_input, outputs=proba)
    
    return model, model


if __name__ == '__main__':
    model, _ = db_simpler()
    # model.summary()
