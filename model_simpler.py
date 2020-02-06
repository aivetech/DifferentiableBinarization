#from keras_resnet.models import ResNet50
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras import layers, models
import tensorflow as tf
from losses import db_loss, simpler_loss
import pydot

def db_simpler(input_size=640, k=50):
    image_input = layers.Input(shape=(None, None, 3))
    backbone = ResNet50(include_top=False, weights='imagenet', input_tensor = image_input)
    
    output_names = ['conv2_block3_out','conv3_block4_out','conv4_block6_out','conv5_block3_out']
    [C2, C3, C4, C5] = [x.output for x in backbone.layers if x.name in output_names]
    in2 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in2')(C2)
    in3 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in3')(C3)
    in4 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in4')(C4)
    in5 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in5')(C5)
    print("IN shapes")
    print(in2.shape)
    print(in3.shape)
    print(in4.shape)
    print(in5.shape)

    # 1 / 32 * 8 = 1 / 4
    P5 = layers.UpSampling2D(size=(8, 8))(
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(in5))
    # 1 / 16 * 4 = 1 / 4
    out4 = layers.Add()([in4, layers.UpSampling2D(size=(2, 2))(in5)])
    P4 = layers.UpSampling2D(size=(4, 4))(
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out4))
    # 1 / 8 * 2 = 1 / 4
    out3 = layers.Add()([in3, layers.UpSampling2D(size=(2, 2))(out4)])
    P3 = layers.UpSampling2D(size=(2, 2))(
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out3))
    # 1 / 4
    P2 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(
        layers.Add()([in2, layers.UpSampling2D(size=(2, 2))(out3)]))
    # (b, /4, /4, 256)

    print("P shapes")
    print(P2.shape)
    print(P3.shape)
    print(P4.shape)
    print(P5.shape)
    fuse = layers.Concatenate()([P2, P3, P4, P5])

    # probability map
    p = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
    p = layers.BatchNormalization()(p)
    p = layers.ReLU()(p)
    p = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(p)
    p = layers.BatchNormalization()(p)
    p = layers.ReLU()(p)
    p = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                               activation='sigmoid')(p)

    training_model = models.Model(inputs=image_input,
                                  outputs = p
                                  )

    prediction_model = models.Model(inputs=image_input, outputs=p)
    print(training_model.summary())
    return training_model, prediction_model


if __name__ == '__main__':
    model, _ = my_resnet_labels()
    # model.summary()
