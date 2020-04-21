#from keras_resnet.models import ResNet50
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from losses import db_loss, dummy_loss
import pydot

def my_resnet(input_size=640, k=50):
    image_input = layers.Input(shape=(None, None, 3))
    backbone = ResNet50(include_top=False, weights='imagenet', input_tensor = image_input)
    out = tf.keras.layers.Dense(1)(backbone.output)

    loss = tf.compat.v1.keras.layers.Lambda(dummy_loss, name='dummy_loss', output_shape = (None, None, None, 1))([out])
    training_model = models.Model(inputs=[image_input],
                                  outputs=loss)
    prediction_model = models.Model(inputs=image_input, outputs=out)
    print(training_model.summary())
    return training_model, prediction_model


if __name__ == '__main__':
    model, _ = my_resnet()
    # model.summary()
