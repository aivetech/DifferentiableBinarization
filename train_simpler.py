import datetime
import os.path as osp
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras.utils import get_file
import os

from losses import simpler_loss
from generator import generate, generate_simpler_model
from model import dbnet
from model_simpler import db_simpler
checkpoints_dir = f'checkpoints/{datetime.date.today()}'

batch_size = 2

if not osp.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

train_generator = generate_simpler_model('datasets/total_text_subsample', batch_size=batch_size, is_training=True)
val_generator = generate_simpler_model('datasets/total_text_subsample', batch_size=batch_size, is_training=False)

model, prediction_model = db_simpler()

model.compile(optimizer=optimizers.Adam(lr=1e-3), loss={'conv2d_transpose_1': lambda y_true, y_pred: simpler_loss(y_pred, y_true)})
checkpoint = callbacks.ModelCheckpoint(
    osp.join(checkpoints_dir, 'simpler_{epoch:02d}_{loss:.4f}_{val_loss:.4f}.h5'),
    verbose=1,
)
model.fit(
    x=train_generator,
    steps_per_epoch=2,
    initial_epoch=0,
    epochs=1,
    verbose=1,
    callbacks=[checkpoint],
    validation_data=val_generator,
    validation_steps=2
)

