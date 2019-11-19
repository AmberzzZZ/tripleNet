from keras.layers import GlobalAveragePooling2D, Dense, Input, concatenate, Activation, \
                         Conv2D, MaxPooling2D, BatchNormalization, PReLU, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD, Adam
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_utils
import keras.backend as K
from dataLoader import *
import os

os.environ["CUDA_VISABLE_DEVICES"] = "2, 3, 4, 7"
GPU_COUNT = 4


def cls_acc(y_true, y_pred):
    return categorical_accuracy(y_true, y_pred)


def cls_loss(y_true, y_pred):
    print("for cls branch, y_pred.shape:  ", y_pred)       # [Batch_dim, num_classes]
    return categorical_crossentropy(y_true, y_pred)


def triplet_loss(y_true, y_pred, alpha=0.4):
    print("for distance branch, y_pred.shape:  ", y_pred)       # [Batch_dim, vec_dim*3]

    vec_len = y_pred.shape.as_list()[-1]

    anchor = y_pred[:, :int(vec_len/3)]
    positve = y_pred[:, int(vec_len/3):int(vec_len*2/3)]
    negative = y_pred[:, int(vec_len*2/3):]

    pos_dist = K.sum(K.square(anchor - positve), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    loss = K.maximum(0., pos_dist - neg_dist + alpha)

    return loss


def conv_block(filters, kernel_size, x):
    x = Conv2D(filters,kernel_size)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters,kernel_size)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    return x


def base_model(input_shape=(512,512,1)):
    input = Input(shape=input_shape)

    # # resnet50
    # base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape)
    # x = base_model(input)
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(100, activation='prelu')(x)

    # mnist
    x = conv_block(32, (3,3), input)
    x = conv_block(64, (5,5), x)
    x = conv_block(128, (7,7), x)
    x = Flatten()(x)
    x = Dense(100)(x)
    x = PReLU()(x)

    model = Model(inputs=input, outputs=x)

    return model


def cls_model(lr=3e-4, input_shape=(512,512,1), n_classes=10):
    input = Input(shape=input_shape)
    basemodel = base_model(input_shape)
    x = basemodel(input)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)

    sgd = SGD(lr, momentum=0.9, decay=1e-6, nesterov=True)
    adam = Adam(lr, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=adam, loss=cls_loss, metrics=['acc'])

    return model


def triple_model(input_shape=(512,512,1), n_classes=10, multi_gpu=False):
    anchor_input = Input(shape=input_shape)
    positive_input = Input(shape=input_shape)
    negative_input = Input(shape=input_shape)

    sharedCNN = base_model(input_shape)
    encoded_anchor = sharedCNN(anchor_input)
    encoded_positive = sharedCNN(positive_input)
    encoded_negative = sharedCNN(negative_input)

    # class branch
    x = Dense(n_classes, activation='softmax')(encoded_anchor)

    # distance branch
    encoded_anchor = Activation('sigmoid')(encoded_anchor)
    encoded_positive = Activation('sigmoid')(encoded_positive)
    encoded_negative = Activation('sigmoid')(encoded_negative)
    merged = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='tripleLossLayer')

    model = Model(inputs=[anchor_input,positive_input,negative_input], outputs=[x, merged])
    if multi_gpu:
        model = multi_gpu_model(model, GPU_COUNT)

    sgd = SGD(lr=3e-4, momentum=0.9, decay=1e-6, nesterov=True)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=adam, loss=[cls_loss, triplet_loss], metrics=['acc'])

    return model


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)


if __name__ == '__main__':

    train_path = "train/"
    val_path = "val/"
    batch_size = 6
    n_classes = 15
    target_size = 224

    x_train, y_train = loadData(train_path, target_size)
    train_generator = triplet_generator(x_train, y_train, batch_size)
    x_val, y_val = loadData(val_path, target_size)
    val_generator = triplet_generator(x_val, y_val, batch_size)

    # for multi GPU
    # model = triple_model(input_shape=(target_size,target_size,1), n_classes=n_classes, multi_gpu=True)
    # for CPU
    model = triple_model(input_shape=(target_size,target_size,1), n_classes=n_classes)

    filepath = "./tripleNet_{epoch:02d}_val_loss_{val_loss:.3f}.h5"
    checkpoint = ParallelModelCheckpoint(model, filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=20,
                        epochs=100,
                        validation_data=val_generator,
                        validation_steps=10,
                        verbose=1, callbacks=[checkpoint])

    











