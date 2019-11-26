from models import *
from keras.layers import Embedding, Lambda
import tensorflow as tf
from keras.utils import to_categorical
from keras.utils import plot_model


def TCL(y_true, y_pred):
    return y_pred


def l2distance(args, n_classes):
    embedding, center_standard, y_true = args
    n_centers = center_standard.shape[0]
    lst = []
    for i in range(n_centers):
        lst.append(K.sum(K.square(embedding-center_standard[i,0,:]), 1, keepdims=True))
    distances = K.concatenate(lst, axis=1)

    classes = K.arange(0, n_classes, dtype=tf.float32)
    y_true = K.repeat_elements(y_true, n_classes, axis=1)
    mask = K.cast(K.equal(y_true, classes), dtype=tf.float32)

    inter_distances = tf.where(tf.equal(mask, 0.0), distances, np.inf*tf.ones_like(mask))
    min_inter_distance = tf.math.reduce_min(inter_distances, axis=1, keepdims=True)

    intra_distances = tf.where(tf.equal(mask, 1.0), distances, np.inf*tf.ones_like(mask))
    intra_distance = tf.math.reduce_min(intra_distances, axis=1, keepdims=True)

    # intra_distance = tf.Print(intra_distance,
    #                             [center_standard],
    #                             message='center_standard: ',
    #                             summarize=300)
    # min_inter_distance = tf.Print(min_inter_distance,
    #                                     [min_inter_distance],
    #                                     message='inter distance: ',
    #                                     summarize=2)

    return [intra_distance, min_inter_distance]


def sharedEmbedding(n_classes, embedding_size, x):
    return Embedding(n_classes, embedding_size)(x)


def triple_center_model(lr=3e-4, input_shape=(512,512,1), n_classes=10, m=4):
    x_input = Input(shape=input_shape)
    basemodel = base_model(input_shape)
    embedding = basemodel(x_input)               # (None,100)

    # cls branch
    softmax = Dense(n_classes, activation='softmax')(embedding)      # dense3

    # center branch
    embedding_size = embedding.shape.as_list()[-1]        # 100: the outdim of dense1
    y_input = Input((1,))
    # ##### past calculation of l2_loss, keep to compare ####
    # center = sharedEmbedding(n_classes, embedding_size, y_input)
    # l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([embedding, center])
    # #####
    labels = np.arange(n_classes).reshape([-1,1])
    y_standard_input = Input(tensor=K.constant(labels))      # (10,1)  assume n_classes=10
    center_standard = sharedEmbedding(n_classes, embedding_size, y_standard_input)   # (10, 1, 100)

    intra_distance, min_inter_distance = Lambda(l2distance, arguments={'n_classes': n_classes},
                                    name='l2distance')([embedding, center_standard, y_input])

    triplet_center_loss = Lambda(lambda x: K.maximum(x[0]+m-x[1],0),
                        name='triple_center_loss')([intra_distance, min_inter_distance])

    model = Model(inputs=[x_input, y_input, y_standard_input], outputs=[softmax, triplet_center_loss])

    sgd = SGD(lr, momentum=0.9, decay=1e-6, nesterov=True)
    adam = Adam(lr, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=adam,
                  loss=['categorical_crossentropy', TCL],
                  metrics=['acc'])   # loss_weights

    return model


def debug(args):
    embedding = args
    embedding = tf.Print(embedding, [K.max(embedding), K.min(embedding)])

    return embedding


def debug_unlinear_loss(args, beta):
    intra_distance, min_inter_distance = args
    epsilon = 1e-7
    l1 = -K.log(-(intra_distance)/beta+1+epsilon)
    l2 = -K.log(-(beta-min_inter_distance)/beta+1+epsilon)
    l1 = tf.Print(l1, [l1], message='l1 debug info: ', summarize=5)
    l2 = tf.Print(l2, [l2], message='l2 debug info: ', summarize=5)

    return l1+l2


def lossless_tcl_model(lr=3e-4, input_shape=(512,512,1), n_classes=10):
    x_input = Input(shape=input_shape)
    basemodel = base_model(input_shape, activation='sigmoid')
    embedding = basemodel(x_input)               # (None,100)

    # cls branch
    softmax = Dense(n_classes, activation='softmax')(embedding)      # dense3

    # center branch
    embedding_size = embedding.shape.as_list()[-1]        # 100: the outdim of dense1
    y_input = Input((1,))
    labels = np.arange(n_classes).reshape([-1,1])
    y_standard_input = Input(tensor=K.constant(labels))      # (10,1)  assume n_classes=10
    center_standard = sharedEmbedding(n_classes, embedding_size, y_standard_input)   # (10, 1, 100)
    center_standard = Activation('sigmoid')(center_standard)

    intra_distance, min_inter_distance = Lambda(l2distance, arguments={'n_classes': n_classes},
                                    name='l2distance')([embedding, center_standard, y_input])
    # intra_distance, min_inter_distance = Lambda(debug)([intra_distance, min_inter_distance])
    # raw
    lossless_tcl_loss = Lambda(lambda x: K.maximum(x[0]+embedding_size-x[1],0),
                        name='lossless_loss')([intra_distance, min_inter_distance])
    # unlinear
    beta = embedding_size
    unlinear_loss = Lambda(lambda x: -K.log(-(x[0])/beta+1+K.epsilon())-K.log(-(beta-x[1])/beta+1+K.epsilon()),
                    name='unlinear_loss')([intra_distance, min_inter_distance])
    # unlinear_loss = Lambda(debug_unlinear_loss, arguments={'beta': beta})([intra_distance, min_inter_distance])

    model = Model(inputs=[x_input, y_input, y_standard_input], outputs=[softmax, unlinear_loss])

    sgd = SGD(lr, momentum=0.9, decay=1e-6, nesterov=True)
    adam = Adam(lr, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=adam,
                  loss=['categorical_crossentropy', TCL],
                  # loss_weights=[1, 3],s
                  metrics=['acc'])   # loss_weights

    return model


if __name__ == '__main__':

    train_path = "data/train/"
    val_path = "data/val/"
    n_classes = 3
    target_size = 28
    batch_size = 128

    x_train, y_train = loadData(train_path, target_size)
    x_train = np.expand_dims(x_train, axis=-1)
    y_train = to_categorical(y_train, num_classes=n_classes)
    print(x_train.shape, y_train.shape)

    model = triple_center_model(lr=3e-4, input_shape=(target_size,target_size,1), n_classes=n_classes)
    # plot_model(model, to_file='triple_center_model.png', show_shapes=True, show_layer_names=True)


    filepath = "./triple_center_model_{epoch:02d}_val_acc_{dense_2_acc:.3f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    y_dummy = np.zeros((x_train.shape[0], 1))

    model.fit(x=[x_train, y_dummy],
              y=[y_train, y_dummy],
              batch_size=batch_size,
              epochs=100, verbose=1,
              callbacks=[checkpoint],
              validation_split=0.2)

    # model.load_weights('triple_center_model_01_val_acc_0.981.h5', by_name=True)

    # img = cv2.imread("data/test/d2/d2_0002.png", 0)
    # img = cv2.resize(img, (target_size, target_size))
    # tmp = np.reshape(img, (1, target_size, target_size, 1))
    # dummy = np.array([1])
    # preds = model.predict([tmp, dummy])[0]
    # print(preds)

    # label = np.argmax(preds)
    # print(label)



