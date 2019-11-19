from dataLoader import *
from models import *


if __name__ == '__main__':

    train_path = "data/train/"
    val_path = "data/val/"
    n_classes = 3
    target_size = 28

    # # train 1 : raw cls model
    # batch_size = 256

    # x_train, y_train = loadData(train_path, target_size)
    # print(x_train.shape, y_train.shape)
    # train_generator = base_generator(x_train, y_train, batch_size, n_classes)
    # x_val, y_val = loadData(val_path, target_size)
    # val_generator = base_generator(x_val, y_val, batch_size, n_classes)

    # model = cls_model(lr=5e-3, input_shape=(target_size,target_size,1), n_classes=n_classes)

    # # train 2 : use pretained weight
    # # model.load_weights("~/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)

    # filepath = "./raw_cls_{epoch:02d}_val_acc_{val_acc:.3f}.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # model.fit_generator(generator=train_generator,
    #                     steps_per_epoch=40,
    #                     epochs=100,
    #                     validation_data=val_generator,
    #                     validation_steps=4,
    #                     verbose=1, callbacks=[checkpoint])


    # train 3 : triplet model
    batch_size = 16

    x_train, y_train = loadData(train_path, target_size)
    train_generator = triplet_generator(x_train, y_train, batch_size, n_classes=n_classes)
    x_val, y_val = loadData(val_path, target_size)
    val_generator = triplet_generator(x_val, y_val, batch_size, n_classes=n_classes)

    model = triple_model(input_shape=(target_size,target_size,1), n_classes=n_classes)

    filepath = "./tripleNet_{epoch:02d}_val_acc_{dense_2_acc:.3f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=20,
                        epochs=100,
                        validation_data=val_generator,
                        validation_steps=2,
                        verbose=1, callbacks=[checkpoint])


    # train 4: triple-center-loss model


