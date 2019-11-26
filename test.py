from models import *
from triple_center_model import *


if __name__ == '__main__':

    test_path = "data/test/"
    n_classes = 3
    target_size = 28

    # model = cls_model(input_shape=(target_size,target_size,1), n_classes=n_classes)
    # model.load_weights('raw_cls_04_val_acc_1.000.h5', by_name=True)
    # model = triple_model(input_shape=(target_size,target_size,1), n_classes=n_classes)
    # model.load_weights('tripleNet_01_val_acc_0.971.h5', by_name=True)
    # model = triple_center_model(input_shape=(target_size,target_size,1), n_classes=n_classes)
    # model.load_weights('triple_center_model_01_val_acc_0.981.h5', by_name=True)
    model = lossless_tcl_model(input_shape=(target_size,target_size,1), n_classes=n_classes)
    model.load_weights('weights/unlinear_model_27_val_acc_1.000.h5', by_name=True)

    cate = [i for i in glob.glob(test_path+"/*") if i[0]!='.']
    print(cate)

    # use cls branch
    cate = ['foreign/']
    for each_cls in cate:
        cls = each_cls.split('/')[-1]
        print("current class: ", cls)
        for file in os.listdir(each_cls)[:5]:
            img = cv2.imread(each_cls + '/' + file, 0)
            img = cv2.resize(img, (target_size, target_size))
            tmp = np.reshape(img, (1, target_size, target_size, 1))

            # preds = model.predict(tmp)
            # preds = model.predict([tmp, tmp, tmp])[0]
            preds = model.predict([tmp, np.array([1])])[0]
            print(preds)

            label = np.argmax(preds)
            print(label)

    # # use metric branch
    # train_path = "data/train/"
    # func = K.function(inputs=[], outputs=[model.get_layer('activation_2').output])
    # centers = func([])[0]        # (n, 1, 100)
    # thresholds = []
    # for i, each_cls in enumerate(cate):
    #     center = centers[i,...]
    #     x_train, y_train = loadData(train_path, target_size=target_size, folder=each_cls.split("/")[-1])
    #     x_train = np.expand_dims(x_train, axis=-1)
    #     y_train = np.reshape(y_train, (y_train.shape[0], 1))
    #     func = K.function(inputs=model.inputs, outputs=[model.get_layer('l2distance').output[0]])
    #     intra_distances = func([x_train, y_train])[0].squeeze().tolist()
    #     intra_distances.sort()
    #     print(intra_distances[:10])
    #     threshold = intra_distances[int(y_train.shape[0]*0.9)]
    #     thresholds.append(threshold)

    # foreign_path = "foreign/"
    # for file in os.listdir(foreign_path)[:50]:
    #     img = cv2.imread(foreign_path + '/' + file, 0)
    #     img = cv2.resize(img, (target_size, target_size))
    #     tmp = np.reshape(img, (1, target_size, target_size, 1))
    #     func = K.function(inputs=[model.inputs[0]], outputs=[model.get_layer('model_1').get_output_at(-1)])
    #     dense = func([tmp])[0]
    #     distances = []
    #     for i in range(3):
    #         center = centers[i]
    #         # print(center)
    #         distance = np.sum(np.square(center-dense), axis=1)
    #         distances.append(distance[0])

    #     print("thresholds:   ", thresholds)
    #     print("distances: ", distances)







