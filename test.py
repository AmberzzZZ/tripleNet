from models import *
from triple_center_model import *


if __name__ == '__main__':

    test_path = "data/train/"
    n_classes = 3
    target_size = 28

    # model = cls_model(input_shape=(target_size,target_size,1), n_classes=n_classes)
    # model.load_weights('raw_cls_04_val_acc_1.000.h5', by_name=True)
    # model = triple_model(input_shape=(target_size,target_size,1), n_classes=n_classes)
    # model.load_weights('tripleNet_01_val_acc_0.971.h5', by_name=True)
    # model = triple_center_model(input_shape=(target_size,target_size,1), n_classes=n_classes)
    # model.load_weights('triple_center_model_01_val_acc_0.981.h5', by_name=True)
    model = lossless_tcl_model(input_shape=(target_size,target_size,1), n_classes=n_classes)
    model.load_weights('lossless_tcl_model_02_val_acc_0.995.h5', by_name=True)

    cate = [i for i in glob.glob(test_path+"/*") if i[0]!='.']
    print(cate)

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
            # print(preds)

            label = np.argmax(preds)
            print(label)


