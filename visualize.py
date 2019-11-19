from models import *
from keras.utils import to_categorical
import keras.backend as K
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects



def scatter(x, labels, n_classes=10, file_name='scatter.png'):
    # choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", n_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit.
    txts = []
    classes = np.unique(labels)
    for i in classes:
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)


    plt.savefig(file_name)


if __name__ == '__main__':

    test_path = "data/test/"
    target_size = 28
    n_classes = 3

    x_test, y_test = loadData(test_path, target_size)
    x_test = np.reshape(x_test, (-1, target_size, target_size, 1))

    # # raw cls model
    # model = cls_model(input_shape=(target_size,target_size,1), n_classes=n_classes)
    # model.load_weights('raw_cls_02_val_acc_0.940.h5', by_name=True)
    # func = K.function(inputs=[model.input], outputs=[model.layers[-2].get_output_at(-1)])
    # y_pred = func([x_test])[0]
    # print(y_pred.shape)           # (N, 100)
    # print("visualizing...")
    # tsne = TSNE()
    # tsne_embeds = tsne.fit_transform(y_pred)
    # print(tsne_embeds.shape)      # (N, 2)
    # scatter(tsne_embeds, y_test, n_classes, 'raw_scatter.png')

    # triplet model
    model = triple_model(input_shape=(target_size,target_size,1), n_classes=n_classes)
    model.load_weights('tripleNet_01_val_acc_0.971.h5', by_name=True)
    func = K.function(inputs=[model.inputs[0]], outputs=[model.get_layer('activation_1').output])
    y_pred = func([x_test])[0]
    print(y_pred.shape)
    print("visualizing...")
    tsne = TSNE()
    tsne_embeds = tsne.fit_transform(y_pred)
    print(tsne_embeds.shape)      # (N, 2)
    scatter(tsne_embeds, y_test, n_classes, 'tripleNet_scatter.png')





