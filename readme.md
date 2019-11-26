## learning rate:
    adam要比sgd小一个数量级

## backbone：
    mnist不建议pad／resize之后用resnet50，不fit
    目前这个三个conv block的结构很fit，2-3个epoch就双90了

## dataset:
    triplet用mnist测试有效，但是换到我的X光片数据上很难收敛。一方面原因是类别较多，图片尺寸较大，GPU有限因此开的batch比较小。不知道这个现象是不是就是原paper中提到的FLOPS vs. Accuracy trade-off。

    todo：用小图像多类别（cifar10）和大图像小类别（imageNet）再实验一下。

## 新加入triple-center-model:
    原论文说模型不用分类分支也能训(未验证)，但是initial stage缺乏guidance
    实测triple center loss太容易变成0了，后面就剩下classification loss在驱动
    猜测一个原因：目前center loss用的embedding layer，没有自己的学习率

    todo：换custom defined layer试一下

## 新加入lossless TCL model:
    也是针对triple center loss太容易变成0这个问题，这样l2loss始终大于0
    这个模型cls branch快速收敛，lossless tcl loss收敛比较慢，尝试tf.Print一些中间变量（distances）
    lossless tcl可视化的结果有点诡异，二维scatter呈现线形，在x光片数据上也是这样的，目前没想到怎么解释。
    一个猜测：我们对embedding做sigmoid norm的时候，相当于把embedding限定到一个hypertube上一个点，高维点-->低维线？？

## loss_weights:
    loss_weights affects conversion
    log loss also affects conversion

## done:
    验证噪声样本在这几个模型上的分布，单独的metric learning不足以generalizing new samples
    高维分布和低维分布之间没有直接可解释的关系，
    噪声样本分布在训练样本之间，没有隔离开
    将噪声样本加入训练是比较有效的抗干扰方法

## todo:
    关于noisy samples：
    1. added with random labels
    2. added with all-negative labels & sigmoid


    




