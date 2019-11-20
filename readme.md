## learning rate:
    adam要比sgd小一个数量级

## backbone：
    mnist不建议pad／resize之后用resnet50，不fit
    目前这个三个conv block的结构很fit，2-3个epoch就双90了

## dataset:
    用mnist测试有效，但是换到我的X光片数据上很难收敛。一方面原因是类别较多，图片尺寸较大，GPU有限因此开的batch比较小。
    不知道这个现象是不是就是原paper中提到的FLOPS vs. Accuracy trade-off。

    todo：用小图像多类别（cifar10）和大图像小类别（imageNet）再实验一下。