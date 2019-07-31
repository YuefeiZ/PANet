# PANet
这个网络模型，是在Mask RCNN的基础上，根据PANet这篇论文，来实现这个模型，Mask rcnn的模型来自于https://github.com/matterport/Mask_RCNN ，PANet论文地址：https://arxiv.org/abs/1803.01534

在相同的公路数据上做了测试，map为:

Mask RCNN：AP50: 93.3

PANet_AFN：AP50 :94.875

PANet:  AP50:97.6

Mask RCNN的模型在mrcnn/model.py定义。PANet的模型在mrcnn/panetmodel.py中定义。

