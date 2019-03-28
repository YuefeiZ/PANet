# PANet
这个网络模型，是在Mask RCNN的基础上，根据PANet这篇论文，来实现这个模型，Mask rcnn的模型完全来自于https://github.com/matterport/Mask_RCNN ，PANet论文地址：https://arxiv.org/abs/1803.01534

在相同的公路数据上做了测试，map为:

Mask RCNN： "logs\\blockcrack_crop\\blockcrack20190107T2155\\mask_rcnn_blockcrack_0048.h5" 0.921039480275

PANet： "logs\\blockcrack_crop_panet\\blockcrack20190110T2019\\panet_blockcrack_0047.h5" 0.928535732149

Mask RCNN的模型在mrcnn/model.py定义。PANet的模型在mrcnn/panetmodel.py中定义。
在mrcnn文件中的panetmodelAug.py是在Mask RCNN模型上加了PANet的第一步改进。panetmodelAugFF.py是加入了Mask RCNN模型上加入了PANet的第一步和第三步改进。
