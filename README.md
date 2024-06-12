## Test Env

- Pytorch 1.7.1+

- Python 3.6+

## Framework Start

- [环境搭建](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/Environment_setting.md)

- [数据集准备](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/Data_preparing.md)

- [配置文件解释](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/Configs_description.md)

- [训练](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/How_to_train.md)

- [模型评估](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/How_to_eval.md)

- [计算 Flops\&Params](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/Calculate_Flops.md)

- [添加新的模型组件](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/Add_modules.md)

- [类别激活图可视化](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/CAM_visualization.md)

- [学习率策略可视化](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/Lr_visualization.md)

- [数据增强策略可视化](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/Pipeline_visualization.md)

## 运行 cmd

拿到新数据后放到 datas/rgb 文件夹下，要先删除标签信息，运行命令，然后在 datas/format 文件夹下生成不带标签的数据

```markup
python tools/format_MSImg.py
```

生成 K-fold 交叉验证数据，默认 5 折，运行完后会在 datas/文件夹下生成 train1/test1.txt train2/test2.txt ....

```markup
python tools/split_data.py
```

训练/验证/可视化模型

```markup
# 用第一块GPU训练PartNet模型的第4折验证
CUDA_VISIBLE_DEVICES=1 python tools/train.py models/resnet/MS_ResWeightedPartNet50_FL.py --kflod-validation 4


# 验证第0折，注意，要把 models/resnet/MS_ResWeightedPartNet50_FL.py  中的test=dict()换成 WeightedPartFLfold0
python tools/evaluation.py models/resnet/MS_ResWeightedPartNet50_FL.py --kflod-validation 0

# 可视化cam图
python tools/vis_cam.py /code_path/MassSpectrumCls/datasets/test/DNK/ models/resnet/MS_ResWeightedPartNet50_FL.py --save-path /code_path/MassSpectrumCls/logs/ResPartNet/WeightedPartFLfold4/
```


## Reference

    @repo{2020mmclassification,
        title={OpenMMLab's Image Classification Toolbox and Benchmark},
        author={MMClassification Contributors},
        howpublished = {\url{https://github.com/open-mmlab/mmclassification}},
        year={2020}
    }

    this repo is based on [repo](https://github.com/Fafa-DL/Awesome-Backbones)
