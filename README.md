## Test Env

- Pytorch 1.7.1+

- Python 3.6+

## Framework Start

- [build environment](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/Environment_setting.md)

- [prepare dataset](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/Data_preparing.md)

- [config file](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/Configs_description.md)

- [train model](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/How_to_train.md)

- [evaluate model](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/How_to_eval.md)

- [compute Flops\&Params](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/Calculate_Flops.md)

- [add new model](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/Add_modules.md)

- [visualize the CAM](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/CAM_visualization.md)

- [visualize the Loss value](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/Lr_visualization.md)

- [Data Augment](https://github.com/whisperH/ProteoNet/blob/main/datas/docs/Pipeline_visualization.md)

## run cmd

After getting the new data, put it in the datas/rgb folder, delete the label information first, run the command, and then generate unlabeled data in the datas/format folder

```markup
python tools/format_MSImg.py
```

Generate K-fold cross validation data, the default is 5 folds. After running, train1/test1.txt train2/test2.txt ... will be generated in the datas/ folder.

```markup
python tools/split_data.py
```

训练/验证/可视化模型

```markup
# Use the first GPU to train the 4th validation fold of the PartNet model
CUDA_VISIBLE_DEVICES=1 python tools/train.py models/resnet/MS_ResWeightedPartNet50_FL.py --kflod-validation 4


# Verify the 0th fold. Note that you need to replace test=dict() in models/resnet/MS_ResWeightedPartNet50_FL.py with WeightedPartFLfold0
python tools/evaluation.py models/resnet/MS_ResWeightedPartNet50_FL.py --kflod-validation 0

# Visual CAM picture
python tools/vis_cam.py /code_path/MassSpectrumCls/datasets/test/DNK/ models/resnet/MS_ResWeightedPartNet50_FL.py --save-path /code_path/MassSpectrumCls/logs/ResPartNet/WeightedPartFLfold4/
```

# weights file
Kidney dataset: https://drive.google.com/drive/folders/1GvZXIyypbd7_nT8kc7MS7trY1-37MWgC?usp=drive_link


## Reference

    @repo{2020mmclassification,
        title={OpenMMLab's Image Classification Toolbox and Benchmark},
        author={MMClassification Contributors},
        howpublished = {\url{https://github.com/open-mmlab/mmclassification}},
        year={2020}
    }

    this repo is based on [repo](https://github.com/Fafa-DL/Awesome-Backbones)
