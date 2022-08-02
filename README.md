# Xircuits Project Template

This repo comprises of the training, inference and model conversion (pytorch model to onnx model) code for UNet masking model using Xircuits.

## Prerequisites

A project may have prerequisites such as models that needs to be downloaded or non-python related setup. You may list them down here.

http://www.josiahwang.com/dataset/leedsbutterfly/leedsbutterfly_dataset_v1.0.zip

```
import os
for folder in os.listdir("segmentations"):
    path = os.path.join("segmentations", folder)
    for mask_image in os.listdir(path):
        old_path = os.path.join(path, mask_image)
        new_path = os.path.join(path, mask_image.replace("_seg0",""))
        os.rename(old_path, new_path)
```

## Installation

```
pip install -r requirements.txt
```

**Description**

- This repo comprises of the training, inference and model conversion (pytorch model to onnx model) code for UNet masking model.



- Added the component required for the training and inferencing.
1. ConvertTorchModelToOnnx
2. CreateUnetModel
3. ImageTrainTestSplit
4. PrepareUnetDataLoader
5. TrainUnet
6. UNetModel
7. UnetPredict
8. ReadMaskDataset

With 3 workflows:
1. PyTorchUnetInferenceSample.xircuits - To allow inferencing of trained Unet model in either pytorch or onnx format.
2. PyTorchUnetTrainSample.xircuits - To allow training of binary class masking with Unet model
3. PyTorchToOnnxSample.xircuits - To allow conversion of torch model to onnx model.

**To Test**
1. To perform the training in PyTorchUnetTrainSample.xircuits. Please download any mask dataset that contains the images and segmentation images. 
- The dataset that I used would be Leeds butterfly dataset and ensure the tree follows as below:
![image](https://user-images.githubusercontent.com/23378929/146125972-7e13c99e-6c7d-474a-ad8d-b3c7a12ef586.png)
- The trained model should be saved in the examples folder by default but you are allowed to modify the save path.
- To allow it to work, you must have the same image name with the image and mask image. The prepare butterfly dataset component will do the setup for you.

2. To perform the inference code in PyTorchUnetInferenceSample.xircuits. Please put the path that model has be saved as model_path and inference image as image_path.

3.  To perform the model conversion code in PytorchToOnnxSample.xircuits. Please put the pth format model as input_model_path and onnx format path as output_model_path.

Reference:
Citing Leeds buttefly dataset
Josiah Wang, Katja Markert, and Mark Everingham
Learning Models for Object Recognition from Natural Language Descriptions
In Proceedings of the 20th British Machine Vision Conference (BMVC2009)
> http://www.josiahwang.com/dataset/leedsbutterfly/
