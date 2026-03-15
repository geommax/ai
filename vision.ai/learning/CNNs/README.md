## AI Vision README (CNN Focus)

ဒီ README မှာ CNN models တွေရဲ့ reference links, PyTorch နဲ့အသုံးပြုနိုင်မယ့် model families, နဲ့ model history (အသုံးပြုမှု + training datasets) ကို စုစည်းထားပါတယ်။

---

## 1) CNN Models + Links (Core References)

### Classic CNNs
- LeNet-5 (1998)  
	Link: https://ieeexplore.ieee.org/document/726791  
	Typical usage: Digit recognition

- AlexNet (2012)  
	Link: https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html  
	Typical usage: Large-scale image classification

- ZFNet (2013)  
	Link: https://arxiv.org/abs/1311.2901  
	Typical usage: Feature visualization + classification improvements

- VGG (2014)  
	Link: https://arxiv.org/abs/1409.1556  
	Typical usage: Transfer learning backbone

- GoogLeNet / Inception v1 (2014)  
	Link: https://arxiv.org/abs/1409.4842  
	Typical usage: Efficient deep classification

- Inception v3 (2015)  
	Link: https://arxiv.org/abs/1512.00567  
	Typical usage: Classification with factorized convolutions

- ResNet (2015)  
	Link: https://arxiv.org/abs/1512.03385  
	Typical usage: Very deep networks via residual connections

- DenseNet (2016)  
	Link: https://arxiv.org/abs/1608.06993  
	Typical usage: Parameter-efficient feature reuse

- Xception (2016)  
	Link: https://arxiv.org/abs/1610.02357  
	Typical usage: Depthwise separable convolution-based classification

- MobileNet v1/v2/v3 (2017–2019)  
	Links:  
	- v1: https://arxiv.org/abs/1704.04861  
	- v2: https://arxiv.org/abs/1801.04381  
	- v3: https://arxiv.org/abs/1905.02244  
	Typical usage: Edge/mobile inference

- EfficientNet (2019)  
	Link: https://arxiv.org/abs/1905.11946  
	Typical usage: Compound scaling for better accuracy/efficiency

- ConvNeXt (2022)  
	Link: https://arxiv.org/abs/2201.03545  
	Typical usage: Modernized pure CNN alternative to ViT backbones

---

## 2) PyTorch မှာ အသုံးများတဲ့ CNN Model Families

PyTorch official models (`torchvision.models`) မှာ အောက်ပါ CNN တွေကို pretrained weights နဲ့တစ်ခါတည်း သုံးလို့ရပါတယ်။

- `alexnet`
- `vgg11`, `vgg13`, `vgg16`, `vgg19` (BN variants included)
- `googlenet`
- `inception_v3`
- `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- `resnext50_32x4d`, `resnext101_32x8d`
- `wide_resnet50_2`, `wide_resnet101_2`
- `densenet121`, `densenet161`, `densenet169`, `densenet201`
- `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`
- `mnasnet0_5`, `mnasnet0_75`, `mnasnet1_0`, `mnasnet1_3`
- `shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x1_5`, `shufflenet_v2_x2_0`
- `efficientnet_b0` ~ `efficientnet_b7`, `efficientnet_v2_s/m/l`
- `regnet_y_*`, `regnet_x_*`
- `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`

Official docs: https://pytorch.org/vision/stable/models.html

---

## 3) History: CNNs, Usage, Training Datasets (High-Level)

| Era | Models | Main Usage | Common Training Datasets |
|---|---|---|---|
| 1998–2011 | LeNet family | Handwritten digit classification | MNIST |
| 2012 | AlexNet | Large-scale image classification breakthrough | ImageNet-1K |
| 2013–2014 | ZFNet, VGG, GoogLeNet | Deeper feature extraction, transfer learning | ImageNet-1K |
| 2015–2017 | ResNet, Inception v3/v4, DenseNet | Very deep training, robust generalization | ImageNet-1K (often fine-tuned on CIFAR/Caltech/custom data) |
| 2017–2019 | MobileNet, ShuffleNet, NAS families, EfficientNet | Edge/mobile deployment, speed/accuracy trade-off | ImageNet-1K, sometimes COCO/OpenImages fine-tuning for detection |
| 2020–2022 | EfficientNetV2, RegNet, ConvNeXt | Modern scalable CNN backbones for classification/detection/segmentation | ImageNet-1K/21K, downstream COCO, ADE20K, task-specific datasets |
| 2023+ | CNN + Hybrid pipelines | Production systems (latency-sensitive tasks), industrial CV | Domain datasets (medical, satellite, OCR, manufacturing, retail) + transfer from ImageNet-pretrained weights |

---

## 4) CNN Usage Patterns (Practical)

- Classification: ResNet / EfficientNet / ConvNeXt
- Detection backbone: ResNet / MobileNet / ConvNeXt (with Faster R-CNN, RetinaNet, YOLO variants)
- Segmentation backbone: ResNet / EfficientNet / ConvNeXt (with U-Net, DeepLab, Mask2Former style heads)
- Edge deployment: MobileNet / ShuffleNet / EfficientNet-Lite style models
- Feature extraction / transfer learning: VGG, ResNet, DenseNet

---

## 5) Recommended Starting Set (PyTorch)

Project အစမှာ baseline အနေနဲ့ model 3 မျိုးထားပြီး compare လုပ်ရင် practical ဖြစ်ပါတယ်။

1. `resnet50` (strong stable baseline)  
2. `efficientnet_b0` (efficiency-friendly)  
3. `convnext_tiny` (modern CNN baseline)

Datasets (quick experiments):
- CIFAR-10 / CIFAR-100
- Tiny-ImageNet
- Custom dataset (train/val/test split)

Production pretraining reference:
- ImageNet pretrained weights (torchvision)

---

## 6) Notes

- TorchVision pretrained models ကို transfer learning နဲ့ စတင်ပြီး data domain-specific fine-tuning လုပ်တာက အလုပ်တွင်ကျယ်ပါတယ်။
- Dataset size နည်းရင် augmentation + regularization strategy တွေကိုအရေးကြီးစွာထည့်သွင်းသုံးသင့်ပါတယ်။
- Edge deployment target ရှိရင် FLOPs/latency/parameter size ကို accuracy နဲ့တွဲပြီး evaluate လုပ်ပါ။
