# FMDNN
FMDNN: A Fuzzy-guided Multi-granular Deep Neural Network for Histopathological Image Classification

## 1. Project Overview

As shown in the figure below, FMDNN consists of three modules, **Multi-granular Feature Extraction Module** conducts feature extraction on the input image at three distinct granularities, **Universal Fuzzy Feature Module** extracts the universal fuzzy feature of the image, and **Fuzzy-guided Cross-attention Module** performs feature fusion by linear transformation and dimension alignment to get the final classification result.

![image](https://github.com/Choutyear/FMDNN/blob/main/Figs/Fig1.png)


## 2. Environment Setup
[environment.yaml](https://github.com/Choutyear/FMDNN/blob/main/Files/encironment.yaml)
## 3. Code

## 4. Dataset

## 5. Training

## Reference

Some of the codes are borrowed from:
* [ViT_1](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py)
* [ViT_2](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer)
* [CrossViT](https://github.com/IBM/CrossViT)
* [Pre-trained weights](https://github.com/google-research/vision_transformer)
