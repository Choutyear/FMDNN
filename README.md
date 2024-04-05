# FMDNN
FMDNN: A Fuzzy-guided Multi-granular Deep Neural Network for Histopathological Image Classification

## 1. Project Overview

As shown in the figure below, FMDNN consists of three modules, **Multi-granular Feature Extraction Module** conducts feature extraction on the input image at three distinct granularities, **Universal Fuzzy Feature Module** extracts the universal fuzzy feature of the image, and **Fuzzy-guided Cross-attention Module** performs feature fusion by linear transformation and dimension alignment to get the final classification result.

![image](https://github.com/Choutyear/FMDNN/blob/main/Figs/Fig1.png)


## 2. Environment Setup

To install requirements:

```conda env create -f environment.yaml```

[environment.yaml](https://github.com/Choutyear/FMDNN/blob/main/Files/encironment.yaml)


In training, we will use pre-trained weights, which you can import through the following code.

```from model import vit_base_patch16_224_in21k as create_model```

[Pre-trained weights](https://github.com/google-research/vision_transformer)

## 3. Datasets

* The Lung and Colon Cancer Histopathological Images [LC](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
* NCT-CRC-HE-100K [NCT](https://paperswithcode.com/dataset/nct-crc-he-100k)
* APTOS 2019 Blindness Detection [Bl](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
* HAM10000 [HAM](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
* Kvasir [Kv](https://datasets.simula.no/kvasir/)

\* Note: Before training starts, in all data set folders, each category of disease images needs to be placed in a subfolder.

## 4. Training



## Reference

Some of the codes are borrowed from:
* [ViT_1](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py)
* [ViT_2](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer)
* [CrossViT](https://github.com/IBM/CrossViT)

