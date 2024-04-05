# FMDNN
FMDNN: A Fuzzy-guided Multi-granular Deep Neural Network for Histopathological Image Classification

**Thank you very much for your interest in our work! This is the code of FMDNN.**

## 1. Project Overview

As shown in the figure below, FMDNN consists of three modules, **Multi-granular Feature Extraction Module** conducts feature extraction on the input image at three distinct granularities, **Universal Fuzzy Feature Module** extracts the universal fuzzy feature of the image, and **Fuzzy-guided Cross-attention Module** performs feature fusion by linear transformation and dimension alignment to get the final classification result.

![image](https://github.com/Choutyear/FMDNN/blob/main/Figs/Fig1.png)

<br>

## 2. Environment Setup

You can install the environment with the following code:

```conda env create -f environment.yaml```

[environment.yaml](https://github.com/Choutyear/FMDNN/blob/main/Files/encironment.yaml)

<br>

In training, we will use pre-trained weights, which you can import through the following code.

```from model import vit_base_patch16_224_in21k as create_model```

[Pre-trained weights](https://github.com/google-research/vision_transformer)

<br>

## 3. Datasets

* The Lung and Colon Cancer Histopathological Images [LC](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
* NCT-CRC-HE-100K [NCT](https://paperswithcode.com/dataset/nct-crc-he-100k)
* APTOS 2019 Blindness Detection [Bl](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
* HAM10000 [HAM](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
* Kvasir [Kv](https://datasets.simula.no/kvasir/)

\* Note: Before training starts, in all data set folders, each category of disease images needs to be placed in a subfolder.

<br>

## 4. Training

1. This code is responsible for processing data. First determine the device used (GPU or CPU), then check if the folder where the model weights are stored exists and create it if it does not exist. Then use `SummaryWriter` to create a TensorBoard visualization object. Then, call the `read_split_data` function to read the image paths and corresponding labels of the training set and verification set, and use `torchvision.transforms` to define the data preprocessing method.
```python
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    tb_writer = SummaryWriter()
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
```

<br>

2. Then instantiate the training data set and validation data set, use the custom data set class `MyDataSet`, and pass in the image path, label and data preprocessing method.
```python
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
```

<br>

3. Define mode
```python
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
```

<br>

4. Load the pretrained weights and apply them to the model.
```python
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
```

<br>

5. Define an SGD optimizer and a LambdaLR learning rate scheduler, and set the learning rate update strategy.
```python
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
```

<br>

6. This code contains the training and validation loop of the model, in which the model is trained through the `train_one_epoch` function and the model is verified using the `evaluate` function. During the training loop, the learning rate is also updated, the ROC curve is plotted via the `plot_roc_curve` function, and the metrics during training are written to `TensorBoard`.
```python
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc, train_recall, train_specificity, train_precision = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc, val_recall, val_specificity, val_precision, all_labels, all_pred_scores = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch)

        # class!!!
        if epoch % 10 == 0:
            plot_roc_curve(all_labels, all_pred_scores, , epoch)  # num_classes

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
```

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default= )
    parser.add_argument('--epochs', type=int, default= )
    parser.add_argument('--batch-size', type=int, default= )  #
    parser.add_argument('--lr', type=float, default= )  #
    parser.add_argument('--lrf', type=float, default= )  #
    parser.add_argument('--data-path', type=str, default="  ")
    parser.add_argument('--weights', type=str, default='  ',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
```

<br>

## References

Some of the codes are borrowed from:
* [ViT_1](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py)
* [ViT_2](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer)
* [CrossViT](https://github.com/IBM/CrossViT)

