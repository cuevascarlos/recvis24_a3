## Object recognition and computer vision 2024/2025

**Author:** Carlos Cuevas Villarmin

### Assignment 3: Sketch image classification

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cbuoZf4TzcRoK94EPr3jO0CXRxNQjDt-?usp=sharing)

#### Requirements

1. Install PyTorch from <http://pytorch.org>

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
#Additional packages
pip install transformers==4.45.2
pip install wandb
pip install -U bitsandbytes
pip install datasets
```

#### Dataset

The original dataset consists on 500 different classes of sketches adapted from the [ImageNet Sketch](https://github.com/HaohanWang/ImageNet-Sketch). It can be downloaded images from [here](https://www.kaggle.com/competitions/mva-recvis-2024/data) where the split is given. The test image labels are not provided.

In addition, in this project a new dataset has been created considering text embeddings extracted from generated captions of each image. The new dataset is publicly available [HuggingFace dataset](https://huggingface.co/datasets/cuevascarlos/ImageNet-Sketch-Embed).

It can be generated with

```
python dataset_generation.py --batch-size [default: 64]
```

This script will generate 3 parquet files (train/validation/test) and a dataset saved in `./embeddings_dataset` folder.

Remark: Due to the computational cost of its generation, the dataset has been made publicly available at least until the project has been graded.

#### Training and validating your model

Run the script `main.py` in case the chosen model is DINOv2. It has been adapted to save the progress in wandb in case the argument `--wandb_name [name of the run]` is added.

Model options:
- `--model_name "basic_cnn"`: Naive classifier given by default.
- `--model_name "dino"`: Pre-trained model [DINOv2-base]
- `--model_name "dino-giant"`: Pre-trained model [DINOv2-giant]

Run the script `main_dataset.py` in case the proposed dataset and MLPs are the chosen models.

Model options:
- `--model_name "multi-embeddings"`: The embeddings are concatenated as the input of the MLP.
- `--model_name "double"`: Two independent classifiers with the last layer a weighted sum between the logists of both classifiers.
- `--model_name "fusion"`: Embeddings projected into a pre-defined fusion dimension. Then the projections are concatenated and passed through a MLP to do the classification.

Two different scripts have been defined because in case of using the HuggingFace dataset the DataLoader does not need a transformation. Each of the datasets need a different preprocessing. If statements could have been added to have a unique code but the readability of the code could be more challenging.

#### Evaluating your model on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```bash
#For main.py training procedure
python evaluate.py --data [data_dir] --model [model_file] --model_name [model_name]
#For main_dataset.py training procedure
python evaluate_dataset.py --model [model_file] --model_name [model_name]
```

That generates a file `kaggle.csv` that you can upload to the private kaggle competition website.

#### Acknowledgments

Adapted from Rob Fergus and Soumith Chintala <https://github.com/soumith/traffic-sign-detection-homework>.<br/>
Origial adaptation done by Gul Varol: <https://github.com/gulvarol><br/>
New Sketch dataset and code adaptation done by Ricardo Garcia and Charles Raude: <https://github.com/rjgpinel>, <http://imagine.enpc.fr/~raudec/>
