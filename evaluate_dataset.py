import argparse
import os

import PIL.Image as Image
import torch
from tqdm import tqdm

from model_factory import ModelFactory
from datasets import load_dataset
import pandas as pd


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 evaluation script")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="folder where data is located. test_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model",
        type=str,
        metavar="M",
        help="the model file to be evaluated. Usually it is of the form model_X.pth",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="experiment/kaggle.csv",
        metavar="D",
        help="name of the output csv file",
    )
    args = parser.parse_args()
    return args

'''
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")
'''

def map_labels(example):
    example['visual_embedding'] = torch.tensor(example['visual_embedding'])
    example['textual_embedding'] = torch.tensor(example['textual_embedding'])
    return example

def main() -> None:
    """Main Function."""
    # options
    args = opts()
    #test_dir = args.data + "/test_images/mistery_category"

    # cuda
    use_cuda = torch.cuda.is_available()

    # load model and transform
    state_dict = torch.load(args.model)
    model = ModelFactory(args.model_name).get_model()
    model.load_state_dict(state_dict)
    model.eval()
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    ds_test = load_dataset("cuevascarlos/ImageNet-Sketch-Embed", split="test")
    #ds_test_f = ds_test.map(map_labels)

    output_file = open(args.outfile, "w")
    output_file.write("Id,Category\n")
    for i in tqdm(range(len(ds_test))):
        visual_embed = torch.tensor(ds_test[i]['visual_embedding']).unsqueeze(0)
        textual_embed = torch.tensor(ds_test[i]['textual_embedding']).unsqueeze(0)
        if use_cuda:
            visual_embed = visual_embed.cuda()
            textual_embed = textual_embed.cuda()
        output = model(visual_embed, textual_embed)
        #print(output.shape)
        pred = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (ds_test[i]['image_path'][67:-5], pred))


    output_file.close()

    print(
        "Succesfully wrote "
        + args.outfile
        + ", you can upload this file to the kaggle competition website"
    )


if __name__ == "__main__":
    main()
