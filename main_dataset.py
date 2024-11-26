import argparse
import os
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from model_factory import ModelFactory

import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch.nn.functional as F

def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        metavar="NW",
        help="number of workers for data loading",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        metavar="WN",
        help="Name for the wandb run",
    )
    args = parser.parse_args()
    return args

def map_labels(example, label_map):
    example['label'] = label_map[example['label']]
    example['visual_embedding'] = torch.tensor(example['visual_embedding'])
    example['textual_embedding'] = torch.tensor(example['textual_embedding'])
    example['label'] = torch.tensor(example['label'])
    return example

def collate_fn(batch):
    labels = torch.tensor([item['label'] for item in batch])
    visual_embeddings = torch.stack([torch.tensor(item['visual_embedding']) for item in batch])
    textual_embeddings = torch.stack([torch.tensor(item['textual_embedding']) for item in batch])
    return visual_embeddings, textual_embeddings, labels

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    args: argparse.ArgumentParser,
) -> None:
    """Default Training Loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        args (argparse.ArgumentParser): Arguments parsed from command line
    """
    model.train()
    correct = 0
    total_loss = 0
    for batch_idx, (data, textual, target) in enumerate(train_loader):
        if use_cuda:
            data, textual, target = data.cuda(), textual.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data, textual)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        if args.model_name == "double":
            loss = criterion(output, target) + 0.3*model.regularization_loss()
        else:
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        if batch_idx % args.log_interval == 0:
            wandb.log({"train/train_loss": loss.data.item(), "train/train_accuracy": 100.0 * correct / ((batch_idx + 1) * len(data))}) if args.wandb_name else None
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
        # Empty gpu cache
        if use_cuda:
            torch.cuda.empty_cache()
    accuracy = 100.0 * correct / len(train_loader.dataset)
    total_loss /= len(train_loader)
    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )
    if args.model_name == "double":
        weight_layer_weights = model.weight_layer.weight
        print(weight_layer_weights)
        wandb.log({"train/weight_visual_emb": weight_layer_weights[0, 0].item(), "train/weight_text_emb": weight_layer_weights[0, 1].item()}) if args.wandb_name else None
    wandb.log({"train/epoch_train_loss": total_loss, "train/epoch_train_accuracy": accuracy}) if args.wandb_name else None



def validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
) -> float:
    """Default Validation Loop.

    Args:
        model (nn.Module): Model to train
        val_loader (torch.utils.data.DataLoader): Validation data loader
        use_cuda (bool): Whether to use cuda or not

    Returns:
        float: Validation loss
    """
    model.eval()
    validation_loss = 0
    correct = 0
    all_preds, all_targets = [], []
    for data, textual, target in val_loader:
        if use_cuda:
            data, textual, target = data.cuda(), textual.cuda(), target.cuda()
        output = model(data, textual)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        criterion  = torch.nn.CrossEntropyLoss(reduction="mean")
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    validation_loss /= len(val_loader)
    accuracy = 100.0 * correct / len(val_loader.dataset)
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            accuracy,
        )
    )
    wandb.log({"validation/val_loss": validation_loss, "validation/val_accuracy": accuracy}) if args.wandb_name else None
    return validation_loss


def main():
    """Default Main Function."""
    # options
    args = opts()

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    model = ModelFactory(args.model_name, 500).get_model()

    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    # Data loading code
    from datasets import load_dataset

    ds_train = load_dataset("cuevascarlos/ImageNet-Sketch-Embed", split="train")
    ds_val = load_dataset("cuevascarlos/ImageNet-Sketch-Embed", split="validation")
    
    #Map labels to values taking into account the tuples of class2category.csv
    class2category = pd.read_csv('class2category.csv', header=0, names=['class', 'category'])
    label_map = dict(zip(class2category['class'], class2category['category']))
    ds_train_f = ds_train.map(map_labels, fn_kwargs={"label_map": label_map})
    ds_val_f = ds_val.map(map_labels, fn_kwargs={"label_map": label_map})

    train_loader = torch.utils.data.DataLoader(
        ds_train_f,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        ds_val_f,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Loop over the epochs
    best_val_loss = 1e8
    for epoch in range(1, args.epochs + 1):
        # training loop
        train(model, optimizer, train_loader, use_cuda, epoch, args)
        # validation loop
        val_loss = validation(model, val_loader, use_cuda)
        if val_loss < best_val_loss:
            # save the best model for validation
            best_val_loss = val_loss
            best_model_file = args.experiment + "/model_best.pth"
            torch.save(model.state_dict(), best_model_file)
        # also save the model every epoch
        model_file = args.experiment + "/model_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), model_file)
        print(
            "Saved model to "
            + model_file
            + f". You can run `python evaluate_dataset.py --model_name {args.model_name} --model "
            + best_model_file
            + "` to generate the Kaggle formatted csv file\n"
        )


if __name__ == "__main__":
    args = opts()
    if args.wandb_name:
        wandb.init(
        # set the wandb project where this run will be logged
        project="A3-RecVis",
        # set the name of the run
        name=args.wandb_name,
        )
    else:
        print("Not saving the training process in wandb. To do that, use the --wandb_name argument.")

    main()

    if args.wandb_name:
        wandb.finish()
