import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from transformers import AutoModel, AutoProcessor, AutoModelForVision2Seq, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import os
import gc
import pandas as pd
import argparse

'''
parser = argparse.ArgumentParser(description="RecVis A3 auxiliar dataset script")
parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size",
    )
args = parser.parse_args()
batch_size = args.batch_size
'''
batch_size = 16

class ImageTextDataset(Dataset):
    def __init__(self, save_path="./dataset.parquet"):
        """
        Initialize the dataset.
        """
        self.data = []
        self.save_path = save_path
        
        if os.path.exists(self.save_path):
            df_loaded = pd.read_parquet(self.save_path)
            self.data = df_loaded.to_dict(orient="records")
            
        print(f"Loaded dataset with {len(self.data)} samples from {self.save_path}")

        
    def add_data(self, label, visual_embedding, caption, textual_embedding, image_path):
        """
        Add a new batch of data to the dataset.
        """
        
        for lab, visual_embed, capt, textual_embed, im_path in zip(label, visual_embedding, caption, textual_embedding, image_path):
            self.data.append({
                'label': lab,
                'visual_embedding': visual_embed,
                'caption': capt,
                'textual_embedding': textual_embed,
                'image_path': im_path
            })
        self._save_dataset()

    def _save_dataset(self, name=None):
        if name is not None:
            self.save_path = f"./{name}.parquet"
        
        df = pd.DataFrame(self.data)
        
        df.to_parquet(f"{self.save_path}", compression="snappy")
        print(f"Dataset saved in {self.save_path}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        label = item['label']
        visual_embedding = item['visual_embedding']
        caption = item['caption']
        textual_embedding = item['textual_embedding']
        image_path = item['image_path']
        
        return {
            'label': label,
            'visual_embedding': visual_embedding,
            'caption': caption,
            'textual_embedding': textual_embedding,
            'image_path': image_path
        }
        


def extract_visual_embedding_batch(images, model, processor, device):
    inputs = processor(images=images, return_tensors="pt").to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1) # Pooling global

def generate_caption_batch(images, model, processor, device):
    inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)
    print(f"Inputs shape: {inputs['pixel_values'].shape}")
    
    generated_ids = model.generate(**inputs)
    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return captions

def extract_text_embeddings_batch(captions, model, tokenizer, device):
    inputs = tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Pooling global


# Set device to cuda if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model DINOv2
dino_processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

# Model BLIP-2
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None 
)

blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip_model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip2-flan-t5-xl",  quantization_config=bnb_config, torch_dtype=torch.float16, device_map="auto")

# Model BERT
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)

directory = './sketch_recvis2024/sketch_recvis2024/'

# Check if the datasets are fully created in the directory

split_list = []
if not os.path.exists("./train_images.parquet"):
    split_list.append("train_images")
else:
    print("Train dataset already created")
if not os.path.exists("./val_images.parquet"):
    split_list.append("val_images")
else:
    print("Validation dataset already created")
if not os.path.exists("./test_images.parquet"):
    split_list.append("test_images")
else:
    print("Test dataset already created")


for split in split_list:
    dataset = ImageTextDataset()

    # Check the images that have already been processed
    image_paths_in_dataset = [item['image_path'] for item in dataset]

    # Calculate total iterations for the single progress bar
    if split == "train_images" or split == "val_images":
        total_iterations = sum(len(os.listdir(os.path.join(directory+split, label))) for label in os.listdir(directory+split))-len(image_paths_in_dataset)
    else:
        total_iterations = len(os.listdir(directory+split+"/mistery_category/"))-len(image_paths_in_dataset)
    
    with tqdm(total=total_iterations, desc=f"Processing {split}") as pbar:
        for label in sorted(os.listdir(directory+split)):
            images_per_batch = {'image': [],'label': [], 'visual_embedding': [], 'caption': [], 'textual_embedding': [], 'path': []}
            
            for image in sorted(os.listdir(os.path.join(directory+split, label))):
                image_path = os.path.join(directory+split, label, image)

                # Skip if the image has already been processed
                if image_path in image_paths_in_dataset:
                    continue
                
                images_per_batch['path'].append(image_path)
                images_per_batch['label'].append(label)
                
                # Load image
                img = Image.open(image_path).resize((224, 224)).convert("RGB")
                images_per_batch['image'].append(img)

                # Process in batch
                if len(images_per_batch['image']) == batch_size:
                    # Get visual embeddings in batch
                    visual_embeddings = extract_visual_embedding_batch(images_per_batch['image'], dino_model, dino_processor, device)
                    images_per_batch['visual_embedding'].extend(visual_embeddings.cpu().detach().numpy().tolist())

                    # Generate captions in batch
                    captions = generate_caption_batch(images_per_batch['image'], blip_model, blip_processor, device)
                    images_per_batch['caption'].extend(captions)

                    # Get textual embeddings in batch
                    textual_embeddings = extract_text_embeddings_batch(captions, bert_model, bert_tokenizer, device)
                    images_per_batch['textual_embedding'].extend(textual_embeddings.cpu().detach().numpy().tolist())
                    
                    pbar.update(batch_size)
                    dataset.add_data(images_per_batch['label'], images_per_batch['visual_embedding'], images_per_batch['caption'], images_per_batch['textual_embedding'], images_per_batch['path'])
                    
                    #Empty the batch dict
                    images_per_batch = {'image': [],'label': [], 'visual_embedding': [], 'caption': [], 'textual_embedding': [], 'path': []}

            # Process the remaining images
            if images_per_batch['image']:
                visual_embeddings = extract_visual_embedding_batch(images_per_batch['image'], dino_model, dino_processor, device)
                images_per_batch['visual_embedding'].extend(visual_embeddings.cpu().detach().numpy().tolist())
                captions = generate_caption_batch(images_per_batch['image'], blip_model, blip_processor, device)
                images_per_batch['caption'].extend(captions)
                textual_embeddings = extract_text_embeddings_batch(captions, bert_model, bert_tokenizer, device)
                images_per_batch['textual_embedding'].extend(textual_embeddings.cpu().detach().numpy().tolist())                

                pbar.update(len(images_per_batch['image']))

                dataset.add_data(images_per_batch['label'], images_per_batch['visual_embedding'], images_per_batch['caption'], images_per_batch['textual_embedding'], images_per_batch['path'])
                images_per_batch = {'image': [],'label': [], 'visual_embedding': [], 'caption': [], 'textual_embedding': [], 'path': []}

            # Clean ram
            gc.collect()
    
    # Save the complete dataset
    dataset._save_dataset(name=split)
    # Remove the temporary dataset
    del dataset

    # Remove the dataset.parquet file
    os.remove(f"./dataset.parquet")


# Load the parquet files and create a dataset ompatible with HuggingFace
train_dataset = pd.read_parquet("./train_images.parquet").to_dict(orient="list")
val_dataset = pd.read_parquet("./val_images.parquet").to_dict(orient="list")
test_dataset = pd.read_parquet("./test_images.parquet").to_dict(orient="list")

from datasets import Dataset, DatasetDict

train_dataset = Dataset.from_dict(train_dataset)
val_dataset = Dataset.from_dict(val_dataset)
test_dataset = Dataset.from_dict(test_dataset)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

dataset.save_to_disk("./embeddings_dataset")





