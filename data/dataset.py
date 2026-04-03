from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from utils.keypoints import extract_hand_keypoints_np, get_or_compute_2d_kp, get_or_compute_3d_kp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from config_ import n_feat, n_classes, n_T, batch_size
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#Dataset
data_path = "./Bharatanatyam_hasta_mudra_dataset"

# Original label mapping
label_mapping = [name for name in os.listdir(data_path) 
         if os.path.isdir(os.path.join(data_path, name))][:28]

print("Folders found:", label_mapping)
# Sort alphabetically (as ImageFolder does)
sorted_label_mapping = sorted(label_mapping)
label_to_index = {label: idx for idx, label in enumerate(sorted_label_mapping)}
print("Alphabetically Sorted Label Mapping:", label_to_index)
print("lable:",label_to_index)
img_size = 128
# Updated image transformations (without grayscale)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # Resize to 128x128
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize([0.5], [0.5]),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels to [-1, 1]
])

# Step 2: One-hot encode function according to the label mapping
#def one_hot_encode(labels, num_classes):
    # Convert labels to one-hot encoding based on number of classes
#    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

class AsamyuktaHandDataset(Dataset):
    def __init__(self, root_dir, transform, allowed_folders, label_to_index, kp_cache_dir, kp3d_cache_dir):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.allowed_folders = allowed_folders
        self.label_to_index = label_to_index
        self.kp_cache_dir = kp_cache_dir
        self.kp3d_cache_dir = kp3d_cache_dir

        self.filter_dataset()

    def filter_dataset(self):
        filtered_classes = [cls for cls in self.dataset.classes if cls in self.allowed_folders]
        class_to_idx = {cls: idx for idx, cls in enumerate(filtered_classes)}

        filtered_samples = [
            (path, class_to_idx[self.dataset.classes[label]])
            for path, label in self.dataset.samples
            if self.dataset.classes[label] in self.allowed_folders
        ]

        self.dataset.classes = filtered_classes
        self.dataset.samples = filtered_samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        path, _ = self.dataset.samples[idx]

        kp_2d = get_or_compute_2d_kp(path, self.kp_cache_dir)
        kp_3d = get_or_compute_3d_kp(path, self.kp3d_cache_dir)

        return img, label, torch.tensor(kp_2d), torch.tensor(kp_3d)
        

dataset = AsamyuktaHandDataset(root_dir=data_path,transform=transform,allowed_folders=label_mapping,label_to_index=label_to_index,kp_cache_dir="./hasta_kp_cahe",kp3d_cache_dir="./hasta_3d_kp_cahe")  


# Step 1: Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

# Step 2: Create DataLoaders for training and validation sets

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# Function to convert labels in a batch to one-hot encoding

