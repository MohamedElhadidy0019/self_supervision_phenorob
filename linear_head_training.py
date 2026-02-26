
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader 
from torchvision import datasets
import torch.nn as nn
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from PIL import Image
import PIL
# import cv2
from sklearn.model_selection import train_test_split
import random as python_random
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import glob
import re
from get_metadata import get_metadata
import torchvision
gpus = "7"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
seed = 78
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

#---------------------------------Using the pre-built function "get_metadata"-------------------------------
import pandas as pd
root_path = "/data/home/mfarag/farag_data/cp_journal/technical/datasets/Sugar Beet/DND-SB/images"

print("Analysis: collecting metadata")
random = True                               # True: random split, False: temporal split
split_random = ['./train.txt', './test.txt'] # ./train.txt | ./test.txt
split_time = [1]                            # a sub set of [1,2,3,4,5,6,7,8,9,10], indicating different dates

for split_type in split_random:
    if split_type == './train.txt':                            
        _, train_img_names, train_img_labels = get_metadata(root_path, split_type if random else '', split_time if not random else [])
        print('\n\n\n')
    elif split_type == './test.txt':
        _, test_img_names, test_img_labels = get_metadata(root_path, split_type if random else '', split_time if not random else [])

#Saving the data into excel sheets to make it easier to load them later
# pd.DataFrame({'Filename': test_img_names, 'Label': test_img_labels}).to_csv('C:\\Users\\midok\\OneDrive\\Desktop\\CP_journal\\technical\\code\\Sugar Beet\\DND-SB-main\\dataset\\extra information\\test_data.csv', index=False)
#---------------------------------pre-processing data and labels, and builidng the data-loader--------------


# Encoding label 
from sklearn.preprocessing import LabelEncoder
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights, resnet18, vit_b_16
import matplotlib.pyplot as plt
import os
from torchvision.transforms import ToPILImage, ToTensor
from torchvision import transforms

# print(train_img_labels[0])

label_encoder = LabelEncoder()

train_img_enc_labels = label_encoder.fit_transform(train_img_labels)

# print(label_encoder.classes_)

# print(train_img_enc_labels[1])
# print(type(train_img_enc_labels))
test_img_enc_labels = label_encoder.transform(test_img_labels)


train_img_names_array = np.array(train_img_names)
test_img_names_array = np.array(test_img_names)




from torch.utils.data import random_split
from sklearn.model_selection import train_test_split

torch.manual_seed(seed)
np.random.seed(seed)
# You can now use train_dataloader for training and val_dataloader for validation



# print(test_img_enc_labels[1])

class sb_dataset(Dataset):
    def __init__(self, main_dir, img_paths, labels, transform=None):
        """
        Args:
            img_paths (list): List of paths to the images.
            labels (list): List of labels corresponding to each image.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.main_dir  = main_dir
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.main_dir, self.img_paths[idx].strip())  # Use os.path.join for better path handling and strip() to remove newline characters
        image = Image.open(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label

# # Example usage
# if __name__ == "__main__":
#     # Assuming train_img_names and train_img_labels are your lists of image paths and labels
#     transform = ToTensor()  # Example transform; you can customize this as needed
#     dataset = CustomImageDataset(train_img_names, train_img_labels, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

#     # Example: Iterate through the DataLoader
#     for images, labels in dataloader:
#         # Here you can use images and labels
#         pass

# main_dir = "C:\\Users\\midok\\OneDrive\\Desktop\\CP_journal\\technical\\code\\Sugar Beet\\DND-SB-main\\dataset\\pre_processed\\"
# img_paths = train_img_names
# labels = train_img_enc_labels


# def preprocess_and_save_images(image_paths, labels, main_dir, save_dir, transform):
#     """
#     Apply transformations to the images and save them to disk.

#     Args:
#         image_paths (list): List of image file paths.
#         labels (list): List of labels corresponding to each image.
#         main_dir (str): Directory containing the original images.
#         save_dir (str): Directory where transformed images will be saved.
#         transform (callable): Transformation to apply to each image.
#     """
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     for img_path, label in zip(image_paths, labels):
#         full_path = os.path.join(main_dir, img_path.strip())
#         image = Image.open(full_path)
#         if transform:
#             image = transform(image)
#             print(image)
#         # Assuming img_path is a filename, adjust if it includes subdirectories
#         filename = os.path.basename(img_path).strip()
#         # print(filename[:-4])
#         transform_to_pil = ToPILImage()  # Again, use strip() to ensure no newline character
#         image_pil = transform_to_pil(image)
        
#         save_path = os.path.join(save_dir, filename)
#         print(save_path)
#         image_pil.save(save_path)

main_dir = "/data/home/mfarag/farag_data/cp_journal/technical/code/Sugar Beet/DND-SB-main/dataset/pre_processed_new"

# save_dir= "C:\\Users\\midok\\OneDrive\\Desktop\\CP_journal\\technical\\code\\Sugar Beet\\DND-SB-main\\dataset\\pre_processed_new\\"
img_paths = train_img_names
labels = train_img_enc_labels

# data_transform = transforms.Compose([transforms.Resize(256), 
#                                 transforms.CenterCrop(224),
#                                 transforms.ToTensor()])

# preprocess_and_save_images(img_paths, labels, main_dir, save_dir, data_transform)

img_paths_train, img_paths_valcal, labels_train, labels_valcal = train_test_split(
    img_paths, labels, test_size=0.25, random_state=seed, shuffle=True, stratify=train_img_enc_labels)

train_transform = transforms.Compose([
    # transforms.RandomVerticalFlip(), 
                                transforms.RandomHorizontalFlip(),
                                # transforms.RandomRotation(45),
                                # transforms.RandomCrop 
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225]) ])


IMAGENET_NORMALIZE = {"mean":[0.485,0.456,0.406], "std":[0.229,0.224,0.225]}
input_size = 256  # <- set this to what kNN used

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=IMAGENET_NORMALIZE["mean"],
                                     std=IMAGENET_NORMALIZE["std"]),
])


train_dataset = sb_dataset(main_dir, img_paths_train, labels_train, transform=train_transform)

# # Assuming train_dataset is your original training dataset
# total_samples = len(train_dataset)
# val_cal_size = 1000  # Size of the validation dataset
# train_size = total_samples - val_cal_size  # Size of the new training dataset

# # Splitting the dataset
# train_dataset, val_cal_dataset = random_split(train_dataset, [train_size, val_cal_size], generator=torch.Generator().manual_seed(seed))




train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, 
                              generator=torch.Generator().manual_seed(seed), 
                              drop_last=True,num_workers=3, pin_memory=True, persistent_workers=True, prefetch_factor=4)

valcal_transform =  transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225]) ])
valcal_dataset = sb_dataset(main_dir, img_paths_valcal, labels_valcal, transform=valcal_transform)

valcal_dataloader = DataLoader(valcal_dataset, batch_size=256, shuffle=False, 
                              generator=torch.Generator().manual_seed(seed), drop_last=True, 
                              num_workers=3, pin_memory=True, persistent_workers=True, prefetch_factor=4)

# train_label_list = []
# for _, label in train_dataloader:
#     train_label_list.append(label)

# train_label_tensor = torch.cat(train_label_list)    

# plt.hist(train_label_tensor)

# # Assuming train_label_tensor contains the labels
# label_counts = torch.bincount(train_label_tensor)
# total_labels = train_label_tensor.size(0)
# label_percentages = (label_counts / total_labels) * 100

# # Printing the percentage of each label
# for i, percentage in enumerate(label_percentages):
#     print(f"Label {i}: {percentage:.2f}%")

total_samples = len(valcal_dataset)
cal_size = 500  # Size of the validation dataset
val_size = total_samples - cal_size

val_dataset, cal_dataset = random_split(valcal_dataset, [val_size, cal_size], generator=torch.Generator().manual_seed(seed))

val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                              generator=torch.Generator().manual_seed(seed), drop_last=True)

cal_dataloader = DataLoader(cal_dataset, batch_size=32, shuffle=False, 
                              generator=torch.Generator().manual_seed(seed), drop_last=True)
print(len(train_dataloader))
print(len(valcal_dataloader))

print(len(val_dataloader))
print(len(cal_dataloader))


# train_label_list = []
# for _, label in val_dataloader:
#     train_label_list.append(label)

# train_label_tensor = torch.cat(train_label_list)   

# label_counts = torch.bincount(train_label_tensor)
# total_labels = train_label_tensor.size(0)
# label_percentages = (label_counts / total_labels) * 100
# for i, percentage in enumerate(label_percentages):
#     print(f"Label {i}: {percentage:.2f}%")
# plt.hist(train_label_tensor)

# train_label_list = []
# for _, label in cal_dataloader:
#     train_label_list.append(label)

# train_label_tensor = torch.cat(train_label_list)    

# label_counts = torch.bincount(train_label_tensor)
# total_labels = train_label_tensor.size(0)
# label_percentages = (label_counts / total_labels) * 100
# for i, percentage in enumerate(label_percentages):
#     print(f"Label {i}: {percentage:.2f}%")

# plt.hist(train_label_tensor)

# print(transforms)

# Function to show images
# def show_images(images, labels, n_images=4):
#     fig, axes = plt.subplots(1, n_images, figsize=(n_images * 3, 3))
#     for i in range(n_images):
#         ax = axes[i]
#         img = images[i].permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
#         ax.imshow(img)
#         ax.set_title(f'Label: {labels[i]}')
#         ax.axis('off')
#     plt.show()

# # Get a batch of images from the train_dataloader
# for images, labels in train_dataloader:
#     show_images(images, labels, n_images=4)
#     break  # Only show the first batch


# def preprocess_and_save_images(image_paths, labels, main_dir, save_dir, transform):
#     """
#     Apply transformations to the images and save them to disk.

#     Args:
#         image_paths (list): List of image file paths.
#         labels (list): List of labels corresponding to each image.
#         main_dir (str): Directory containing the original images.
#         save_dir (str): Directory where transformed images will be saved.
#         transform (callable): Transformation to apply to each image.
#     """
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     for img_path, label in zip(image_paths, labels):
#         full_path = os.path.join(main_dir, img_path.strip())
#         image = Image.open(full_path)
#         if transform:
#             image = transform(image)
#             print(image)
#         # Assuming img_path is a filename, adjust if it includes subdirectories
#         filename = os.path.basename(img_path).strip()
#         # print(filename[:-4])
#         transform_to_pil = ToPILImage()  # Again, use strip() to ensure no newline character
#         image_pil = transform_to_pil(image)
        
#         save_path = os.path.join(save_dir, filename)
#         print(save_path)
#         image_pil.save(save_path)
        

# # Example usage
# transform = ToTensor()  # Replace with your actual transform
# preprocess_and_save_images(train_img_names, train_img_labels, main_dir, "path/to/save_dir", transform)





# main_dir = "C:\\Users\\midok\\OneDrive\\Desktop\\CP_journal\\technical\\code\\Sugar Beet\\DND-SB-main\\dataset\\pre_processed\\"
main_dir = "/data/home/mfarag/farag_data/cp_journal/technical/code/Sugar Beet/DND-SB-main/dataset/pre_processed_new"
img_paths = test_img_names
labels = test_img_enc_labels
# save_dir= "C:\\Users\\midok\\OneDrive\\Desktop\\CP_journal\\technical\\code\\Sugar Beet\\DND-SB-main\\dataset\\pre_processed_new\\"
# # pre_trained_weights = ResNet18_Weights.IMAGENET1K_V1
# # transform = pre_trained_weights.transforms()

# preprocess_and_save_images(img_paths, labels, main_dir, save_dir, data_transform)


# img_path = os.path.join(main_dir, img_paths[0].strip())
# print(img_paths[0])
# # Load the image
# image = Image.open(img_path)

# # Plot the original image
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title('Original Image')

# # Apply the transformation
# transformed_image = transforms(image)

# # Convert the transformed image back to PIL Image to display it
# transform_to_pil = ToPILImage()
# image_pil = transform_to_pil(transformed_image)

# # Plot the transformed image
# plt.subplot(1, 2, 2)
# plt.imshow(image_pil)
# plt.title('Transformed Image')
# plt.show()



# img_paths = test_img_names_array
# labels = test_img_enc_labels


import torchvision
# test_transform =  transforms.Compose([
#                                 # transforms.ColorJitter(brightness=.5, hue=.3),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize([0.485, 0.456, 0.406], 
#                                                      [0.229, 0.224, 0.225]) ])

IMAGENET_NORMALIZE = {"mean":[0.485,0.456,0.406], "std":[0.229,0.224,0.225]}
input_size = 256  # <- set this to what kNN used

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=IMAGENET_NORMALIZE["mean"],
                                     std=IMAGENET_NORMALIZE["std"]),
])
test_dataset = sb_dataset(main_dir, img_paths, labels, transform=test_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                              generator=torch.Generator().manual_seed(seed), drop_last=True)

print(len(test_dataloader))

# train_label_list = []
# for _, label in test_dataloader:
#     train_label_list.append(label)

# train_label_tensor = torch.cat(train_label_list)    

# label_counts = torch.bincount(train_label_tensor)
# total_labels = train_label_tensor.size(0)
# label_percentages = (label_counts / total_labels) * 100
# for i, percentage in enumerate(label_percentages):
#     print(f"Label {i}: {percentage:.2f}%")


# plt.hist(train_label_tensor)

# Dictionary to store one image path for each class
class_to_image = {}

# Iterate over the train_dataloader to find one image for each class
for images, labels in train_dataloader:
    for image, label in zip(images, labels):
        label = label.item()
        if label not in class_to_image:
            # Convert the image tensor to a numpy array and transpose it to (H, W, C) format
            inv_normalize = transforms.Normalize(
            mean=[-m / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
            std=[1 / s for s in [0.229, 0.224, 0.225]])
                                                

            # Reverse the normalization process
            

            # Ensure the image data is in the valid range [0, 1] after reversing normalization
            image_data = image
            image_inv = inv_normalize(image_data)

            # Ensure the tensor is in the correct format for visualization
            image_inv = image_inv.permute(1, 2, 0)  # Reorder dimensions to [H, W, C]
            image_inv = torch.clamp(image_inv, 0, 1)
            # image_np = image.permute(1, 2, 0).numpy()
            class_to_image[label] = image_inv
            if len(class_to_image) == len(set(train_img_enc_labels)):
                break
    if len(class_to_image) == len(set(train_img_enc_labels)):
        break

# Display the images for each class
for class_label, image_data in class_to_image.items():
    class_name = label_encoder.classes_[class_label]
    plt.imshow(image_data)
    # plt.title(f"Class: {class_label} - {class_name}")
    plt.axis('off')
    plt.show()
    
    print(f"Class: {class_label} - {class_name}")
# break

# -------------------------------Building ResNet Ensemble---------------------------------    
torch.manual_seed(seed)
np.random.seed(seed)


ckpt = torch.load("/data/home/mfarag/farag_data/cp_journal/technical/code/Sugar Beet/DND-SB-main/dataset/benchmark_logs/imagenet/version_16/SwaV/checkpoints/epoch=1199-step=16800.ckpt", map_location="cpu", weights_only=False)
ckpt['state_dict'].keys()

import torchvision
# resnet = torchvision.models.resnet18()
# feature_dim = list(resnet.children())[-1].in_features
# backbone = nn.Sequential(*list(resnet.children())[:-1])

import timm
# vit = timm.create_model('vit_small_patch8_224.dino',num_classes=0, pretrained=False, dynamic_img_size=True, dynamic_img_pad=True)
# vit.dynamic_img_size = True
# feature_dim = vit.embed_dim
# backbone=vit


convx = torchvision.models.convnext_tiny()
feature_dim = list(convx.children())[-1][2].in_features
backbone = nn.Sequential(*list(convx.children())[:-1])

# Filter out only backbone weights from ckpt['state_dict']
backbone_state_dict = {k.replace('backbone.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('backbone.')}

backbone.load_state_dict(backbone_state_dict)
state = ckpt.get("state_dict", ckpt)

# strip common prefixes
for pref in ("module.", "model."): 
    state = { (k[len(pref):] if k.startswith(pref) else k): v for k,v in state.items() }

# take only backbone.* and drop the prefix
bb_state = { k[len("backbone."):]: v for k,v in state.items() if k.startswith("backbone.") }

missing, unexpected = backbone.load_state_dict(bb_state, strict=False)
print("Missing:", missing, "Unexpected:", unexpected)

# sanity: weights shouldn't be near-zero like initialization
print("Backbone weight mean abs:",
      sum(p.abs().mean() for p in backbone.parameters()).item())
# ---------------------------------Linear head training---------------------------------
import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ==== Config ====
# data_dir    = "/path/to/dataset"   # expects data_dir/train and data_dir/val
batch_size  = 256
epochs      = 300
num_workers = 0
device      = "cuda" if torch.cuda.is_available() else "cpu"
seed        = 78

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ==== Data ====
# For ResNet-sized inputs + ImageNet normalization
tfm_train = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
tfm_val = tfm_train

# train_set = datasets.ImageFolder(os.path.join(data_dir, "train"), tfm_train)
# val_set   = datasets.ImageFolder(os.path.join(data_dir, "val"),   tfm_val)
num_classes = 7

train_loader = train_dataloader
val_loader   = valcal_dataloader

# ==== Backbone (frozen) + Linear Head ====
# resnet = models.resnet18(weights=None)  # or models.ResNet18_Weights.IMAGENET1K_V1
# feature_dim = resnet.fc.in_features     # 512 for resnet18
# backbone = nn.Sequential(*list(resnet.children())[:-1])  # up to avgpool -> (N,C,1,1)


# resnet = torchvision.models.resnet50(weights="IMAGENET1K_V1")
# feature_dim = list(resnet.children())[-1].in_features
# backbone = nn.Sequential(*list(resnet.children())[:-1])
# print(feature_dim
      
#       )
# Filter out only backbone weights from ckpt['state_dict']
# backbone_state_dict = {k.replace('backbone.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('backbone.')}

# backbone.load_state_dict(backbone_state_dict)


# missing, unexpected = backbone.load_state_dict(bb_state, strict=False)
# print("Missing:", missing, "Unexpected:", unexpected)

for p in backbone.parameters():
    p.requires_grad = False
backbone.eval()

head = nn.Linear(feature_dim, num_classes)
bn = nn.BatchNorm2d(num_features=feature_dim)
model = nn.Sequential(
    backbone,
    bn,
    nn.Flatten(start_dim=1),
    head,
).to(device)

# Train only the linear head
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(head.parameters(), lr=0.001)

optimizer = optim.SGD([
    # {'params': backbone.parameters(), 'lr': 1e-3},  # frozen backbone
    {'params': model[1].parameters(), 'lr': 1e-3},   # bn params
    {'params': head.parameters(), 'lr': 1e-3},       # head slightly higher LR
])
# scheduler = optim.CosineAnnealingLR(optimizer, T_max=epochs)

# ==== Train / Eval helpers ====
def train_epoch(loader):
    model[0].eval()    # frozen backbone
    model[1].train()   # train Batch Normalization
    model[-1].train()  # train linear head
    running_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.no_grad():
            feats = model[0](x)
            # for vits make the shape B, C, 1, 1
            # to be acceted by bn layer
            # feats = feats.unsqueeze(-1).unsqueeze(-1)
            feats_bn = model[1](feats)
            feats_final = torch.flatten(feats_bn, 1)
            

        optimizer.zero_grad(set_to_none=True)
        logits = model[-1](feats_final)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(loader):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        feats = model[0](x)
        # for vits make the shape B, C, 1, 1
        # to be acceted by bn layer
        # feats = feats.unsqueeze(-1).unsqueeze(-1)
        feats_bn = model[1](feats)
        feats_final = torch.flatten(feats_bn, 1)
        logits = model[-1](feats_final)
        loss = criterion(logits, y)
        running_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

# ==== Loop ====
best_acc = 0.0
best_loss = 1e7

os.makedirs("linear_checkpoints", exist_ok=True)

for epoch in range(1, epochs + 1):
    t0 = time.time()
    train_loss, train_acc = train_epoch(train_loader)
    val_loss, val_acc     = evaluate(val_loader)
    # scheduler.step()

    if val_acc > best_acc:
        best_acc = val_acc
        # torch.save({"head": head.state_dict(), 'bn': bn.state_dict(),
        #             "classes": 7}, "/data/home/mfarag/farag_data/cp_journal/technical/code/Sugar Beet/DND-SB-main/dataset/linear_checkpoints/sl_lh_bn_best_acc.pt")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({"head": head.state_dict(), 'bn': bn.state_dict(),
                    "classes": 7}, "/data/home/mfarag/farag_data/cp_journal/technical/code/Sugar Beet/DND-SB-main/dataset/linear_checkpoints/version_16/SwaV/lh_bn_best_val_sgd.pt")
        # torch.save({"head": head.state_dict(),
        #             "classes": 7}, "/data/home/mfarag/farag_data/cp_journal/technical/code/Sugar Beet/DND-SB-main/dataset/linear_checkpoints/version_15/sl_lh_bn_best_val.pt")
        
    print(f"Epoch {epoch:03d}/{epochs} "
          f"| train loss {train_loss:.4f} acc {train_acc*100:.2f}% " 
          f"| val loss {val_loss:.4f} acc {val_acc*100:.2f}% "
          f"| time {(time.time()-t0):.1f}s")

print(f"Best val acc: {best_acc*100:.2f}%")
