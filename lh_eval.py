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
gpus = "5"
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
                                # transforms.RandomHorizontalFlip(),
                                # transforms.RandomRotation(45),
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
                              generator=torch.Generator().manual_seed(seed), drop_last=True, 
                              num_workers=3, pin_memory=True, persistent_workers=True, prefetch_factor=4)

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
    # torchvision.transforms.ColorJitter(brightness=.5, hue=.3),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=IMAGENET_NORMALIZE["mean"],
                                     std=IMAGENET_NORMALIZE["std"]),
])

# test_transforms = torchvision.transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: torch.ones_like(x))  # creates black image
# ])
test_dataset = sb_dataset(main_dir, img_paths, labels, transform=test_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                              generator=torch.Generator().manual_seed(seed), drop_last=True, 
                              num_workers=3, pin_memory=True, persistent_workers=True, prefetch_factor=4)

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

# Import metrics
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, cohen_kappa_score
from torch_uncertainty.metrics.classification import CalibrationError, AdaptiveCalibrationError, AURC
import timm
# Configuration
base_dir = "/data/home/mfarag/farag_data/cp_journal/technical/code/Sugar Beet/DND-SB-main/dataset/benchmark_logs/imagenet/"
lh_base_dir = "/data/home/mfarag/farag_data/cp_journal/technical/code/Sugar Beet/DND-SB-main/dataset/linear_checkpoints/"
version = 'version_16/'
checkpoint_dir = "checkpoints/epoch=1199-step=16800.ckpt"
lh_checkpoint_dir = "lh_bn_best_val_sgd.pt"

# List of approaches to evaluate
# approaches = ['SimSiam', 'SwaV', 'Moco', 'DINO', 'SimCLR', 'NNCLR', 'BYOL', 'BarlowTwins']

approaches = ['SwaV',  'DINO', 'SimCLR']


num_classes = 7
device = "cuda" if torch.cuda.is_available() else "cpu"
num_bins = 15

# Create output directory
output_dir = "convx_T_bn_sgd"
os.makedirs(output_dir, exist_ok=True)

# Helper function to load backbone
def load_backbone(approach):
    """Load backbone for a given SSL approach."""
    ckpt_path = base_dir + version + approach + '/' + checkpoint_dir
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # resnet = torchvision.models.resnet50()
    # feature_dim = list(resnet.children())[-1].in_features
    # backbone = nn.Sequential(*list(resnet.children())[:-1])

    
    # vit = timm.create_model('vit_small_patch8_224.dino',num_classes=0, pretrained=False, dynamic_img_size=True, dynamic_img_pad=True)
    # vit.dynamic_img_size = True
    # feature_dim = vit.embed_dim
    # backbone=vit

    convx = torchvision.models.convnext_tiny()
    feature_dim = list(convx.children())[-1][2].in_features
    backbone = nn.Sequential(*list(convx.children())[:-1])
    
    state = ckpt.get("state_dict", ckpt)
    # strip common prefixes
    for pref in ("module.", "model."): 
        state = { (k[len(pref):] if k.startswith(pref) else k): v for k,v in state.items() }
    
    # take only backbone.* and drop the prefix
    bb_state = { k[len("backbone."):]: v for k,v in state.items() if k.startswith("backbone.") }
    
    missing, unexpected = backbone.load_state_dict(bb_state, strict=False)
    print(f"{approach} - Missing: {missing}, Unexpected: {unexpected}")
    
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    
    return backbone, feature_dim

# Helper function to load model with head
def load_model_with_head(backbone, feature_dim, approach):
    """Load model with backbone and linear head."""
    head = nn.Linear(feature_dim, num_classes)
    bn = nn.BatchNorm2d(num_features=feature_dim)
    if "wo_bn" in lh_checkpoint_dir:
        model = nn.Sequential(
            backbone,
            nn.Flatten(start_dim=1),
            head,
        ).to(device)
    else:
        model = nn.Sequential(
            backbone,
            bn,
            nn.Flatten(start_dim=1),
            head,
        ).to(device)
    
    # Load linear head weights
    lh_path = lh_base_dir + version + approach + '/' + lh_checkpoint_dir
    try:
        ckpt_head = torch.load(lh_path, map_location="cpu")
        head = model[-1]
        if "wo_bn" in ckpt_head:
            head.load_state_dict(ckpt_head["head"], strict=False)
        else:    
            bn = model[1]
            head.load_state_dict(ckpt_head["head"], strict=False)
            bn.load_state_dict(ckpt_head["bn"], strict=False)
        
            
        print(f"✅ {approach} weights loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load weights for {approach}: {e}")
    
    return model

def _bin_stats(y_true_bin, y_prob_bin, bins):
    """Helper: compute counts, mean prob, observed frac for one binning."""
    bin_idx = np.digitize(y_prob_bin, bins[1:-1], right=True)  # 0..n_bins-1
    n_bins = len(bins) - 1
    counts = np.zeros(n_bins, dtype=int)
    mean_pred = np.zeros(n_bins, dtype=float)
    frac_pos = np.zeros(n_bins, dtype=float)
    frac_pos[:] = np.nan
    for b in range(n_bins):
        mask = (bin_idx == b)
        counts[b] = mask.sum()
        if counts[b] > 0:
            mean_pred[b] = y_prob_bin[mask].mean()
            frac_pos[b] = y_true_bin[mask].mean()
        else:
            # fallback for plotting x coordinate
            mean_pred[b] = 0.5 * (bins[b] + bins[b+1])
    return counts, mean_pred, frac_pos

# Modified reliability plots function to save instead of show
def multiclass_reliability_plots_save(y_true, y_prob_matrix, approach, save_dir, *,
                                    n_bins=15, strategy='uniform',
                                    show_counts=True, figsize_per_plot=(4,3), dpi=120):
    """Save reliability diagrams for a multiclass model."""
    y_true = np.asarray(y_true)
    P = np.asarray(y_prob_matrix)
    assert P.ndim == 2
    n_samples, K = P.shape
    assert y_true.shape[0] == n_samples

    # 1) define bins
    if strategy == 'uniform':
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == 'quantile':
        q = np.linspace(0, 1, n_bins + 1)
        bins = np.quantile(P.flatten(), q)
        bins = np.unique(bins)
        if len(bins) - 1 < n_bins:
            n_bins = len(bins) - 1
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    per_class = {}
    total = n_samples

    # 2) Per-class one-vs-rest diagrams & ECEs
    for k in range(K):
        y_bin = (y_true == k).astype(int)
        probs_k = P[:, k]
        counts, mean_pred, frac_pos = _bin_stats(y_bin, probs_k, bins)
        abs_diff = np.abs(mean_pred - frac_pos)
        ece_k = np.nansum((counts / total) * abs_diff)
        per_class[k] = {
            'counts': counts,
            'mean_pred': mean_pred,
            'frac_pos': frac_pos,
            'ece': ece_k
        }

    # 3) Top-label diagram
    pred_labels = np.argmax(P, axis=1)
    pred_conf = P[np.arange(n_samples), pred_labels]
    correct = (pred_labels == y_true).astype(int)
    counts_t, mean_pred_t, frac_pos_t = _bin_stats(correct, pred_conf, bins)
    ece_top = np.nansum((counts_t / total) * np.abs(mean_pred_t - frac_pos_t))

    # 4) Plotting - Per-class
    ncols = int(np.ceil(np.sqrt(K)))
    nrows = int(np.ceil(K / ncols))
    fig1, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * figsize_per_plot[0], nrows * figsize_per_plot[1]),
                              dpi=dpi, squeeze=False)
    for k in range(K):
        r = k // ncols
        c = k % ncols
        ax = axes[r][c]
        stats = per_class[k]
        valid = ~np.isnan(stats['frac_pos'])
        ax.plot([0,1],[0,1],'--', linewidth=0.8)
        ax.plot(stats['mean_pred'][valid], stats['frac_pos'][valid], marker='o', label=f'class {k}')
        ax.vlines(stats['mean_pred'][valid], ymin=stats['mean_pred'][valid], ymax=stats['frac_pos'][valid],
                  linestyle=':', linewidth=0.8)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title(f'class {k} ECE={stats["ece"]:.4f}')
        ax.set_xlabel('Mean predicted prob'); ax.set_ylabel('Fraction positives')
        ax.grid(alpha=0.3)
        if show_counts:
            ax2 = ax.twinx()
            bin_mids = 0.5 * (bins[:-1] + bins[1:])
            ax2.bar(bin_mids, stats['counts'], width=(bins[1:]-bins[:-1])*0.9, alpha=0.15)
            ax2.set_ylabel('count')

    for k in range(K, nrows*ncols):
        r = k // ncols
        c = k % ncols
        axes[r][c].axis('off')

    fig1.suptitle(f'{approach} - Per-class reliability diagrams (one-vs-rest)', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{approach}_per_class_reliability.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

    # Top-label plot
    fig2, ax = plt.subplots(figsize=(6,5), dpi=dpi)
    valid = ~np.isnan(frac_pos_t)
    ax.plot([0,1],[0,1],'--', linewidth=0.8)
    ax.plot(mean_pred_t[valid], frac_pos_t[valid], marker='o')
    ax.vlines(mean_pred_t[valid], ymin=mean_pred_t[valid], ymax=frac_pos_t[valid], linestyle=':', linewidth=0.8)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel('Mean confidence (predicted prob)')
    ax.set_ylabel('Observed accuracy')
    ax.set_title(f'{approach} - Top-label reliability diagram — ECE={ece_top:.4f}')
    if show_counts:
        ax2 = ax.twinx()
        bin_mids = 0.5 * (bins[:-1] + bins[1:])
        ax2.bar(bin_mids, counts_t, width=(bins[1:]-bins[:-1])*0.9, alpha=0.15)
        ax2.set_ylabel('count')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{approach}_top_label_reliability.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

    return {
        'bins': bins,
        'per_class': per_class,
        'top_label': {
            'counts': counts_t,
            'mean_pred': mean_pred_t,
            'frac_pos': frac_pos_t,
            'ece': ece_top
        }
    }
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score

def model_eval(model, data_loader, eval=True):
    """
    This function evaluates a given model on a provided dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        data_loader (torch.utils.data.DataLoader): The DataLoader for the dataset on which the model is to be evaluated.
        eval (bool, optional): A flag indicating whether the model should be put in evaluation mode. Defaults to True.

    Returns:
        list: A list of softmax outputs from the model.
        list: A list of predicted labels from the model.
        list: A list of actual labels from the dataset.
    """
    
    # Initialize lists to store softmax outputs, predicted labels, and actual labels
    logits = []
    predictions = []
    actual_labels = []
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize a counter for the number of correct predictions
    correct_predictions = 0

    # Put the model in evaluation mode if the eval flag is True
    if eval:
        model.eval()

    # Disable gradient computation since we are only evaluating the model, not training it
    with torch.no_grad():
        # Iterate over batches
        for batch, (inputs, labels) in enumerate(data_loader):
            # Move inputs and labels to the GPU
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            logits.append(outputs)

            # For vits
            # feats = model[0](inputs)
            # # for vits make the shape B, C, 1, 1
            # # to be acceted by bn layer
            # feats = feats.unsqueeze(-1).unsqueeze(-1)
            # feats_bn = model[1](feats)
            # feats_final = torch.flatten(feats_bn, 1)
            # outputs = model[-1](feats_final)
            # logits.append(outputs)
            # Store softmax outputs
            

            # Get predicted labels
            _, predicted = torch.max(outputs.data, 1)
            # Update the number of correct predictions
            correct_predictions += (predicted == labels).sum().item()

            # Store predicted and actual labels
            predictions.append(predicted)
            actual_labels.append(labels)

    # Return softmax outputs, predicted labels, and actual labels
    return logits, predictions, actual_labels



def output_postprocessing(list_outputs):
    """
    This function concatenates a list of PyTorch tensors along the first dimension (batch dimension),
    and then converts the resulting tensor to a numpy array.

    Args:
        list_outputs (list): A list of PyTorch tensors to be concatenated and converted.

    Returns:
        numpy.ndarray: The concatenated tensor converted to a numpy array.
    """
    # Concatenate the list of tensors along the first dimension
    output_postprocess = torch.cat([x for x in list_outputs], dim=0)
    # Convert the resulting tensor to a numpy array
    output_postprocess = output_postprocess.cpu().numpy()
    
    return output_postprocess




def evaluate_approach(approach, test_dataloader):
    """Evaluate a single SSL approach and return all metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating {approach}")
    print(f"{'='*60}")
    
    try:
        # Load backbone
        backbone, feature_dim = load_backbone(approach)
        
        # Load model with head
        model = load_model_with_head(backbone, feature_dim, approach)
        
        # Evaluate model
        logits_outputs_test, predictions_test, actual_labels_test = model_eval(
            model=model, data_loader=test_dataloader)
        
        act_test = output_postprocessing(actual_labels_test)
        pred_test = output_postprocessing(predictions_test)
        
        soft_test = torch.cat([x for x in logits_outputs_test], dim=0)
        soft_test = torch.softmax(soft_test, dim=1)
        
        # Compute accuracy
        test_acc = accuracy_score(act_test, pred_test)
        
        # Compute calibration metrics
        soft = soft_test.float().cpu()
        if not isinstance(act_test, torch.Tensor):
            act = torch.from_numpy(act_test.astype(np.int64))
        else:
            act = act_test.long()
        act = act.cpu()
        
        ece = CalibrationError(num_bins=num_bins, norm='l1', task='multiclass', num_classes=num_classes)
        max_ece = CalibrationError(num_bins=num_bins, norm='max', task='multiclass', num_classes=num_classes)
        test_ece = ece(soft, act)
        test_max_ece = max_ece(soft, act)
        
        ace = AdaptiveCalibrationError(num_bins=num_bins, norm='l1', task='multiclass', num_classes=num_classes)
        max_ace = AdaptiveCalibrationError(num_bins=num_bins, norm='max', task='multiclass', num_classes=num_classes)
        test_ace = ace(soft, act)
        test_max_ace = max_ace(soft, act)
        
        aurc = AURC()
        test_aurc = aurc(soft, act)
        
        # Confusion matrix
        cm = confusion_matrix(pred_test, act_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(['0','1','2','3','4','5','6']))
        fig, ax = plt.subplots(figsize=(6, 6), dpi=600)
        disp.plot(ax=ax, cmap='Blues', colorbar=True)
        plt.title(f'{approach} - Confusion Matrix')
        plt.savefig(os.path.join(output_dir, f'{approach}_confusion_matrix.png'), dpi=600, bbox_inches='tight')
        plt.close()
        
        # Reliability plots
        y_test = act_test.astype(np.int64)
        probs = soft_test.cpu().numpy()
        reliability_stats = multiclass_reliability_plots_save(
            y_test, probs, approach, output_dir, n_bins=num_bins, strategy='uniform', show_counts=True)
        
        # Clean up GPU memory
        del model, backbone
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            'approach': approach,
            'test_acc': test_acc * 100,
            'ece': test_ece.item() if torch.is_tensor(test_ece) else test_ece,
            'max_ece': test_max_ece.item() if torch.is_tensor(test_max_ece) else test_max_ece,
            'test_ace': test_ace.item() if torch.is_tensor(test_ace) else test_ace,
            'test_max_ace': test_max_ace.item() if torch.is_tensor(test_max_ace) else test_max_ace,
            'test_aurc': test_aurc.item() if torch.is_tensor(test_aurc) else test_aurc,
        }
        
    except Exception as e:
        print(f"❌ Error evaluating {approach}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'approach': approach,
            'test_acc': None,
            'ece': None,
            'max_ece': None,
            'test_ace': None,
            'test_max_ace': None,
            'test_aurc': None,
        }

# Main evaluation loop
results = []
for approach in approaches:
    result = evaluate_approach(approach, test_dataloader)
    results.append(result)
    print(f"\n{approach} Results:")
    print(f"  Test Acc: {result['test_acc']:.4f}%")
    print(f"  ECE: {result['ece']:.4f}%")
    print(f"  Max ECE: {result['max_ece']:.4f}%")
    print(f"  ACE: {result['test_ace']:.4f}%")
    print(f"  Max ACE: {result['test_max_ace']:.4f}%")
    print(f"  AURC: {result['test_aurc']:.4f}%")

# Save results to CSV
df_results = pd.DataFrame(results)
csv_path = os.path.join(output_dir, 'evaluation_results.csv')
df_results.to_csv(csv_path, index=False)
print(f"\n✅ Results saved to {csv_path}")
print("\nSummary:")
print(df_results.to_string(index=False))