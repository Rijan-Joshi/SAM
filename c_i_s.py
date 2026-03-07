import torch

from torchvision import models, transforms

import timm

import matplotlib.pyplot as plt

import numpy as np

import cv2

import seaborn as sns

from torchsummary import summary

import os

import torch.nn as nn

import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

device
#Understanding Intersection over Union Concept

ground_truth = torch.zeros(10, 10)

ground_truth[3:7, 3:7] = 1



prediction = torch.zeros(10, 10)

prediction[4:8, 4:8] = 1



intersection = (ground_truth * prediction).sum().item()

union = (ground_truth + prediction).clamp(0, 1).sum().item()

iou = intersection/union



print("Intersection ", intersection)

print("Union ", union)

print("IoU", iou)



#Visualization of the IoU

fig, axes = plt.subplots(1, 4, figsize = (10, 4))



axes[0].imshow(ground_truth, cmap = "Blues", vmin = 0, vmax = 1)

axes[0].set_title("Ground truth")

axes[0].axis('off')



axes[1].imshow(prediction, cmap = "Greens", vmin = 0, vmax = 1)

axes[1].set_title("Prediction")

axes[1].axis('off')



axes[2].imshow(ground_truth * prediction, cmap = "Oranges", vmin = 0, vmax = 1)

axes[2].set_title("Intersection")

axes[2].axis('off')



axes[3].imshow((ground_truth + prediction).clamp(0,1), cmap = "Reds", vmin = 0, vmax = 1)

axes[3].set_title("Union")

axes[3].axis('off')



plt.tight_layout()

plt.title("This is IoU illustration")

plt.show()
#Loading the dataset

import urllib.request

import tarfile

import os



os.makedirs("data", exist_ok = True)



datasets = {

    "data/images.tar.gz": "http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",

    "data/annotations.tar.gz": "http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",

}



for save_path, url in datasets.items():

    print(f"Downloading {save_path}")

    urllib.request.urlretrieve(url, save_path)



    print(f"Extracting {save_path}")

    with tarfile.open(save_path, 'r:gz') as tar:

        tar.extractall(path = "data/")

    print(f"Done! \n")



print("Datasets Ready")
import pathlib



images_dir = pathlib.Path("data/images")

target_dir = pathlib.Path("data/annotations/trimaps")



input_img_paths = sorted(images_dir.glob("*.jpg"))

target_mask_paths = sorted(target_dir.glob("[!.]*.png"))
from PIL import Image



transform = transforms.Compose([

    transforms.ToTensor()

])

img = Image.open(input_img_paths[0]).convert("RGB")



image_tensor = transform(img)

plt.imshow(img)

plt.axis('off')

plt.show()
mask = Image.open(target_mask_paths[0]).convert('L')

plt.imshow(mask)

plt.axis('off')

plt.show()
#Loading into the memory coz the dataset is small

import random



print("Size of dataset", len(input_img_paths))



image_transform = transforms.Compose([

    transforms.Resize((200, 200)),

    transforms.ToTensor()

])



mask_transform = transforms.Compose([

    transforms.Resize((200, 200), interpolation=transforms.InterpolationMode.NEAREST)

])



class SegmentationDataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, image_transform, mask_trasnform):

        self.img_paths = img_paths

        self.mask_paths = mask_paths

        self.image_transform = image_transform

        self.mask_transform = mask_transform



    

    def __len__(self):

        return len(self.img_paths)

    

    def __getitem__(self, index):

        img = Image.open(self.img_paths[index]).convert("RGB")

        img = self.image_transform(img)



        mask = Image.open(self.mask_paths[0]).convert('L')

        mask = torch.from_numpy(np.array(self.mask_transform(mask))).long() - 1



        return img, mask
from torch.utils.data import Dataset, DataLoader



random.Random(1).shuffle(input_img_paths)

random.Random(1).shuffle(target_mask_paths)



train_input_imgs = input_img_paths[:1000]

train_targets = target_mask_paths[:1000]

val_input_imgs = input_img_paths[1000:]

val_targets = target_mask_paths[1000:]



train_dataset = SegmentationDataset(train_input_imgs, train_targets, image_transform, mask_transform)

val_dataset = SegmentationDataset(val_input_imgs, val_targets, image_transform, mask_transform)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = True)
train_dataset[0][1].max(), train_dataset[0][0].max()
#Creating the segmentation model from scratch

from torchsummary import summary

from torch.nn.modules import activation

import torch.nn.functional as F

import torch.nn as nn



class CSModel(nn.Module):

    def __init__(self):

        super().__init__()

        

        #Encoder

        self.Encoder = nn.Sequential(

            #Encoder 1

            nn.Conv2d(3, 64, 3, stride = 2, padding = 1),

            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding = 1),

            nn.ReLU(),



            #Encoder 2

            nn.Conv2d(64, 128, 3, stride = 2, padding = 1),

            nn.ReLU(),

            nn.Conv2d(128,128, 3, padding = 1),

            nn.ReLU(),



            #Encoder 3

            nn.Conv2d(128, 256, 3, stride = 2, padding = 1),

            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding = 1),

            nn.ReLU(),

        )



        #Decoder 

        self.Decoder = nn.Sequential(

            #Decoder 1

            nn.ConvTranspose2d(256, 256, 3, padding = 1), 

            nn.ReLU(),

            nn.ConvTranspose2d(256,256,3, stride = 2, padding = 1, output_padding = 1),

            nn.ReLU(),



            #Decoder 2

            nn.ConvTranspose2d(256,128,3, padding = 1),

            nn.ReLU(),

            nn.ConvTranspose2d(128,128,3, stride = 2, padding = 1, output_padding = 1),

            nn.ReLU(),



            #Decoder 3

            nn.ConvTranspose2d(128,64,3, padding = 1),

            nn.ReLU(),

            nn.ConvTranspose2d(64,64,3, stride = 2, padding = 1, output_padding = 1),

            nn.ReLU(),

        )



        self.output = nn.Conv2d(64, 3, 3, padding = 1)



    

    def forward(self, X, inference = False):

        X = self.Encoder(X)

        X = self.Decoder(X)

        X = self.output(X)

        if inference: 

            X = F.softmax(X, dim = 1)

        return X



model = CSModel().to(device)

summary(model, (3, 200, 200))
#Intersection over Union Implementation 



class IoU:

    def __init__(self, target_class = 0, num_classes = 3):

        self.target_class = target_class

        self.num_classes = num_classes

        self.reset()

        

    def reset(self):

        self.intersections = 0

        self.unions = 0



    def update(self, outputs, targets):

        

        preds = torch.argmax(outputs, dim = 1)



        pred_mask = (preds == self.target_class)

        target_mask  = (targets == self.target_class)



        intersection = (pred_mask & target_mask).sum().item()

        union = (pred_mask | target_mask).sum().item()



        self.intersections += intersection

        self.unions += union



    def compute(self):

        if self.unions == 0:

            return float('nan')

        return self.intersections / self.unions
#Traning and Validation the model

from torch.optim import Adam



optimizer = Adam(model.parameters(), lr = 1e-3)

criterion = nn.CrossEntropyLoss()

epochs  = 50



train_iou = IoU(0, 3)

val_iou = IoU(0, 3)



history = {

    "loss":[],

    "val_loss":[],

    "foreground_iou":[],

    "val_foreground_iou":[]

}



best_val_loss = float('inf')

checkpoint_path = "best_custom_segmentation.pt"



#Training Model



for epoch in range(1, epochs + 1):

    model.train()

    train_iou.reset()



    epoch_losses = []



    for images, targets in train_loader:

        images = images.to(device)

        targets = targets.to(device)



        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()



        epoch_losses.append(loss.item())

        train_iou.update(outputs, targets)

    

    #Validation Phase

    model.eval()

    val_losses = []

    val_iou.reset()

    

    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(device)

            targets = targets.to(device)



            outputs = model(images)

            loss = criterion(outputs, targets)



            val_losses.append(loss.item())

            val_iou.update(outputs, targets)

    

    train_loss = np.mean(epoch_losses)

    val_loss = np.mean(val_losses)

    train_iou_score = train_iou.compute()

    val_iou_score = val_iou.compute()



    history['loss'].append(train_loss)

    history['val_loss'].append(val_loss)

    history['foreground_iou'].append(train_iou_score)

    history['val_foreground_iou'].append(val_iou_score)



    print(f"Epoch {epoch} / {epochs}")

    print(f" Train Loss: {train_loss:.4f}, IoU: {train_iou_score:.4f}")

    print(f" Train Loss: {val_loss:.4f}, IoU: {val_iou_score:.4f}")



    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(model.state_dict(), checkpoint_path)

        print(f"Saved the best model at {checkpoint_path}")



    print()

#Plotting Traning History 



epochs = range(1, len(history['loss']) + 1)



plt.figure(figsize=(12, 4))



plt.subplot(1,2,1)

plt.plot(epochs, history['loss'], 'r--', label = "Training Loss")

plt.plot(epochs, history['val_loss'], 'b', label = "Validation Loss")

plt.title("Training and Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True, alpha = 0.3)



plt.subplot(1,2,2)

plt.plot(epochs, history['foreground_iou'], 'r--', label = "Training Foreground IoU")

plt.plot(epochs, history['val_foreground_iou'], 'b', label = "Validation Foreground IoU")

plt.title("Training and Validation Foreground IoU")

plt.xlabel("Epoch")

plt.ylabel("IoU")

plt.legend()

plt.grid(True, alpha = 0.3)



plt.tight_layout()

plt.show()
#Load best model and predict the mask



model.load_state_dict(torch.load(checkpoint_path, map_location = device))

model.eval()



def predict(model, image, device):

    

    with torch.no_grad():

        image = image.unsqueeze(0).to(device)

        logits = model(image)

        pred = torch.argmax(logits, dim = 1)

    return pred.squeeze(0).cpu().numpy()



def display_prediction(dataset, idx, model, device):

    """Display image, ground truth, and prediction"""

    img, target = dataset[idx]

    

    # Get prediction

    pred_mask = predict(model, img, device)

    

    plt.figure(figsize=(15, 4))

    

    # Input image

    plt.subplot(1, 3, 1)

    plt.axis("off")

    plt.title("Input")

    plt.imshow(img.permute(1, 2, 0).numpy())

    

    # Ground truth

    plt.subplot(1, 3, 2)

    plt.axis("off")

    plt.title("Ground Truth")

    gt_vis = target.numpy() * 127

    plt.imshow(gt_vis, cmap='gray')

    

    # Prediction

    plt.subplot(1, 3, 3)

    plt.axis("off")

    plt.title("Prediction")

    pred_vis = pred_mask * 127

    plt.imshow(pred_vis, cmap='gray')

    

    plt.tight_layout()

    plt.show()



# Visualize a few validation samples

for i in [4, 10, 25]:

    display_prediction(val_dataset, i, model, device)
img, target = train_dataset[20]



plt.figure(figsize=(15, 4))

    

# Input image

plt.subplot(1, 3, 1)

plt.axis("off")

plt.title("Input")

plt.imshow(img.permute(1, 2, 0).numpy())



# Ground truth

plt.subplot(1, 3, 2)

plt.axis("off")

plt.title("Ground Truth")

gt_vis = target.numpy() * 127

plt.imshow(gt_vis, cmap='gray')



