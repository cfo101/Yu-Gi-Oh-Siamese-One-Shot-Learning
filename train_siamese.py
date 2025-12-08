from __future__ import annotations
import os
import random
import math
from typing import Optional
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import open_clip
from tqdm import tqdm


DATA_DIR = "dataset_split"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 20        
LR = 1e-4


#training augmentations
augment = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(8),
    transforms.ColorJitter(0.10, 0.10, 0.10, 0.05),
])


#preprocess
def load_clip_transform():
    _, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    return preprocess

clip_preprocess = load_clip_transform()


#dataset
class SiameseDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.files = []
        self.by_card = {}

        for f in os.listdir(root_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                cid = f.split("_")[0]
                self.files.append(f)
                self.by_card.setdefault(cid, []).append(f)

        self.card_ids = list(self.by_card.keys())

    def __len__(self):
        return len(self.files)

    def load_image(self, path):
        try:
            return Image.open(path).convert("RGB")
        except:
            return None

    def synthetic_positive(self, img):
        return augment(img)

    def __getitem__(self, idx):
        anchor_file = self.files[idx]
        anchor_id = anchor_file.split("_")[0]
        anchor_path = os.path.join(self.root, anchor_file)

        anchor_img = self.load_image(anchor_path)
        if anchor_img is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        #positives
        same_imgs = self.by_card[anchor_id]
        if len(same_imgs) > 1:
            pos_file = random.choice([f for f in same_imgs if f != anchor_file])
            pos_img = self.load_image(os.path.join(self.root, pos_file))
            if pos_img is None:
                pos_img = self.synthetic_positive(anchor_img)
        else:
            pos_img = self.synthetic_positive(anchor_img)

        #negatives
        neg_id = random.choice([cid for cid in self.card_ids if cid != anchor_id])
        neg_file = random.choice(self.by_card[neg_id])
        neg_img = self.load_image(os.path.join(self.root, neg_file))
        if neg_img is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        
        return (
            clip_preprocess(anchor_img),
            clip_preprocess(pos_img),
            clip_preprocess(neg_img),
        )


#MODEL
def build_model():
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    model = model.to(DEVICE)
    model.eval()

    # Freeze CLIP encoder
    for p in model.parameters():
        p.requires_grad = False

    # Trainable projection head
    proj = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 128)
    ).to(DEVICE)

    return model, proj



def compute_margin(epoch, total_epochs, m_min=0.10, m_max=0.60):
    """Cosine-annealed margin schedule."""
    progress = epoch / total_epochs
    return m_min + 0.5 * (m_max - m_min) * (1 - math.cos(math.pi * progress))



def train_siamese():
    print("Loading CLIP model...")
    model, proj = build_model()

    dataset = SiameseDataset(TRAIN_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(proj.parameters(), lr=LR)

    print("Starting training...")
    for epoch in range(EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        running_loss = 0.0

        # Dynamic margin for this epoch
        margin = compute_margin(epoch, EPOCHS)
        triplet_loss = nn.TripletMarginLoss(margin=margin)

        print(f"Using margin = {margin:.4f}")

        for anchor_img, pos_img, _ in loop:
            anchor_img = anchor_img.to(DEVICE, non_blocking=True)
            pos_img = pos_img.to(DEVICE, non_blocking=True)

            
            with torch.no_grad():
                a = model.encode_image(anchor_img)
                p = model.encode_image(pos_img)

            # Project + normalize
            a = nn.functional.normalize(proj(a), dim=1)
            p = nn.functional.normalize(proj(p), dim=1)

            
            dist_matrix = torch.cdist(a, a, p=2)
            mask = ~torch.eye(a.size(0), dtype=bool, device=DEVICE)
            hard_neg_idx = dist_matrix[mask].reshape(a.size(0), -1).argmin(dim=1)
            n = a[hard_neg_idx]

           
            loss = triplet_loss(a, p, n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / (loop.n + 1e-8))

    print("Training complete!")
    torch.save(proj.state_dict(), "siamese_proj.pth")
    print("Projection head saved.")


if __name__ == "__main__":
    train_siamese()