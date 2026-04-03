from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from scipy import io
from torchvision.datasets import ImageFolder


#Imports
import os
import torch
import torch.nn as nn
import numpy as np
#import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.nn.functional import softmax
from scipy.linalg import sqrtm
from torchvision.transforms import functional as F
import mediapipe as mp
import numpy as np
import cv2
import os
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torch.utils.data import DataLoader
#from models.pose_head import PoseHead
import torch.nn.functional as F
from data.dataset import AsamyuktaHandDataset,train_dataset
from models.unet import ContextUnet
from losses.pose_loss import flexion_loss, compute_flexion_angles_torch, HandPoseHead, compute_3d_loss, HandPoseHead3D
from utils.keypoints import extract_hand_keypoints_np, get_or_compute_2d_kp, get_or_compute_3d_kp
from models.ddpm import DDPM, ddpm_schedules
from config_ import n_feat, n_classes, n_T, batch_size, lrate, n_epoch
from utils.visualize import denormalize
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ws_test = [0.0, 0.5, 2.0]  # classifier-free guidance weights

save_dir="./adavu_asamyukta_hand"

def train_adavu(): 
 
    # Model setup
    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.0
    ).to(device)


    #ddpm.load_state_dict(torch.load("./adavu_asamyukta_result/model_128_102342.pth"    ))

    # Image transform
    img_size = 128
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    # Loss weights
    lambda_kp = 1.0
    lambda_angle = 0.5
    lambda_mesh = 0.2

    # --- TRAIN LOOP ---
    for ep in range(n_epoch):
        print(f"\nEpoch {ep}/{n_epoch}")
        ddpm.train()


        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)
        pbar = tqdm(dataloader)
        loss_ema = None

        hand_head_3d = HandPoseHead3D().to(device)
        pose_head = HandPoseHead().to(device)

        for batch_idx, (x0, c, real_kp, real_kp_3d) in enumerate(pbar):

            x0 = x0.to(device)
            c = c.to(device)
            real_kp = real_kp.to(device)

            B = x0.size(0)

            # ---- Diffusion ----
            t = torch.randint(1, n_T + 1, (B,), device=device)
            eps = torch.randn_like(x0)
 
            sqrt_abar_t = ddpm.sqrtab[t][:, None, None, None]
            sqrt_1mabar_t = ddpm.sqrtmab[t][:, None, None, None]

            x_t = sqrt_abar_t * x0 + sqrt_1mabar_t * eps

            context_mask = torch.zeros_like(c).to(device)

            eps_pred = ddpm.nn_model(x_t, c, t / n_T, context_mask)

            loss_diff = F.mse_loss(eps_pred, eps)

            # ---- Reconstruction ----
            x_hat0 = (x_t - sqrt_1mabar_t * eps_pred) / sqrt_abar_t
            x_hat0 = torch.clamp(x_hat0, -1, 1)

            # ---- Pose Prediction (DIFFERENTIABLE) ----
            pred_hand_kp = pose_head(x_hat0)

            # ---- Keypoint Loss ----
            hand_kp_loss = F.mse_loss(pred_hand_kp, real_kp)

            # flexio_Angle loss (GEOMETRIC 🔥)
            # -------------------------------
            #angle_pred = compute_flexion_angles(pred_kp)
            #angle_gt = compute_flexion_angles(real_kp)
            
            hand_flex_loss = flexion_loss(pred_hand_kp, real_kp)
            
            # ---- Keypoint Loss ----
            hand_kp_loss = F.mse_loss(pred_hand_kp, real_kp)

            # ---- 3D Keypoint(pose) Loss ----
            real_kp_3d = torch.tensor(real_kp_3d, dtype=torch.float32)
            real_kp_3d = real_kp_3d.to(device)
            loss_3d = compute_3d_loss(x_hat0, real_kp_3d, hand_head_3d)
            #loss_3d = compute_3d_loss(x_hat0, real_kp_3d)
            # -------------------------------
            # TOTAL LOSS
            # -------------------------------
            total_loss = (loss_diff + 1.0 * hand_kp_loss + 0.5 * hand_flex_loss + 0.3 * loss_3d)

            optim.zero_grad()
            total_loss.backward()
            optim.step()

            pbar.set_description(f"loss: {total_loss.item():.4f}")
        
        # --- Evaluation (image generation) ---
        
        ddpm.eval()
        with torch.no_grad():
            print(f"[Eval] Sampling epoch {ep}")

            real_batch = next(iter(dataloader))
            x_real = real_batch[0][:n_classes].to(device)

            for w in ws_test:
                labels = torch.arange(0, n_classes).to(device)

                x_gen, _ = ddpm.sample(n_sample=n_classes,size=(3, 128, 128),device=device,labels=labels,guide_w=w)

                # Save individual images
            for idx in range(n_classes):
                img = torch.clamp(denormalize(x_gen[idx]), 0, 1)
                save_path = os.path.join(save_dir, f"class_{ep}_{idx}_w{w}.png")
                save_image(img, save_path)

            # Grid
            x_all = torch.cat([x_gen, x_real])
            x_all = torch.clamp(denormalize(x_all), 0, 1)

            grid = make_grid(x_all, nrow=n_classes)
            grid_path = os.path.join(save_dir, f"image_ep{ep}_w{w}.png")

            save_image(grid, grid_path)
            print(f"Saved grid at {grid_path}")    
        # --- Save model ---
        if save_model and ep % 10 == 0:
            ckpt_path = os.path.join(save_dir, f"model_128_256{ep}.pth")
            torch.save(ddpm.state_dict(), ckpt_path)
            print(f"Saved model at {ckpt_path}")                   
                  
       
if __name__ == "__main__":
    train_adavu()
