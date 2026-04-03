


import torch.nn as nn

class HandPoseHead(nn.Module):
    def __init__(self, in_channels=3, num_kp=21):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_kp * 2)
        )

    def forward(self, x):
        return self.net(x).view(-1, 21, 2)

import torch
import torch.nn.functional as F

def compute_flexion_angles_torch(kp):
    """
    kp: [B, 21, 2]
    return: [B, 10]
    """

    fingers = [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [13,14,15,16],
        [17,18,19,20]
    ]

    angles = []

    for joints in fingers:
        for i in range(2):
            j1, j2, j3 = joints[i], joints[i+1], joints[i+2]

            v1 = kp[:, j2] - kp[:, j1]
            v2 = kp[:, j3] - kp[:, j2]

            v1 = F.normalize(v1, dim=1)
            v2 = F.normalize(v2, dim=1)

            cos = torch.sum(v1 * v2, dim=1)
            cos = torch.clamp(cos, -1.0, 1.0)

            angles.append(cos)  # use cosine directly

    return torch.stack(angles, dim=1)

def flexion_loss(pred_kp, real_kp):
    pred_angles = compute_flexion_angles_torch(pred_kp)
    real_angles = compute_flexion_angles_torch(real_kp)

    return F.mse_loss(pred_angles, real_angles)
'''
def structural_loss(pred_kp):
    tips = pred_kp[:, [4,8,12,16,20], :]  # [B,5,2]

    pairwise = torch.cdist(tips, tips)
    max_dist = pairwise.max()
    min_dist = pairwise[pairwise > 0].min()

    return F.relu(0.01 - min_dist) + F.relu(max_dist - 0.3)    

def compute_3d_loss_batch(x_hat0, gt_paths):
    losses = []

    for i in range(x_hat0.size(0)):
        try:
            loss_3d, _ = mesh_loss_from_images(x_hat0[i], gt_paths[i])
            losses.append(loss_3d)
        except:
            continue

    if len(losses) == 0:
        return torch.tensor(0.0, device=x_hat0.device)

    return torch.stack(losses).mean()
'''
class HandPoseHead3D(nn.Module):
    def __init__(self, in_channels=3, num_kp=21):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_kp * 3)
        )

    def forward(self, x):
        return self.net(x).view(-1, 21, 3)

def normalize_3d(kp):
    """
    Normalize using wrist → middle finger base
    kp: [B, 21, 3]
    """
    bone = kp[:, 9:10] - kp[:, 0:1]
    scale = bone.norm(dim=2, keepdim=True) + 1e-6
    return kp / scale

# -------- 3D KEYPOINT --------
def get_or_compute_3d_kp(path, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

    file = os.path.join(cache_dir, os.path.basename(path) + ".npy")

    if os.path.exists(file):
        return np.load(file)

    try:
        img = preprocess_image(path)
        joints = get_single_hand_mesh(img)   # ✅ correct
        kp3d = joints.cpu().numpy()
    except Exception as e:
        print(f"[WARN] {path} failed: {e}")
        kp3d = np.zeros((21, 3), dtype=np.float32)

    np.save(file, kp3d)
    return kp3d


def compute_3d_loss(x_hat0, gt_3d, hand_head_3d):
    """
    x_hat0 : [B,3,H,W]
    gt_3d  : [B,21,3]  (precomputed & loaded)
    """

    # Predict 3D joints
    pred_3d = hand_head_3d(x_hat0)   # [B,21,3]

    # Normalize both (VERY IMPORTANT)
    pred_3d = normalize_3d(pred_3d)
    gt_3d = normalize_3d(gt_3d)

    # Mask invalid GT (all zeros)
    valid_mask = (gt_3d.abs().sum(dim=(1,2)) > 0).float()  # [B]

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=x_hat0.device)

    loss = ((pred_3d - gt_3d) ** 2).mean(dim=(1,2))  # per-sample
    loss = (loss * valid_mask).sum() / valid_mask.sum()

    return loss

'''
def compute_3d_loss(x_hat0, gt_3d):
    x_in = preprocess_tensor_for_3d(x_hat0)


    right_list, left_list, trans_list = model(x_in)

    pred = right_list[-1]['joints3d']  # [B,21,3]

    # Normalize (important!)
    pred = pred / (pred[:, 9:10] - pred[:, 0:1]).norm(dim=2, keepdim=True)

    loss = F.mse_loss(pred, gt_3d)

    return loss        
'''    
  
