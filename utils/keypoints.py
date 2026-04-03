
from torch.utils.data import Dataset
from torchvision import datasets
import torch
import os
import numpy as np
import cv2
import mediapipe as mp
from torchvision.datasets import ImageFolder



kp_cache_dir = "./hasta_kp_cahe"
hasta_3d_kp_cahe = "./hasta_3d_kp_cahe"
#data_path = "./asamyukta_dataset"
data_path = "./Bharatanatyam_hasta_mudra_dataset"

dataset = ImageFolder(root=data_path)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)


import os
import cv2
import numpy as np
import torch 

# -------- 2D HAND KEYPOINT --------
def extract_hand_keypoints_np(image, hands):
    h, w, _ = image.shape
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_hand_landmarks:
        return np.zeros((21, 2), dtype=np.float32)

    hand = results.multi_hand_landmarks[0]
    return np.array([[lm.x * w, lm.y * h] for lm in hand.landmark], dtype=np.float32)


def get_or_compute_2d_kp(path, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    file = os.path.join(cache_dir, os.path.basename(path) + ".npy")

    if os.path.exists(file):
        return np.load(file)

    img = cv2.imread(path)
    if img is None:
        kp = np.zeros((21, 2), dtype=np.float32)
    else:
        kp = extract_hand_keypoints_np(img, hands)

    np.save(file, kp)
    return kp


import os
import sys
sys.path.append('./Two-Hand-Shape-Pose')  # replace with your actual path
import torch
import numpy as np
import cv2

from network.full_model import InterShape


# ---- INITIALIZE MODEL ONCE ----
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 256

# Load InterShape model
model = InterShape(
    input_size=3,
    resnet_version=50,
    mano_neurons=[512, 512, 512, 512],
    mano_use_pca=False,
    cascaded_num=3,
    cascaded_input='double',
    heatmap_attention=True
)
MODEL_PATH = "./Two-Hand-Shape-Pose/model/model.pts"
para_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(para_dict, strict=False)
model.to(DEVICE)
model.eval()
print("[✓] 3D Hand Model loaded.")


def preprocess_image(img_path, input_size=INPUT_SIZE):
    """Load and preprocess image for InterShape model"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    ratio = input_size / max(*img.shape[:2])
    M = np.array([[ratio, 0, 0], [0, ratio, 0]], dtype=np.float32)
    img_resized = cv2.warpAffine(img, M, (input_size, input_size), flags=cv2.INTER_LINEAR, borderValue=[0,0,0])
    img_rgb = img_resized[:, :, ::-1].astype(np.float32)/255.0 - 0.5
    input_tensor = torch.tensor(img_rgb.transpose(2,0,1), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    return input_tensor


def get_single_hand_mesh(input_tensor):
    with torch.no_grad():
        right_list, left_list, trans_list = model(input_tensor)

        right = right_list[-1]
        left = left_list[-1]

        # lengths
        r_len = (right['joints3d'][:, 9] - right['joints3d'][:, 0]).norm(dim=1)
        l_len = (left['joints3d'][:, 9] - left['joints3d'][:, 0]).norm(dim=1)

        # choose valid hand
        if r_len.item() > l_len.item():
            joints = right['joints3d'] / r_len[:, None, None]
        else:
            joints = left['joints3d'] / l_len[:, None, None]

    return joints[0]   # (21,3)

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


from torchvision.datasets import ImageFolder

dataset = ImageFolder(root="./Bharatanatyam_hasta_mudra_dataset")


for i, (path, _) in enumerate(dataset.samples):
    print(f"[{i}/{len(dataset)}] {path}")
    get_or_compute_2d_kp(path, "/home/jagdish/hasta_kp_cahe")


for i, (path, _) in enumerate(dataset.samples):
    print(f"[{i}/{len(dataset)}] {path}")
    get_or_compute_3d_kp(path, "/home/jagdish/hasta_3d_kp_cahe")

#for i, (path, _) in enumerate(dataset.samples):
#    print(f"[2D KP {i}/{len(dataset)}] {path}")
#    get_or_compute_keypoints(path, kp_cache_dir)    
