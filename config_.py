# config.py
import os

n_epoch = 200
batch_size = 4
n_T = 400
device = "cuda:0"
n_classes = 28
n_feat = 256
lrate = 1e-4
save_model = True
save_dir = './adavu_asamyukta_result'
ws_test = [0.0, 0.5, 2.0]  # classifier-free guidance weights
os.makedirs(save_dir, exist_ok=True)

