# -*- coding: utf-8 -*-
"""slam_project_main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1l02Z6Vah06haonNdTcur7S0rl3Pknwcr

#Get Data
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import shutil
if not os.path.exists("data"):
  os.mkdir("data")

shutil.copy("/content/drive/MyDrive/SLAM/rgbd_dataset_freiburg2_pioneer_slam.tgz", "/content/data/rgbd_dataset_freiburg2_pioneer_slam.tgz")

cd data/

!tar -xvf /content/data/rgbd_dataset_freiburg2_pioneer_slam.tgz



"""#Imports"""

# importing required libraries
import pickle
import numpy as np
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import torchvision.transforms as transforms

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import os
import pandas as pd
from torchvision.io import read_image

from torch.utils.data import Dataset

from utils import *
from data_factory import *
from networks import *

"""#Read Data"""

rgb_images = os.listdir("/content/data/rgbd_dataset_freiburg2_pioneer_slam/rgb")
rgb_images.sort()
depth_images = os.listdir("/content/data/rgbd_dataset_freiburg2_pioneer_slam/depth")
depth_images.sort()

len(rgb_images), len(depth_images)

rgb_images[0]

root_dir = "/content/data/rgbd_dataset_freiburg2_pioneer_slam/"
show_image(root_dir+"/rgb/", rgb_images[-1])

for i in range(5):
  show_image(root_dir+"/rgb/", rgb_images[i])

for i in range(5):
  show_image(root_dir+"/depth/", depth_images[i])

"""#Plot Ground Truth

Quaternion
"""

gt = np.loadtxt("/content/data/rgbd_dataset_freiburg2_pioneer_slam/groundtruth.txt")

gt.shape

44212/15.3

timestamps = gt[:, 0]
t = gt[:, 1:4]
q = gt[:, 4:]

timestamps[0]

t[0]

odometry = t[1:, :] - t[:-1, :]
odometry.shape

np.around(odometry[:10], 4)

q[0]

SO3(q[0], t[0])

new_t = []
for j in range(0, t.shape[0], 16):
  new_t.append(t[j])
new_t = np.array(new_t)
t = new_t

gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(t[:,0], t[:,1], t[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth Trajectory')
ax.set_xlim(min(t[:, 0]), max(t[:, 0]))
ax.set_ylim(min(t[:, 1]), max(t[:, 1]))
ax.set_zlim(min(t[:, 2]), max(t[:, 2]))
plt.show()
plt.clf()

gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(t[:,0], t[:,1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
ax.set_title('Ground Truth Trajectory')
ax.set_xlim(min(t[:, 0]), max(t[:, 0]))
ax.set_ylim(min(t[:, 1]), max(t[:, 1]))
# ax.set_zlim(min(t[:, 2]), 0.58)
plt.show()
plt.clf()

gt_fig = plt.figure()
ax = gt_fig.add_subplot(111)
ax.plot(t[:,0], t[:,1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
ax.set_title('Ground Truth Trajectory')
# ax.set_xlim(min(t[:, 0]), max(t[:, 0]))
# ax.set_ylim(min(t[:, 1]), max(t[:, 1]))
# ax.set_zlim(min(t[:, 2]), 0.58)
plt.scatter(t[0, 0], t[0, 1], marker='o', color='b', label='Start')
plt.scatter(t[-1, 0], t[-1, 1], marker='o', color='r', label='End')
plt.legend()
plt.show()
plt.clf()

new_t.shape

"""#Build ML Pipeline"""

root_dir = "/content/data/rgbd_dataset_freiburg2_pioneer_slam"

train_data = SLAM_DATA(root_dir, scale_down=4)

out = train_data.__getitem__(0)

out[0][0].shape, out[0][1].shape, out[1], out[-1]

rgb_text = np.loadtxt("/content/data/rgbd_dataset_freiburg2_pioneer_slam/rgb.txt", dtype=str)
depth_text = np.loadtxt("/content/data/rgbd_dataset_freiburg2_pioneer_slam/depth.txt", dtype=str)
acc_text = np.loadtxt("/content/data/rgbd_dataset_freiburg2_pioneer_slam/accelerometer.txt", dtype=str)
gt_text = np.loadtxt("/content/data/rgbd_dataset_freiburg2_pioneer_slam/groundtruth.txt", dtype=str)

rgb_text.shape, depth_text.shape, acc_text.shape, gt_text.shape

print(rgb_text[0])
print(depth_text[0])
print(acc_text[0])
print(gt_text[0])

show_image('/content/data/rgbd_dataset_freiburg2_pioneer_slam/'+rgb_text[0][-1], "")
show_image('/content/data/rgbd_dataset_freiburg2_pioneer_slam/'+depth_text[0][-1], "")

"""Dataloader"""

train_dataloader = DataLoader(train_data, batch_size=12, shuffle=True)

"""Model"""

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = VPT().to(device)
print(model)

"""#Test Training"""

train_data = SLAM_DATA(root_dir, scale_down=4)

train_dataloader = DataLoader(train_data, batch_size=12, shuffle=True)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = VPT_1().to(device)
print(model)

loss_R = nn.CrossEntropyLoss()
loss_t = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

size = len(train_data)
all_losses = []
losses =[]
best_loss = 100
model.train()
for i in range(13, size):
  start, stop = i-13, i-1
  rgbs = torch.zeros((12, 3, 120, 160))
  depths = torch.zeros((12, 1, 120, 160))
  odos = torch.zeros((12, 3))
  Ts = torch.zeros((12, 4, 4))
  for j in range(start, stop):
    ((rgb, depth), odo, T) = train_data.__getitem__(j)

    rgbs[j-start, :, :, :] = rgb
    depths[j-start, :, : ,:] = depth
    odos[j-start, :] = odo
    Ts[j-start, :, :] = T
  
  dR, dt = model(rgbs, odos, depths)
  print(dR)
  print(dt)
  break

  _, _, T_now = train_data.__getitem__(i)
  R = T[:3, :3]
  t = T[:3, -1]

  # use latest T variable for immediately previous pose
  R_pre = T[:3, :3].float()
  t_pre = T[:3, -1].float()

  # get prediction
  R_new = dR @ R_pre
  t_new = dt + t_pre

  # loss
  loss = 0.75*loss_R(torch.flatten(R_new), torch.flatten(R)) + 0.25*loss_t(torch.flatten(t_new), torch.flatten(t))
  # print("loss: ", loss.item())
  all_losses.append(loss)
    

  # Backpropagation
  optimizer.zero_grad()
  loss.backward(retain_graph=True)
  optimizer.step()

  if loss.item() < best_loss:
    best_loss = loss.item()
    torch.save(model, "best_model.pt")
    # print("===Model saved===")

  if i % 100 == 0:
    loss, current = loss.item(), i
    losses.append(loss)
    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

  # break

losses

size = len(train_data)
for i in range(13, size):
  start, stop = i-13, i-1
  rgbs = torch.zeros((12, 3, 120, 160))
  depths = torch.zeros((12, 1, 120, 160))
  odos = torch.zeros((12, 3))
  Ts = torch.zeros((12, 4, 4))
  for j in range(start, stop):
    ((rgb, depth), odo, T) = train_data.__getitem__(j)
    # print(rgb.shape)
    # print(depth.shape)
    # print(odo)
    # print(T)
    # print(j)
    rgbs[j, :, :, :] = rgb
    depths[j, :, : ,:] = depth
    odos[j, :] = odo
    Ts[j, :, :] = T
  print(rgbs.shape)
  print(depths.shape)
  print(odos)
  print(Ts.shape)
  break

rnn = nn.GRU(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)

rnn = nn.GRU(27, 12, 1)
input = torch.randn(12, 27)
h0 = torch.randn(1, 12)
output, hn = rnn(input, h0)

output.shape

hn.shape

lines = np.load('/content/all_losses.npy', allow_pickle=True)

plt.plot(lines)

"""#RNN Training"""

root_dir = "/content/data/rgbd_dataset_freiburg2_pioneer_slam"
train_data = SLAM_DATA(root_dir, scale_down=4)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VPT_RNN_4().to(device)

loss_R = nn.MSELoss()
loss_t = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

def get_samples(i, data):
  start, stop = i-13, i-1
  rgbs = torch.zeros((12, 3, 120, 160))
  depths = torch.zeros((12, 1, 120, 160))
  odos = torch.zeros((12, 3))
  Ts = torch.zeros((12, 4, 4))
  for j in range(start, stop):
    ((rgb, depth), odo, T) = data.__getitem__(j)

    rgbs[j-start, :, :, :] = torch.from_numpy(rgb)
    depths[j-start, :, : ,:] = torch.from_numpy(depth)
    odos[j-start, :] = torch.from_numpy(odo)
    Ts[j-start, :, :] = torch.from_numpy(T)

  return rgbs, depths, odos, Ts

size = len(train_data)
all_losses = []
losses =[]
best_loss = 100
model.train()
for i in range(size):
    rgbs, depths, odos, Ts = get_samples(i, train_data)
    rgbs = rgbs.to(device)
    depths = depths.to(device)
    odos = odos.to(device)
    Ts = Ts.to(device)
    
    dR, dt = model(rgbs, odos, depths)
    # print(dR)
    # print(dt)
    # break

    _, _, T_now = train_data.__getitem__(i)
    R = T_now[:3, :3].to(device)
    t = T_now[:3, -1].to(device)

    # use latest T variable for immediately previous pose
    # R_pre = T[:3, :3].float()
    # t_pre = T[:3, -1].float()

    # get prediction
    R_new = dR @ Ts[-1, :3, :3].float()
    t_new = dt + Ts[-1, :3, -1].float()

    # loss
    loss = 0.75*loss_R(torch.flatten(R_new), torch.flatten(R)) + 0.25*loss_t(torch.flatten(t_new), torch.flatten(t))

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    # print("loss: ", loss.item())
    all_losses.append(loss.detach().cpu())
    np.save("all_losses_vptrnn4_.npy", all_losses)
        
    if loss.item() < best_loss:
      best_loss = loss
      torch.save(model, "vpt_rnn_4_best_model.pt")
      # print("===Model saved===")

    if i % 100 == 0:
      loss, current = loss.item(), i
      # losses.append(loss.detach().cpu())
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

      # break

size = len(train_data)
all_losses = []
losses =[]
best_loss = 100
model.train()
for i in range(size):
    rgbs, depths, odos, Ts = get_samples(i, train_data)
    rgbs = rgbs.to(device)
    depths = depths.to(device)
    odos = odos.to(device)
    Ts = Ts.to(device)
    
    _, dt = model(rgbs, odos, depths)
    # print(dR)
    # print(dt)
    # break

    _, _, T_now = train_data.__getitem__(i)
    # R = T_now[:3, :3].to(device)
    t = T_now[:3, -1].to(device)

    # use latest T variable for immediately previous pose
    # R_pre = T[:3, :3].float()
    # t_pre = T[:3, -1].float()

    # get prediction
    # R_new = dR @ Ts[-1, :3, :3].float()
    t_new = dt + Ts[-1, :3, -1].float()

    # loss
    loss = 0.25*loss_t(torch.flatten(t_new), torch.flatten(t)) # + 0.75*loss_R(torch.flatten(R_new), torch.flatten(R)) + 

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    # print("loss: ", loss.item())
    all_losses.append(loss.detach().cpu())
    np.save("all_losses_vptrnn3.npy", all_losses)
        
    if loss.item() < best_loss:
      best_loss = loss
      torch.save(model, "vpt_rnn_3_best_model.pt")
      # print("===Model saved===")

    if i % 100 == 0:
      loss, current = loss.item(), i
      # losses.append(loss.detach().cpu())
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

      # break

"""#Train_Val Training"""

train_data = SLAM_DATA("/content/data/rgbd_dataset_freiburg2_pioneer_slam", scale_down=4)
val_data = SLAM_DATA("/content/data/rgbd_dataset_freiburg2_pioneer_slam2", scale_down=4)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VPT_RNN_4().to(device)
loss_R = nn.MSELoss()
loss_t = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

def get_samples(i, data):
  start, stop = i-13, i-1
  rgbs = torch.zeros((12, 3, 120, 160))
  depths = torch.zeros((12, 1, 120, 160))
  odos = torch.zeros((12, 3))
  Ts = torch.zeros((12, 4, 4))
  for j in range(start, stop):
    ((rgb, depth), odo, T) = data.__getitem__(j)

    rgbs[j-start, :, :, :] = rgb
    depths[j-start, :, : ,:] = depth
    odos[j-start, :] = odo
    Ts[j-start, :, :] = T

  return rgbs, depths, odos, Ts

size = len(val_data)
train_losses = val_losses = []
# losses =[]
best_loss = 100
model.train()
for i in range(size):
    rgbs_train, depths_train, odos_train, Ts_train = get_samples(i, train_data)
    rgbs_train = rgbs_train.to(device)
    depths_train = depths_train.to(device)
    odos_train = odos_train.to(device)
    Ts_train = Ts_train.to(device)
    
    dR, dt = model(rgbs_train, odos_train, depths_train)
    # print(dR)
    # print(dt)
    # break

    _, _, T_now = train_data.__getitem__(i)
    R = T_now[:3, :3].to(device)
    t = T_now[:3, -1].to(device)

    # use latest T variable for immediately previous pose
    # R_pre = T[:3, :3].float()
    # t_pre = T[:3, -1].float()

    # get prediction
    R_new = dR @ Ts_train[-1, :3, :3].float()
    t_new = dt + Ts_train[-1, :3, -1].float()

    # loss
    train_loss = 0.75*loss_R(torch.flatten(R_new), torch.flatten(R)) + 0.25*loss_t(torch.flatten(t_new), torch.flatten(t))

    optimizer.zero_grad()
    train_loss.backward(retain_graph=True)
    optimizer.step()

    # print("loss: ", loss.item())
    train_losses.append(train_loss.detach().cpu())
    np.save("train_losses_vptrnn4_.npy", train_losses)

    #=======================VAL=======================#
    rgbs_val, depths_val, odos_val, Ts_val = get_samples(i, val_data)
    rgbs_val = rgbs_val.to(device)
    depths_val = depths_val.to(device)
    odos_val = odos_val.to(device)
    Ts_val = Ts_val.to(device)
    
    dR, dt = model(rgbs_val, odos_val, depths_val)
    # print(dR)
    # print(dt)
    # break

    _, _, T_now = val_data.__getitem__(i)
    R = T_now[:3, :3].to(device)
    t = T_now[:3, -1].to(device)

    # use latest T variable for immediately previous pose
    # R_pre = T[:3, :3].float()
    # t_pre = T[:3, -1].float()

    # get prediction
    R_new = dR @ Ts_val[-1, :3, :3].float()
    t_new = dt + Ts_val[-1, :3, -1].float()

    # loss
    val_loss = 0.75*loss_R(torch.flatten(R_new), torch.flatten(R)) + 0.25*loss_t(torch.flatten(t_new), torch.flatten(t))

    val_losses.append(val_loss.detach().cpu())
    np.save("val_losses_vptrnn4_.npy", val_losses)
        
    if val_loss.item() < best_loss:
      best_loss = val_loss
      torch.save(model, "vpt_rnn_4_best_model.pt")
      # print("===Model saved===")

    if i % 100 == 0:
      tloss, vloss, current = train_loss.item(), val_loss.item(), i
      # losses.append(loss.detach().cpu())
      print(f"train loss: {tloss:>7f} val loss: {vloss:>7f} [{current:>5d}/{size:>5d}]")

      # break

"""#Testing"""

shutil.copy("/content/drive/MyDrive/SLAM/rgbd_dataset_freiburg2_pioneer_slam2.tgz", "/content/data/rgbd_dataset_freiburg2_pioneer_slam2.tgz")

!tar -xvf /content/data/rgbd_dataset_freiburg2_pioneer_slam2.tgz

root_dir = "/content/data/rgbd_dataset_freiburg2_pioneer_slam2"
test_data = SLAM_DATA(root_dir, scale_down=4)

((one, two), three, four) = test_data.__getitem__(0)

one.shape, two.shape, three, four

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("/content/vpt_rnn_4_val_best_model.pt", map_location=device)
model.eval()

size = len(val_data)
pred_gt = []
pred_r = []
for i in range(13, size):
  start, stop = i-13, i-1
  rgbs = torch.zeros((12, 3, 120, 160))
  depths = torch.zeros((12, 1, 120, 160))
  odos = torch.zeros((12, 3))
  Ts = torch.zeros((12, 4, 4))
  for j in range(start, stop):
    ((rgb, depth), odo, T) = val_data.__getitem__(j)

    rgbs[j-start, :, :, :] = rgb
    depths[j-start, :, : ,:] = depth
    odos[j-start, :] = odo
    Ts[j-start, :, :] = T

  # print(rgbs.shape)
  # print(depths.shape)
  # print(odos.shape)
  # print(Ts.shape)
  rgbs = rgbs.to(device)
  depths = depths.to(device)
  odos = odos.to(device)
  Ts = Ts.to(device)
  dR, dt = model(rgbs, odos, depths)
  # print(dR)
  # print(dt)
  # break

  _, _, T_now = val_data.__getitem__(i)
  R = T[:3, :3]
  t = T[:3, -1]

  # use latest T variable for immediately previous pose
  R_pre = T[:3, :3].float()
  t_pre = T[:3, -1].float()
  R_pre = R_pre.to(device)
  t_pre = t_pre.to(device)

  # get prediction
  dR = dR.to(device)
  dt = dt.to(device)
  R_new = dR @ R_pre
  t_new = dt + t_pre

  pred_gt.append(t_new.detach().cpu())
  pred_r.append(R_new.detach().cpu())

  if i % 500 == 0:
    print(i)

x = [t.numpy() for t in pred_gt]
np.save("pred_gt_val_rnn4.npy", x)

x = [t.numpy() for t in pred_r]
np.save("pred_r_rnn4.npy", x)

pred = np.load("pred_gt_val_rnn4.npy")
pred[:5, :]

gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(pred[:,0], pred[:,1], pred[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth Trajectory')
ax.set_xlim(min(pred[:, 0]), max(pred[:, 0]))
ax.set_ylim(min(pred[:, 1]), max(pred[:, 1]))
ax.set_zlim(min(pred[:, 2]), max(pred[:, 2]))
plt.show()
plt.clf()

gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(pred[:,0], pred[:,1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
ax.set_title('Ground Truth Trajectory')
ax.set_xlim(min(pred[:, 0]), max(pred[:, 0]))
ax.set_ylim(min(pred[:, 1]), max(pred[:, 1]))
# ax.set_zlim(min(pred[:, 2]), max(pred[:, 2]))
plt.show()
plt.clf()

gt_fig = plt.figure()
ax = gt_fig.add_subplot(111)
ax.plot(pred[:,0], pred[:,1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
ax.set_title('Ground Truth Trajectory')
ax.set_xlim(min(pred[:, 0]), max(pred[:, 0]))
ax.set_ylim(min(pred[:, 1]), max(pred[:, 1]))
# ax.set_zlim(min(pred[:, 2]), max(pred[:, 2]))
plt.show()
plt.clf()

gt = np.loadtxt("/content/data/rgbd_dataset_freiburg2_pioneer_slam2/groundtruth.txt")

gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(gt[:,0], gt[:,1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth Trajectory')
ax.set_xlim(min(gt[:, 0]), max(gt[:, 0]))
ax.set_ylim(min(gt[:, 1]), max(gt[:, 1]))
ax.set_zlim(min(gt[:, 2]), max(gt[:, 2]))
plt.show()
plt.clf()

gt_fig = plt.figure()
ax = gt_fig.add_subplot(111)
ax.plot(gt[:,0], gt[:,1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
ax.set_title('Ground Truth Trajectory')
ax.set_xlim(min(gt[:, 0]), max(gt[:, 0]))
ax.set_ylim(min(gt[:, 1]), max(gt[:, 1]))
# ax.set_zlim(min(gt[:, 2]), max(gt[:, 2]))
plt.show()
plt.clf()

max(gt[:, 0]), max(gt[:, 1]), max(gt[:, 2])

max(pred[:, 0]), max(pred[:, 1]), max(pred[:, 2])

"""#Visulalise"""

import numpy as np
import matplotlib.pyplot as plt

lines = np.load("/content/data/all_losses_vptrnn3.npy", allow_pickle=True)

plt.plot(lines)
