#COMBINED MODEL 1
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from hrnet_wrapper import HRNetBackboneLayer
import json, os


#GPU SETUP
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
#/GPU SETUP


def extract_features(fname, hrnet_model_path):
    assert(fname.lower().endswith((".jpg", ".png")))

    img = load_img(fname, target_size=(512, 512))
    x   = img_to_array(img)[None] / 255.0  # shape (1,512,512,3)

    hrnet = load_model(hrnet_model_path,
                   compile=False,
                   custom_objects={"HRNetBackboneLayer": HRNetBackboneLayer})

    feat = hrnet.predict(x, verbose=0)
    feat = feat[0]# batch-dim
    feat_ds = tf.keras.layers.AveragePooling2D(pool_size=2)(feat[None])[0]  

    del hrnet
    tf.keras.backend.clear_session()
    feat16 = feat_ds.numpy().astype('float16')
    return feat16


def load_and_predict(image_path, feature_extractor_path, detector_2d_path, detector_3d_path):
    
    image = extract_features(image_path, feature_extractor_path)
    print("testing")
    for i in range(1, 10):
        print("=")
    print(image.shape)
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    detector_2d = load_model(detector_2d_path)
    joints_2d = detector_2d.predict(image)
    
    del detector_2d
    tf.keras.backend.clear_session()
    print("Detection 2D Complete\n\n\n")

    detector_3d = load_model(detector_3d_path)
    print(image.shape)
    print(joints_2d.shape)
    #combined_input = image #np.concatenate((joints_2d, image), axis=-1)
    joints_2d = np.expand_dims(joints_2d, axis=0)
    joints_3d = detector_3d.predict((image, joints_2d[0]), batch_size=None)

    return joints_2d, joints_3d


def mpjpe(y_true, y_pred, num_coordinates = 2):
    """
    Computes the Mean Per Joint Position Error (MPJPE)
    between predicted and ground truth keypoints.
    """
    y_true = y_true.reshape(-1, int(y_true.shape[1] // num_coordinates), num_coordinates)
    y_pred = y_pred.reshape(-1, int(y_pred.shape[1] // num_coordinates), num_coordinates)
    # Compute Euclidean distances per joint
    errors = np.linalg.norm(y_true - y_pred, axis=-1)

    # Average over joints and samples
    mpjpe_value = np.mean(errors)

    return mpjpe_value

def load_ground_truth(annotation_file, frame_to_load = 0, is_3d_load = False):
    with open(annotation_file) as f:
        data = json.load(f)
    frames = data['frames']
    
    frame = frames[0]
    points = frame['annorect'][0]['annopoints']['point']
    joints = []
    for p in points:
        if is_3d_load:
            #joints.extend([p['y'] / 512.0, p['x'] / 512.0, p['z']])
            joints.extend([p['y'], p['x'], p['z']])
        else:
            joints.extend([p['y'] / 512.0, p['x'] / 512.0])# / 512.0, p['z']])#Modyfikacja zamieniono x i y joints.extend([p['x'] / 512.0, p['y'] / 512.0])
    joints = np.array(joints, dtype=np.float32)
    return joints


def to_3D(u, v, depth, fx, fy, cx, cy):
    x = (u-cx)*depth/fx
    y = (v-cy)*depth/fy
    z = depth
    x = np.expand_dims(x, axis = -1)
    y = np.expand_dims(y, axis = -1)
    z = np.expand_dims(z, axis = -1)
    return np.concatenate((x,y,z), axis=-1)



image_path = 'Data/images_nocircle/frame_15.png'
anno_path = 'Data/annotations_nocircle.json'
feature_extractor_path = 'hrnet_converted.keras'
detector_2d_path = 'AI_ver_17_result/best_keypoint_model.keras'
detector_3d_path = 'model/combined_model_volumetric6.keras'


joint_positions_2d, joint_positions_3d = load_and_predict(image_path,
    feature_extractor_path, detector_2d_path, detector_3d_path)
predicted_keypoints = joint_positions_2d.reshape(-1, 2) * 512

true_keypoints = load_ground_truth(anno_path, 16, False) * 512
formatted_true_keypoints = 512 - true_keypoints.reshape(-1, 2)
formatted_true_keypoints[:, 1] = 512 -formatted_true_keypoints[:, 1]

true_joints = load_ground_truth(anno_path, 0, False)
true_joints = true_joints.reshape(-1, 2)

print(f"Predicted 2D joint positions: {predicted_keypoints}")
print(f"Predicted 3D joint positions: {joint_positions_3d}")

joint_positions_3d = joint_positions_3d.reshape(-1, 1)
combined = np.hstack((predicted_keypoints, joint_positions_3d))

print(true_joints)
print(combined)

print(f"MPJPE result: {mpjpe(true_joints * 512, predicted_keypoints, 2)}") #14.916847229003906

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

predicted_keypoints = predicted_keypoints[:, ::-1]
true_joints = true_joints[:, ::-1]
h = 512
w = 512
pred = predicted_keypoints.copy()
pred[:, 1] = h - pred[:, 1]

true = (true_joints * 512).copy()
true[:, 1] = h - true[:, 1]

img = mpimg.imread(image_path)
plt.imshow(img)
plt.scatter(*pred.T, c="red", label="Predicted")
plt.scatter(*(true).T, c="lime", marker="x", label="True")
plt.legend()
plt.axis("off")
plt.savefig("roz6_0_result.pdf")
'''

focal = 39.5/10
center = 512/4

u = np.array([128,130,150,189,140])
v = np.array([128,150,128,189,140])
d = np.array([15,16,20,10,5])

#u = np.array([128, 128])
#v = np.array([128, 38])
#d = np.array([15, 128])

com = np.zeros((len(u), 3))

for i, ele in enumerate(u):
    com[i,:] = to_3D(ele, v[i], d[i], focal, focal, center, center)

print(com)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = com.shape[0]
colors = ['red' if i < 3 else 'green' if i < 5 else 'blue' for i in range(N)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(com[:,0], com[:,1], com[:,2],
           #c = np.sqrt(com[0,:]**2 + com[1,:]**2 + com[2,:]**2),
           c = colors,
           s = 50)
           #cmap='viridis')

for i in range(N):
    ax.text(com[i,0], com[i,1], com[i,2], f'{i+1}', color='k', fontsize=9,
            horizontalalignment='center', verticalalignment='center')

ax.set_title("testowe3d")
plt.show()
plt.savefig("testowe3d.png")
plt.close()

imgage = load_img(image_path, target_size=(512, 512))
img = (img_to_array(imgage)[None] / 255.0)[0]
keypoints_pred_flipped = 512 - predicted_keypoints.copy()
print(type(keypoints_pred_flipped))
keypoints_pred_flipped[:, 1] = 512 -keypoints_pred_flipped[:, 1]
plt.figure(figsize=(6, 6))
if img.max() <= 1.0:
    plt.imshow(img)
else:
    plt.imshow(img.astype(np.uint8))
plt.scatter(keypoints_pred_flipped[:, 1], keypoints_pred_flipped[:, 0],
    marker='x', color='green', s=60, label='Prediction')
plt.scatter(formatted_true_keypoints[:, 1], formatted_true_keypoints[:, 0],
    marker='+', color='blue', s=60, label='Truth')
#plt.show()
plt.savefig("combined.png")
plt.close()
'''
