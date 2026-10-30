import os, logging, sys

def _suppress_stderr_stdout():
    logging.disable(logging.WARNING)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 0=all, 1=INFO, 2=WARNING, 3=ERROR
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'  # optional
    os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'

def _restore():
    os.environ.pop('TF_CPP_MIN_LOG_LEVEL', None)
    os.environ.pop('XLA_FLAGS', None)
    os.environ.pop('TF_ENABLE_GPU_GARBAGE_COLLECTION', None)

_suppress_stderr_stdout()
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from hrnet_wrapper import HRNetBackboneLayer
import json, os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




#GPU SETUP
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
#/GPU SETUP

def swap_columns(arr: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Return a new array with columns i and j swapped.
    """
    out = arr.copy()
    out[:, [i, j]] = out[:, [j, i]]
    return out

def project_to_plane(arr: np.ndarray, plane: str) -> np.ndarray:
    """
    Project a (N,3) array onto one of the coordinate planes and return an (N,2) array.
    plane: one of 'xy', 'yz', 'xz' (case-insensitive)
    """
    p = plane.lower()
    if p == 'xy' or p == 'yx':
        return arr[:, [0, 1]].copy()
    if p == 'yz' or p == 'zy':
        return arr[:, [1, 2]].copy()
    if p == 'xz' or p == 'zx':
        return arr[:, [0, 2]].copy()
    raise ValueError("plane must be one of 'xy', 'yz', or 'xz'")


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


def load_or_create_features(feature_path, image_path, feature_extractor_path, verbose=False):
    # treat sentinel values that mean "don't save"
    if feature_path in (None, "", "-1"):
        feature_path = None
    
    # If a feature_path is provided, make it relative to cwd and normalize
    if feature_path is not None and feature_path[0] != '.':
        feature_path = '.' + feature_path
    '''
        # Treat provided path as relative to cwd (so absolute input still works)
        if not os.path.isabs(feature_path):
            if verbose:
                print("Path absolute")
            feature_path = os.path.abspath(os.path.join(os.getcwd(), feature_path))
        else:
            # keep absolute but normalize
            if verbose:
                print("Path is NOT absolute")
            feature_path = os.path.abspath(feature_path)'''

    if feature_path is not None and os.path.exists(feature_path):
        if verbose:
            print(f"Loading features from {feature_path}")
        data = np.load(feature_path)
        image = data['features']
    else:
        if verbose:
            print("Extracting features...")
        image = extract_features(image_path, feature_extractor_path)

        if feature_path is not None:
            # ensure path is normalized
            feature_path = os.path.normpath(feature_path)
            parent_dir = os.path.dirname(feature_path)

            # If parent_dir is empty or '.', do not create; treat as current dir
            if parent_dir and parent_dir != os.path.curdir:
                os.makedirs(parent_dir, exist_ok=True)

            np.savez_compressed(feature_path, features=image)
            if verbose:
                print(f"Saved features to {feature_path}")

            file_created = os.path.exists(feature_path)
            assert file_created, f"Feature file was not created at {feature_path}"
            if verbose:
                print("Feature file verified.")

    return image


def load_and_predict_direct3d(image_path, feature_extractor_path, detector_3d_path, feature_path="-1", verbose = False):
    image = load_or_create_features(feature_path, image_path, feature_extractor_path, verbose=verbose)
    '''
    if os.path.exists(feature_path):
        if verbose:
            print(f"Loading features from {feature_path}")
        data = np.load(feature_path)
        image = data['features']
    else:
        if verbose:
            print("Extracting features...")
        parent_dir = os.path.dirname(feature_path) or '.'
        os.makedirs(parent_dir, exist_ok=True)
        image = extract_features(image_path, feature_extractor_path)

        if feature_path != "-1":
            os.makedirs(os.path.dirname(feature_path) or '.', exist_ok=True)
            np.savez_compressed(feature_path, features=image)
            if verbose:
                print(f"Saved features to {feature_path}")
            
            file_created = os.path.exists(feature_path)
            assert file_created, f"Feature file was not created at {feature_path}"
            if verbose:
                print("Feature file verified.")'''

                
    #image = extract_features(image_path, feature_extractor_path)
    if verbose:
        print("testing")
    if verbose:
        for i in range(1, 10):
            print("=")
        print(image.shape)
    image = np.expand_dims(image, axis=0)
    if verbose:
        print(image.shape)
    
    detector_3d = load_model(detector_3d_path)
    joints_3d = detector_3d.predict(image)
    del detector_3d
    tf.keras.backend.clear_session()
    if verbose:
        print("Detection 3D Complete\n\n\n")

    #combined_input = image #np.concatenate((joints_2d, image), axis=-1)
    #joints_2d = np.expand_dims(joints_2d, axis=0)
    #joints_3d = detector_3d.predict((image, joints_2d[0]), batch_size=None)

    return joints_3d

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

def mpjpe_single(y_true, y_pred):
    #y_true = y_true.reshape(-1, int(y_true.shape[0] // num_coordinates), num_coordinates)
    #y_pred = y_pred.reshape(-1, int(y_pred.shape[0] // num_coordinates), num_coordinates)
    errors = np.linalg.norm(y_true - y_pred, axis=-1)
    return errors

def load_ground_truth(annotation_file, frame_to_load = 1, is_3d_load = False, coord_format = 'yxz'):
    with open(annotation_file) as f:
        data = json.load(f)
    frames = data['frames']
    
    frame = frames[frame_to_load-1]
    points = frame['annorect'][0]['annopoints']['point']
    print("loaded frame ", frame['image']['name'])
    joints = []
    for p in points:
        if is_3d_load:
            match coord_format:
                case "xyz":
                    joints.extend([p['x'], p['y'], p['z']])
                case "yxz":
                    joints.extend([p['y'], p['x'], p['z']])
                case "yzx":
                    joints.extend([p['y'], p['z'], p['x']])
                case "xzy":
                    joints.extend([p['x'], p['z'], p['y']])
                case "zyx":
                    joints.extend([p['z'], p['y'], p['x']])
                case "zxy":
                    joints.extend([p['z'], p['x'], p['y']])
                case _:
                    print("i dunno, wrong format, here have a yxz data")
                    joints.extend([p['y'], p['x'], p['z']])   
            #joints.extend([p['y'] / 512.0, p['x'] / 512.0, p['z']])
            #joints.extend([p['y'], p['x'], p['z']])
            #joints.extend([p['x'], p['y'], p['z']])
        else:
            joints.extend([p['y'] / 512.0, p['x'] / 512])# / 512.0, p['z']])#Modyfikacja zamieniono x i y joints.extend([p['x'] / 512.0, p['y'] / 512.0])
    joints = np.array(joints, dtype=np.float32)
    return joints

def load_new_json_datapoint(
        annotation_file,
        frame_to_load=1,
        coord_format='yxz',
        is_3d_load = False,
        verbose = False
):

    with open(annotation_file, 'r') as f:
        data = json.load(f)

    if frame_to_load >= len(data):
        print("JSON ERROR - out of bounds")
        return "-1", -1
    frame = data[frame_to_load]

    if verbose:
        print("loaded frame:", frame["image_path"])

    joints_3d = frame["joints_3d"]
    image_name = frame["image_path"]

    joints = []

    for p in joints_3d:

        x, y, z = p
        x = -x/1000
        y = -y/1000 
        z = -z/1000

        match coord_format:

            case "xyz":
                joints.extend([x, y, z])

            case "yxz":
                joints.extend([y, x, z])

            case "yzx":
                joints.extend([y, z, x])

            case "xzy":
                joints.extend([x, z, y])

            case "zyx":
                joints.extend([z, y, x])

            case "zxy":
                joints.extend([z, x, y])

            case _:
                print(
                    "Unknown coord_format, using yxz"
                )
                joints.extend([y, x, z])

    joints = np.array(
        joints,
        dtype=np.float32
    )

    return image_name, joints

def to_3D(u, v, depth, fx, fy, cx, cy):
    x = (u-cx)*depth/fx
    y = (v-cy)*depth/fy
    z = depth
    x = np.expand_dims(x, axis = -1)
    y = np.expand_dims(y, axis = -1)
    z = np.expand_dims(z, axis = -1)
    return np.concatenate((x,y,z), axis=-1)



def detect_and_combine():
    image_path = 'Data/images_nocircle/frame_1.png'
    anno_path = 'Data/annotations_nocircle.json'
    feature_extractor_path = 'hrnet_converted.keras'
    detector_2d_path = 'AI_ver_17_result/best_keypoint_model.keras'
    detector_3d_path = 'model/combined_model_volumetric6.keras'


    joint_positions_2d, joint_positions_3d = load_and_predict(image_path,
        feature_extractor_path, detector_2d_path, detector_3d_path)
    predicted_keypoints = joint_positions_2d.reshape(-1, 2) * 512

    true_keypoints = load_ground_truth(anno_path, 0, False) * 512
    formatted_true_keypoints = 512 - true_keypoints.reshape(-1, 2)
    formatted_true_keypoints[:, 1] = 512 -formatted_true_keypoints[:, 1]

    true_joints = load_ground_truth(anno_path, 0, True)
    true_joints = true_joints.reshape(-1, 3)

    print(f"Predicted 3D joint positions: {predicted_keypoints}")
    print(f"Predicted 3D joint positions: {joint_positions_3d}")

    joint_positions_3d = joint_positions_3d.reshape(-1, 1)
    combined = np.hstack((predicted_keypoints, joint_positions_3d))

    print(true_joints)
    print(combined)

    print(f"MPJPE result: {mpjpe(true_joints.reshape(-1, 3), combined, 3)}")
    return combined

def detect_direct_regression(frame_ix=2, verbose = False):
    image_path = f'Data/Full3D_data/frame_{frame_ix}.png'
    anno_path = 'Data/annotations_full3d.json'
    feature_extractor_path = 'hrnet_converted.keras'
    #detector_2d_path = 'AI_ver_17_result/best_keypoint_model.keras'
    detector_3d_path = 'model/regression3d_model_volumetric_2.keras'


    joint_positions_3d = load_and_predict_direct3d(image_path,
        feature_extractor_path, detector_3d_path)
    if verbose:
        print(f"testing shape {joint_positions_3d}")
    predicted_keypoints = joint_positions_3d.reshape(-1, 3)
    if verbose:
        print(f"testing shape {joint_positions_3d}")

    #true_keypoints = load_ground_truth(anno_path, 0, False)
    #formatted_true_keypoints = 512 - true_keypoints.reshape(-1, 2)
    #formatted_true_keypoints[:, 1] = 512 -formatted_true_keypoints[:, 1]
    if verbose:
        true_joints = load_ground_truth(anno_path, frame_ix, True, 'xyz')
        true_joints = true_joints.reshape(-1, 3)

        print(f"Predicted 3D joint positions: {predicted_keypoints}")
        #print(f"Predicted 3D joint positions: {joint_positions_3d}")
        print(f"True 3D joint positions: {true_joints}")

    #joint_positions_3d = joint_positions_3d.reshape(-1, 1)
    #combined = np.hstack((predicted_keypoints, joint_positions_3d))

    #print(true_joints)
    #print(combined)
    if verbose:
        print(f"MPJPE result: {mpjpe(true_joints.reshape(-1, 3), joint_positions_3d, 3)}")

    pred = predicted_keypoints.copy()#joint_positions_3d.reshape(-1, 3).copy()
    root_idx = 7
    root = pred[root_idx].copy()
    for i in range(pred.shape[0]):
        if i == root_idx:
            continue
            pred[i] += root
    if verbose:
        print(f"Absolute: {pred}")
    return pred.reshape(predicted_keypoints.shape)
    
    #return joint_positions_3d

def detect_direct_regression_for_test(image_path, feature_extractor_path, detector_3d_path, feature_path="-1", verbose = False):
    joint_positions_3d = load_and_predict_direct3d(image_path,
        feature_extractor_path, detector_3d_path, feature_path, verbose=verbose)
    if verbose:
        print(f"testing shape {joint_positions_3d}")
    predicted_keypoints = joint_positions_3d.reshape(-1, 3)
    if verbose:
        print(f"testing shape {joint_positions_3d}")


    pred = predicted_keypoints.copy()#joint_positions_3d.reshape(-1, 3).copy()
    root_idx = 7
    root = pred[root_idx].copy()
    for i in range(pred.shape[0]):
        if i == root_idx:
            continue
            pred[i] += root
    if verbose:
        print(f"Absolute: {pred}")
    return pred.reshape(predicted_keypoints.shape)

def test_3d():
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


def convert_corner_coords(arr, src_corner, dst_corner, image_shape, is_yx_format = False):
    """
    Convert x,y coordinates in arr (Nx3) that are based on distance from src_corner
    to coordinates based on distance from dst_corner. Third column preserved.
    Corners: 'tl' (top-left), 'tr' (top-right), 'bl' (bottom-left), 'br' (bottom-right).
    image_shape: (height, width)
    """
    if src_corner == dst_corner:
        return arr.copy()

    h, w = image_shape
    a = arr.copy().astype(float)
    x = a[:, 1] if is_yx_format else a[:, 0]
    y = a[:, 0] if is_yx_format else a[:, 1]
    # For corners:
    #  - 'tl': x_abs = x, y_abs = y
    #  - 'tr': x_abs = w - 1 - x
    #  - 'bl': x_abs = x, y_abs = h - 1 - y
    #  - 'br': x_abs = w - 1 - x, y_abs = h - 1 - y
    def to_abs(xi, yi, corner):
        if corner == 'tl':
            return xi, yi
        if corner == 'tr':
            return (w - 1 - xi), yi
        if corner == 'bl':
            return xi, (h - 1 - yi)
        if corner == 'br':
            return (w - 1 - xi), (h - 1 - yi)
        raise ValueError("invalid corner")

    # Convert absolute -> distance-from-corner for destination
    def from_abs(x_abs, y_abs, corner):
        if corner == 'tl':
            return x_abs, y_abs
        if corner == 'tr':
            return (w - 1 - x_abs), y_abs
        if corner == 'bl':
            return x_abs, (h - 1 - y_abs)
        if corner == 'br':
            return (w - 1 - x_abs), (h - 1 - y_abs)
        raise ValueError("invalid corner")

    x_abs, y_abs = to_abs(x, y, src_corner)
    new_x, new_y = from_abs(x_abs, y_abs, dst_corner)

    out = a.copy()
    if is_yx_format:
        out[:, 1] = new_x
        out[:, 0] = new_y
    else:
        out[:, 1] = new_y
        out[:, 0] = new_x
    return out

def convert_to_true_3d(arr, is_yx_format = False, debug_mode = False):
    focal = 20.78461#39.5
    center = 512/2

    #u = np.array([128,130,150,189,140])
    #v = np.array([128,150,128,189,140])
    #d = np.array([15,16,20,10,5])

    #u = np.array([128, 128])
    #v = np.array([128, 38])
    #d = np.array([15, 128])
    com = np.zeros((arr.shape[0], 3))
    for i, row in enumerate(arr):
        if is_yx_format:
            v, u, d = row
        else:
            u, v, d = row

        com[i,:] = to_3D(u, v, d, focal, focal, center, center)
    if debug_mode:
        print(com)
    return com

def direct_ai_output_coords_to_true_coords(data, anchor_point_ix = 6):
    anchor = data[anchor_point_ix]
    new_data = data.copy()
    for i, dat in enumerate(data):
        if i == anchor_point_ix:
            continue
        else:
            new_data[i] = dat + anchor
            #print(f"anchor {anchor}, data {dat}, new_data {new_data[i]}")
    return new_data

def true_coords_to_direct_ai_output_coords(data, anchor_point_ix = 6, zeroize_root = False):
    anchor = data[anchor_point_ix]
    new_data = data.copy()
    for i, dat in enumerate(data):
        if i == anchor_point_ix and zeroize_root == False:
            continue
        else:
            new_data[i] = dat - anchor
            #print(f"anchor {anchor}, data {dat}, new_data {new_data[i]}")
    return new_data


def direct_ai_output_coords_to_true_coords_another_anchor(data, data_for_new_anchor, anchor_point_ix = 6):
    anchor = data_for_new_anchor[anchor_point_ix]
    new_data = data.copy()
    for i, dat in enumerate(data):
        if i == anchor_point_ix:
            continue
        else:
            new_data[i] = dat + anchor
            #print(f"anchor {anchor}, data {dat}, new_data {new_data[i]}")
    return new_data


#####################
#   HUMAN3.6 TEST   #
#####################
import time
def test_on_benchmark_set(test_sample = 1, limit_frames = -1):
    image_directory = f"benchmark_dset/mpi_inf_3dhp_test_set/TS{test_sample}/imageSequence"
    anno_path = f"benchmark_dset/benchmark_dataset_univTS{test_sample}.json"
    feature_path = f"/benchmark_dset/TS{test_sample}_feat"
    feature_extractor_path = 'hrnet_converted.keras'
    detector_3d_path = 'model/regression3d_model_volumetric_2.keras'

    image_name = 1

    length = 0
    with open(anno_path, 'r') as f:
        data = json.load(f)
        length = len(data)


    
    pitch = 0
    yaw = -90
    roll = 0
    anchor_point = 6
    print('<', end = '')
    step = max(1, length // 20)
    next_threshold = step
    count_marks = 0
    start = time.perf_counter()
    mpjpe_all = []

    if limit_frames > 0:
        length = min(length, limit_frames) #DEBUG PURPOSES
        next_threshold = 0
        step = 1
    
    for i in range(length):
        image_name, true_joints =load_new_json_datapoint(anno_path,
            frame_to_load=i, coord_format='yxz')

        joint_positions_3d = detect_direct_regression_for_test(os.path.join(image_directory, image_name), feature_extractor_path, detector_3d_path,
                                                                feature_path+f"/{image_name}.npz", verbose=True) * 0.425
        #load_and_predict_direct3d(os.path.join(image_directory, image_name),feature_extractor_path, detector_3d_path)

        #joint_positions_3d[:, [0, 1]] = joint_positions_3d[:, [1, 0]]
        joint_positions_3d = swap_columns(joint_positions_3d, 0, 1)
        #swap_columns
        #predicted_keypoints = joint_positions_3d.reshape(-1, 3)
        #predicted_keypoints = predicted_keypoints #direct_ai_output_coords_to_true_coords( , anchor_point_ix = anchor_point
        true_joints = true_joints.reshape(-1, 3)
        #true_joints = direct_ai_output_coords_to_true_coords(true_joints, anchor_point_ix = anchor_point)
       
       
        _true_joints = true_joints.copy()#joint_positions_3d.reshape(-1, 3).copy()
        root_idx = 7
        root = _true_joints[root_idx].copy()
        for i in range(_true_joints.shape[0]):
            _true_joints[i] -= root

        _joint_positions_3d = joint_positions_3d.copy()#joint_positions_3d.reshape(-1, 3).copy()
        root_idx = 7
        root = _joint_positions_3d[root_idx].copy()
        for i in range(_joint_positions_3d.shape[0]):
            _joint_positions_3d[i] -= root


        tf.keras.backend.clear_session()
        
        mpjpe_all.append(mpjpe_single(_true_joints, _joint_positions_3d))

        if i + 1 >= next_threshold and count_marks < 20:
            if count_marks % 4 == 0:
                joint_positions_3d = direct_ai_output_coords_to_true_coords(joint_positions_3d, anchor_point_ix = anchor_point)
                #true_joints = true_coords_to_direct_ai_output_coords(true_joints, anchor_point_ix = anchor_point, zeroize_root = True)
                #joint_positions_3d[anchor_point] *= 0
                #joint_positions_3d = swap_columns(joint_positions_3d, 0, 2)
                
                display_skeleton_comparison(true_skeleton = true_joints, predicted_skeleton = joint_positions_3d, name=f"re_TS{test_sample}_{image_name}", scale=4.0,
                    pitch=pitch+(5*2)+0.001, yaw=yaw+(5*2)+0.001, roll=roll, anchor_point=-1, debug_mode=False, show_predicted_skeleton=True, point_size=50)
                
                display_skeleton_comparison(true_skeleton = true_joints, predicted_skeleton = joint_positions_3d, name=f"re_TS{test_sample}_{image_name}_front", scale=4.0,
                    pitch=pitch+(5*0)+0.001, yaw=yaw+(5*0)+0.001, roll=roll, anchor_point=-1, debug_mode=False, show_predicted_skeleton=True, point_size=50)
                
                display_skeleton_comparison(true_skeleton = true_joints, predicted_skeleton = true_joints, name=f"re_TS{test_sample}_{image_name}_true", scale=4.0,
                    pitch=pitch+(5*2)+0.001, yaw=yaw+(5*2)+0.001, roll=roll, anchor_point=-1, debug_mode=False, show_predicted_skeleton=True, point_size=50)
                display_skeleton_comparison(true_skeleton = joint_positions_3d, predicted_skeleton = joint_positions_3d, name=f"re_TS{test_sample}_{image_name}_pred", scale=4.0,
                    pitch=pitch+(5*2)+0.001, yaw=yaw+(5*2)+0.001, roll=roll, anchor_point=-1, debug_mode=False, show_predicted_skeleton=True, point_size=50)

                #swap_columns(arr: np.ndarray, i: int, j: int) -> np.ndarray:
                flat_points = project_to_plane(true_joints, plane = 'xy') * 512
                print(flat_points)
                display_2D_plot(f"TS{test_sample}_{image_name}_flat", flat_points, flat_points, os.path.join(image_directory, image_name))
                
            print('=', end='')
            count_marks += 1
            next_threshold += step
    while count_marks < 20:
        print('=', end='')
        count_marks += 1
    print('>', end='')
    elapsed = time.perf_counter() - start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"Elapsed: {minutes} min {seconds:.4f} sec")
    print(f"mean MPJPE result: {np.mean(mpjpe_all)}")
    return

from mpl_toolkits.mplot3d.art3d import Line3DCollection

def display_to_plot(com, name = "testing3d", scale = 10.0,
                    pitch = None, yaw = None, roll = None, anchor_point=-1, debug_mode = False):
    #N = com.shape[0]
    #colors = ['red' if i < 3 else 'green' if i < 5 else 'blue' for i in range(N)]
    if debug_mode:
        print(com.shape)
        print(com)
    N = com.shape[0]
    def color_for(i):
        if 0 <= i <= 2:
            return 'green'
        if 3 <= i <= 5:
            return 'darkred'
        if 6 <= i <= 9:
            return 'gold'
        if 10 <= i <= 12:
            return 'lime'
        if 13 <= i <= 15:
            return 'red'
        return 'gray'  # fallback for indices outside 0-15
    colors = [color_for(i) for i in range(N)]

    if anchor_point != -1:
        com = direct_ai_output_coords_to_true_coords(com, anchor_point_ix = anchor_point)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    centroid = com.mean(axis=0)
    half_scale = scale/2.0
    ax.set_xlim([centroid[0] - half_scale, centroid[0] + half_scale])
    ax.set_ylim([centroid[1] - half_scale, centroid[1] + half_scale])
    ax.set_zlim([centroid[2] - half_scale, centroid[2] + half_scale])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.scatter(com[:,0], com[:,1], com[:,2],
               #c = np.sqrt(com[0,:]**2 + com[1,:]**2 + com[2,:]**2),
               c = colors,
               s = 50)
               #cmap='viridis')
    #vertical_axis='z'
    ax.view_init(pitch,yaw,roll,vertical_axis='x')
    #ax.view_init(15,-90,0,vertical_axis='x')
    #ax.view_init(vertical_axis='x')
    #ax.view_init(90,90,vertical_axis='y')

    #EDGES
    edges = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(8,9),(10,11),(11,12),(13,14),(14,15),(2,6),(3,6),(12,8),(13,8)]
    segments = [ [com[i], com[j]] for i,j in edges ]
    lc = Line3DCollection(segments, colors=[color_for(j) for i,j in edges], linewidths=1)
    ax.add_collection3d(lc)
    
    for i in range(N):
        ax.text(com[i,0], com[i,1], com[i,2], f'{i}', color='k', fontsize=9,
                horizontalalignment='center', verticalalignment='center')

    ax.set_title(name)
    plt.show()
    plt.savefig(name + ".png")
    plt.close()

def display_2D_plot(plotname, keypoints_true, keypoints_pred, image_path):
    image_unprocessed = load_img(image_path, target_size=(512, 512))
    image   = img_to_array(image_unprocessed)[None] / 255.0
    image = image.squeeze(0)
    plt.figure(figsize=(6, 6))
    if image.max() <= 1.0:
        plt.imshow(image)
    else:
        plt.imshow(image.astype(np.uint8))
    
    plt.scatter(keypoints_true[:, 0], keypoints_true[:, 1],
        marker='+', color='blue', s=60, label='Ground Truth')
    plt.scatter(keypoints_pred[:, 0], keypoints_pred[:, 1],
        marker='x', color='green', s=60, label='Prediction')

    plt.title(f"{plotname}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plotname}.png")
    plt.close()


def display_skeleton_comparison(
        true_skeleton,
        predicted_skeleton,
        name="skeleton_comparison",
        scale=10.0,
        pitch=None,
        yaw=None,
        roll=None,
        anchor_point=-1,
        debug_mode=False,
        show_predicted_skeleton=True,
        point_size=50):

    if debug_mode:
        print("True shape:", true_skeleton.shape)
        print("Pred shape:", predicted_skeleton.shape)

    assert true_skeleton.shape == predicted_skeleton.shape, \
        "True and predicted skeletons must have the same shape"

    N = true_skeleton.shape[0]

    # Same coloring as your original function
    def color_for(i):
        if 0 <= i <= 2:
            return 'green'
        if 3 <= i <= 5:
            return 'darkred'
        if 6 <= i <= 9:
            return 'gold'
        if 10 <= i <= 12:
            return 'lime'
        if 13 <= i <= 15:
            return 'red'
        return 'gray'

    colors = [color_for(i) for i in range(N)]

    # Optional anchoring
    if anchor_point != -1:
        true_skeleton = direct_ai_output_coords_to_true_coords(
            true_skeleton,
            anchor_point_ix=anchor_point
        )

        predicted_skeleton = direct_ai_output_coords_to_true_coords(
            predicted_skeleton,
            anchor_point_ix=anchor_point
        )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Camera centering based on true skeleton
    centroid = true_skeleton.mean(axis=0)
    half_scale = scale / 2.0

    ax.set_xlim([centroid[0] - half_scale, centroid[0] + half_scale])
    ax.set_ylim([centroid[1] - half_scale, centroid[1] + half_scale])
    ax.set_zlim([centroid[2] - half_scale, centroid[2] + half_scale])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # ==========================
    # TRUE SKELETON
    # ==========================
    ax.scatter(
        true_skeleton[:, 0],
        true_skeleton[:, 1],
        true_skeleton[:, 2],
        c=colors,
        s=point_size,
        label="True"
    )

    # ==========================
    # PREDICTED JOINTS
    # ==========================
    ax.scatter(
        predicted_skeleton[:, 0],
        predicted_skeleton[:, 1],
        predicted_skeleton[:, 2],
        c='cyan',
        s=point_size * 0.8,
        marker='x',
        label="Predicted"
    )

    # Skeleton edges
    edges = [
        (0,1),(1,2),
        (3,4),(4,5),
        (6,7),(7,8),(8,9),
        (10,11),(11,12),
        (13,14),(14,15),
        (2,6),(3,6),
        (12,8),(13,8)
    ]

    # ==========================
    # TRUE SKELETON BONES
    # ==========================
    true_segments = [
        [true_skeleton[i], true_skeleton[j]]
        for i, j in edges
    ]

    true_lc = Line3DCollection(
        true_segments,
        colors=[color_for(j) for i, j in edges],
        linewidths=2
    )

    ax.add_collection3d(true_lc)

    # ==========================
    # OPTIONAL PREDICTED SKELETON
    # ==========================
    if show_predicted_skeleton:
        pred_segments = [
            [predicted_skeleton[i], predicted_skeleton[j]]
            for i, j in edges
        ]

        pred_lc = Line3DCollection(
            pred_segments,
            colors='cyan',
            linewidths=1,
            linestyles='dashed',
            alpha=0.7
        )

        ax.add_collection3d(pred_lc)

    # ==========================
    # ERROR LINES
    # ==========================
    error_segments = [
        [true_skeleton[i], predicted_skeleton[i]]
        for i in range(N)
    ]

    error_lc = Line3DCollection(
        error_segments,
        colors='black',
        linewidths=1,
        alpha=0.5
    )

    ax.add_collection3d(error_lc)

    # ==========================
    # JOINT LABELS
    # ==========================
    for i in range(N):

        # True joint label
        ax.text(
            true_skeleton[i, 0],
            true_skeleton[i, 1],
            true_skeleton[i, 2],
            f'{i}',
            color='black',
            fontsize=9,
            horizontalalignment='center'
        )

        # Optional predicted label
        ax.text(
            predicted_skeleton[i, 0],
            predicted_skeleton[i, 1],
            predicted_skeleton[i, 2],
            f'{i}',
            color='blue',
            fontsize=7,
            alpha=0.8
        )

    ax.view_init(
        elev=pitch,
        azim=yaw,
        roll=roll,
        vertical_axis='x'
    )
    ax.set_title(name)
    plt.show()
    plt.savefig(name + ".png")
    plt.close()

'''
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
test_arr = np.array([
[293.03895569, 266.46508789,   9.97382355],
[241.7434082 , 259.55020142,   9.84209538],
[188.60046387, 264.52557373,  10.04490566],
[189.31030273, 287.58145142,  10.06592846],
[234.19488525, 294.69448853,   9.77544594],
[289.23504639, 296.83843994,   9.75327873],
[194.27105713, 278.57214355,  10.06028748],
[178.21130371, 280.12139893,  10.08483696],
[116.6567688 , 282.8885498 ,  10.38862133],
[ 93.3164978 , 283.23788452,  10.57034969],
[161.40274048, 270.48968506,  10.15598583],
[161.93521118, 258.03323364,  10.31442738],
[133.35083008, 262.62145996,  10.36061573],
[126.81811523, 296.49975586,  10.3035717 ],
[154.63867188, 304.46405029,  10.10526943],
[176.15643311, 301.39770508,   9.93408108]
])
test_arr2 = np.array([
[ 0.8597001,  -0.5493281,  -0.25721264],
[ 0.34177947, -0.0829066,  -1.0172664 ],
[-0.18973212, -0.04759353, -0.10468938],
[-0.03270467,  0.15971129,  0.1450377 ],
[ 0.97303474,  0.20721246,  0.11278892],
[ 1.8308115,  -0.2162106,   0.17928018],
[-0.16617307, -1.3158971,   7.1184444 ],
[-0.37995455,  0.161685,   -0.06137639],
[-1.3807323,   0.31435743,  0.34137762],
[-2.018625,    0.5360353,   0.09622455],
[-0.2345607,   0.26273713, -0.44347504],
[-0.8778437,   0.12325543, -0.16333443],
[-1.3800737,   0.07813557, -0.01506232],
[-1.0914166,   0.4027243,   0.56783146],
[-0.9065881,   0.5585319,   1.1610668 ],
[-0.547328,    0.5512057,   0.899191  ],
])

test_arr3 = np.array([
 [-0.63061357,  0.8346565,  -0.19332981],
 [-0.18132162,  0.40441233, -1.0018253 ],
 [-0.10870671, -0.14549069, -0.15169239],
 [ 0.15803695, -0.06015277,  0.16553402],
 [ 0.3886652,   0.920155,    0.03826523],
 [-0.19823813,  1.73262,     0.22820663],
 [-1.3262203,  -0.1850786,   6.9400425 ],
 [ 0.13701856, -0.3698513,  -0.01571846],
 [ 0.38277614, -1.4788293,   0.2752571 ],
 [ 0.5699849,  -1.9747231,   0.08108521],
 [ 0.10501814, -0.37761986, -0.44501305],
 [ 0.10538697, -0.82102513, -0.30082512],
 [ 0.06006718, -1.3670331,  -0.12954617],
 [ 0.50065136, -1.1226075,   0.5409517 ],
 [ 0.62862766, -0.8712623,   1.0409145 ],
 [ 0.6605048,  -0.47351897,  0.7997055 ]
 ])

predicted3d = np.array([[-3.5972486e-03, -1.7379725e+00,  7.2474140e-01],
[ 1.6301420e-01, -9.2815709e-01,  2.0243551e-01],
[-1.5813343e-02, -1.7721146e-02, -2.2966246e-01],
[-4.2095068e-03,  2.2736280e-01,  1.4965446e-01],
[ 3.5926536e-02, -7.4071497e-01,  7.0317793e-01],
[-1.7149267e-01, -1.4496650e+00,  1.2071590e+00],
[-1.0960883e-02, -7.1355872e-02,  5.8312693e+00],
[ 1.2380474e-01,  3.3374324e-01, -1.2088537e-01],
[ 8.7603182e-02,  1.4177555e+00, -7.9862744e-01],
[ 3.5983527e-01,  1.8697977e+00, -1.0821810e+00],
[ 2.4281363e-01,  4.4331661e-01, -7.7423173e-01],
[-2.3586748e-02,  5.1154578e-01, -1.0233692e+00],
[ 1.1616860e-02,  1.0466808e+00, -1.0391575e+00],
[ 2.1741907e-01,  1.4090731e+00, -2.7359831e-01],
[ 3.6213434e-01,  9.7370338e-01, -1.9324429e-02],
[ 6.9049519e-01,  6.6483706e-01, -3.1782400e-02]])

true3d = np.array([[-1.7707287e+00,  1.0671807e-01,  7.4598122e-01],
[-8.8296282e-01,  2.2963326e-01,  2.5874519e-01],
[-1.3018787e-02,  4.7283292e-02, -2.3149586e-01],
[ 1.9578564e-01, -4.7283322e-02,  1.2420845e-01],
[-6.5057355e-01, -5.0190900e-02,  6.8461800e-01],
[-1.5360628e+00, -2.5973663e-01,  1.1457386e+00],
[-6.7503333e-02,  3.8251214e-04,  5.7542157e+00],
[ 3.4318793e-01,  6.0934693e-02, -1.8525696e-01],
[ 1.3552450e+00,  6.8441555e-02, -7.7735615e-01],
[ 1.8405350e+00,  2.2208153e-01, -1.0213833e+00],
[ 5.7241285e-01,  2.7309453e-01, -6.8895626e-01],
[ 4.0224075e-01, -9.2871055e-02, -9.2242813e-01],
[ 9.3157566e-01,  1.1372528e-01, -1.0038977e+00],
[ 1.3783685e+00,  7.3122650e-02, -2.2972012e-01],
[ 9.5136154e-01,  4.2322788e-01, -7.2854996e-02],
[ 6.6040850e-01,  7.8582495e-01, -1.0846233e-01]])

#swapped = true3d[:, [1, 0, 2]]
#print(true3d)
#true3d[:, [0, 1]] = true3d[:, [1, 0]]
predicted3d[:, [0, 1]] = predicted3d[:, [1, 0]]
#print(true3d)

#display_to_plot(true3d, name = f"live3d_ver_5_true",
#            scale = 4.0, anchor_point=6)
#display_to_plot(predicted3d, name = f"live3d_ver_5_pred",
#            scale = 4.0, anchor_point=6)

#convert_to_true_3d(test_arr, True)
#comb = detect_and_combine()
#print('\n\n\n\n')
#print(convert_corner_coords(comb, 'bl', 'tl', (512, 512), True))
#display_to_plot(detect_direct_regression(), name = "live3d_ver_2")

#display_to_plot(test_arr2, scale = 10.0, pitch = 0.0, yaw = 0.0, roll = 0.0))
#test_3d()

#ai_data = convert_to_true_3d(detect_and_combine())#, name="live3d", scale = 4.0)

################################

#detect_direct_regression(2)

#TEST PARAMETERS
run_permutation_tests = False
test_type="no" #"2d"/"3d"/"no"/"3d_raw"/"3d_my_vs_other"


match test_type:
    case "2d":
        test_data = convert_to_true_3d(test_arr, True)
    case "3d":
        test_data = true3d#predicted3d
        test_data2 = predicted3d 
    case "no":
        print("no_test")
    case "3d_raw":
        print("3d_raw")
    case "3d_my_vs_other":
        print("3d_my_vs_other")
    case _:
        print("no_test")
        test_type="no"


if run_permutation_tests:
    from itertools import permutations
    a = [45, -150, 115]
    labels = "xyz"
    for p in permutations(range(len(a))):
        val = ''.join(labels[i] for i in p), [a[i] for i in p] #('xyz', [45, -150, 115])
        #print(type(val), val, val[0], val[1][1])
        if val[0] == "yxz":
            print("here", val[1])
        display_to_plot(test_data, name = "live3d_ver_3_"+val[0],
                        scale = 80.0, pitch = val[1][0], yaw = val[1][1], roll = val[1][2], anchor_point=-1)


################################################
test_on_benchmark_set(test_sample = 1, limit_frames = 5)
_restore()
################################################


#yxz
#[-150, 45, 115]
if test_type!="no":
    ap = -1
    scale = 80.0
    pitch = -90
    yaw = -90
    roll = -90
    
    pitch = 0
    yaw = -90
    roll = 0
    
    #pitch = None
    #yaw = None
    #roll = None
    if test_type=="3d":
        ap=6
        scale = 4.0
    
    #compare_2_skeletons(true_points = test_data, predicted_points = test_data2, name = "true_vs_predicted", scale = 10.0,
    #                pitch = pitch, yaw = yaw, roll = roll, anchor_point=ap, debug_mode = False)
    if test_type=="no":
        ap = 6
        anno_path = 'Data/annotations_full3d.json'
        test_x = detect_direct_regression(frame_ix=2)
        test_x[:, [0, 1]] = test_x[:, [1, 0]]
        test_y = load_ground_truth(anno_path, frame_to_load = 2, is_3d_load = True, coord_format = 'yxz')
        display_skeleton_comparison(true_skeleton = test_x, predicted_skeleton = test_y.reshape(-1, 3), name=f"test_raw", scale=6.0,
                        pitch=pitch+0.001, yaw=yaw+0.001, roll=roll, anchor_point=ap, debug_mode=False, show_predicted_skeleton=True, point_size=50)
    elif test_type=="3d_my_vs_other":
        ap = 6
        image_directory = "benchmark_dset/mpi_inf_3dhp_test_set/TS1/imageSequence"
        image_name = 'img_005603.jpg' #'img_000001.jpg'
        feature_path = f"/benchmark_dset/TS1_feat"
        feature_extractor_path = 'hrnet_converted.keras'
        detector_3d_path = 'model/regression3d_model_volumetric_2.keras'

        test_x = detect_direct_regression_for_test(os.path.join(image_directory, image_name), feature_extractor_path, detector_3d_path,
                                                                feature_path+f"/{image_name}.npz", verbose=True)
        image_path = "Data/Full3D_data/frame_19.png"
        feature_path = f"/Data/Full3D_againFeatures/frame_19.png.npz"
        test_y = detect_direct_regression_for_test(image_path, feature_extractor_path, detector_3d_path, feature_path, verbose=True)

        test_x = swap_columns(test_x, 0, 1)
        test_y = swap_columns(test_y, 0, 1)

        display_skeleton_comparison(true_skeleton = test_x, predicted_skeleton = test_y, name=f"porownanie datasetow1", scale=4.0,
                        pitch=pitch+(5*0)+0.001, yaw=yaw+(5*0)+0.001, roll=roll, anchor_point=ap, debug_mode=False, show_predicted_skeleton=True, point_size=50)
        display_skeleton_comparison(true_skeleton = test_y, predicted_skeleton = test_x, name=f"porownanie datasetow2", scale=4.0,
                        pitch=pitch+(5*0)+0.001, yaw=yaw+(5*0)+0.001, roll=roll, anchor_point=ap, debug_mode=False, show_predicted_skeleton=True, point_size=50)
    else:
        for i in range(5):
            display_to_plot(test_data, name = f"live3d_ver_4{i}",
                scale = scale, pitch = pitch+(5*i)+0.001, yaw = yaw-(5*i)-0.001, roll = roll, anchor_point=ap)
            if test_type=="3d":
                display_to_plot(test_data2, name = f"live3d_ver_pred4{i}",
                    scale = scale, pitch = pitch+5*i, yaw = yaw-5*i, roll = roll, anchor_point=ap)
                display_skeleton_comparison(true_skeleton = test_data, predicted_skeleton = test_data2, name=f"live3d_skeleton_compare{i}", scale=4.0,
                        pitch=pitch+(5*i)+0.001, yaw=yaw+(5*i)+0.001, roll=roll, anchor_point=ap, debug_mode=False, show_predicted_skeleton=False, point_size=50)
    #display_to_plot(convert_to_true_3d(detect_and_combine())), name="live3d", scale = 4.0)




def restart_py():
    import IPython
    IPython.Application.instance().kernel.do_shutdown(restart=True)
