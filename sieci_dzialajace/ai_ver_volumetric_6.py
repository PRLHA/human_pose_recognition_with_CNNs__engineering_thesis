import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import Conv2D, Reshape
from CustomLayer_v3 import SpatialSoftmax, SoftArgMaxConv
from tensorflow.keras.utils import Sequence

import json, os

import matplotlib.pyplot as plt

class data_generator(Sequence):
    def __init__(self, json_path, img_dir, vec_dir, batch_size=16):
        self.img_dir = img_dir
        self.vec_dir = vec_dir
        self.batch_size = batch_size
        with open(json_path, "r") as f:
            metadata = json.load(f)
        with open(json_path) as f:
            data = json.load(f)
            
        self.keys = data['frames']

    def __len__(self):
        #return len(self.keys)
        return int(np.ceil(len(self.keys) / self.batch_size))

    def __getitem__(self, idx):
        #while True:  # infinite loop for Keras
        #np.random.shuffle(keys)
        i = idx * self.batch_size
        batch_keys = self.keys[i:i+self.batch_size]

        images = []
        vectors = []
        zs = []

        for k in batch_keys:
            # Load image features
            img_path = os.path.join(self.img_dir, f"{k['image']['name']}.npz")
            img_data = np.load(img_path)
            image = img_data["features"]  # replace with actual key in .npz
            images.append(image)

                # Load vector + Z
            vec_path = os.path.join(self.vec_dir, f"{k['image']['name']}.npz")
            vec_data = np.load(vec_path)
            vector = vec_data["features"].flatten()  # shape (32,)

            points = k['annorect'][0]['annopoints']['point']
            joints = []
            for p in points:
                joints.append([p['z']])#Modyfikacja zamieniono x i y joints.extend([p['x'] / 512.0, p['y'] / 512.0])
            joints = np.array(joints, dtype=np.float32)
            z = joints#["z"]  # shape (16,)
                
            vectors.append(vector)
            zs.append(z)
        #print(shape((np.array(images), np.array(vectors))))
        return (np.array(images), np.array(vectors)), np.array(zs)

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')

# Configuration parameters
datapath        = './Data'
# Use only reduced features (must exist)
feature_dir     = os.path.join(datapath, 'features_nocircle')
vect_dir        = os.path.join(datapath, 'vector_predictions')
val_feature_dir     = os.path.join(datapath, 'features_validation')
val_vect_dir        = os.path.join(datapath, 'vector_validation')
if not os.path.exists(feature_dir):
    raise FileNotFoundError(f"Reduced feature directory not found: {feature_dir}. Run extraction with reduction enabled.")
print(f"Using reduced feature directory: {feature_dir}")


if os.path.exists(feature_dir):
    feature_dir = feature_dir
    print(f"Using reduced feature directory: {feature_dir}")
elif os.path.exists(feature_dir):
    feature_dir = feature_dir
    print(f"Using raw feature directory: {feature_dir}")
else:
    raise FileNotFoundError(
        f"No feature directory found. Checked {feature_dir_reduced} and {feature_dir_raw}."
    )

annotation_file = os.path.join(datapath, 'annotations_nocircle.json')
val_annotation_file = os.path.join(datapath, 'annotations_validation.json')
model_path      = './model/combined_model_volumetric6.keras'
general_path = "./"

BATCH_SIZE = 4
NUM_KPTS   = 16
EPOCHS     = 100

# Load and split annotations
#with open(annotation_file) as f:
#    data = json.load(f)
#frames = data['frames']

# =============== GENERATORS ===============
train_gen = data_generator(annotation_file, feature_dir, vect_dir, batch_size=BATCH_SIZE)
val_gen = data_generator(val_annotation_file, val_feature_dir, val_vect_dir, batch_size=BATCH_SIZE)

#np.random.seed(42)
#indices = np.arange(len(frames))
#np.random.shuffle(indices)
#split = int(0.8 * len(indices))
#train_frames = [frames[i] for i in indices[:split]]
#val_frames   = [frames[i] for i in indices[split:]]



#train_gen = FeatureDataGenerator(train_frames, feature_dir, batch_size=BATCH_SIZE, num_keypoints=NUM_KPTS)
#val_gen   = FeatureDataGenerator(val_frames,   feature_dir, batch_size=BATCH_SIZE, num_keypoints=NUM_KPTS)
#train_gen = data_generator(annotation_file, feature_dir, vect_dir, batch_size=BATCH_SIZE)
#val_gen = data_generator(annotation_file, feature_dir, vect_dir, batch_size=BATCH_SIZE)

# Infer feature map shape
def infer_shape(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.npy') or f.endswith('.npz')]
    if not files:
        raise RuntimeError(f"No feature files found in {directory}.")
    arr = np.load(os.path.join(directory, files[0]))
    feat = arr['features'] if isinstance(arr, np.lib.npyio.NpzFile) else arr
    return feat.shape

H, W, C = infer_shape(feature_dir)
print(f"Feature map shape inferred as: {H}×{W}×{C}")

# Build head-only model
def build_head_model_old(input_shape, num_keypoints):
    inp = layers.Input(shape=input_shape, name='feature_input')
    x = inp
    # Two residual blocks
    for _ in range(2):
        shortcut = x
        if shortcut.shape[-1] != 256:
            shortcut = Conv2D(256, 1, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = Conv2D(256, 3, padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
    # Heatmap prediction
    x = Conv2D(num_keypoints, 1, activation=None, name='heatmap_conv')(x)
    x = layers.ReLU()(x)
    x = SpatialSoftmax(name='spatial_softmax')(x)
    #x = layers.Dropout(0.3)(x)
    kp = SoftArgMaxConv(name='softargmax')(x)
    out = Reshape((num_keypoints * 2,), name='reshape_output')(kp)
    return models.Model(inputs=inp, outputs=out, name='keypoint_head')

def build_combined_model(image_input_shape=(256, 256, 151), vector_input_shape=(32,), num_keypoints=16):
    # Image input branch
    image_input = Input(shape=image_input_shape)
    x = layers.Conv2D(151, (3, 3), activation='relu', padding='same')(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(160, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten the convolutional output
    x = layers.Flatten()(x)
    
    # Vector input branch
    vector_input = Input(shape=vector_input_shape)
    
    # Combine the outputs
    combined = layers.concatenate([x, vector_input])
    
    # Dense layers after concatenation
    combined = layers.Dense(512, activation='relu')(combined)

    combined = layers.Dense(512, activation='relu')(combined)
    
    # Output layer for 3D coordinate (z) for each keypoint
    output = layers.Dense(num_keypoints)(combined) #Just Z
    
    # Create the model
    model = models.Model(inputs=[image_input, vector_input], outputs=output)
    
    return model

model = build_combined_model((H, W, C), (NUM_KPTS*2,), NUM_KPTS)
model.summary()


from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
# Learning rate schedule
lr_schedule = CosineDecayRestarts(
    initial_learning_rate=1e-4,
    first_decay_steps=30,
    t_mul=2.0,
    m_mul=1.0,
    alpha=1e-6
)
# Model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='best_keypoint_model.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
# Early stopping callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# Compile & train
model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mean_squared_error')
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[checkpoint_callback, early_stop]
)

np.savez(general_path + "training_history_combined_model.npz",
         loss=history.history['loss'],
         val_loss=history.history['val_loss'])

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Learning Curve', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(general_path + "ai_ver_volumetric_6_learning_curve.png")
plt.close()

# Save the trained head model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)
print(f"Head model training complete and saved at {model_path}")
