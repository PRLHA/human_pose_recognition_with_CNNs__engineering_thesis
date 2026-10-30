import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Reshape
from CustomLayer_v3 import SpatialSoftmax, SoftArgMaxConv
from feature_generator import FeatureDataGenerator
import json, os

import matplotlib.pyplot as plt

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')

# Configuration parameters
datapath        = './Data'
# Use only reduced features (must exist)
feature_dir     = os.path.join(datapath, 'features_hrnet')
if not os.path.exists(feature_dir):
    raise FileNotFoundError(f"Reduced feature directory not found: {feature_dir}. Run extraction with reduction enabled.")
print(f"Using reduced feature directory: {feature_dir}")

annotation_file = os.path.join(datapath, 'annotations.json')
model_path      = './model/keypoint_head.keras'

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

annotation_file = os.path.join(datapath, 'annotations.json')
model_path      = './model/keypoint_head.keras'
general_path = "./"

BATCH_SIZE = 4
NUM_KPTS   = 16
EPOCHS     = 100

# Load and split annotations
with open(annotation_file) as f:
    data = json.load(f)
frames = data['frames']

np.random.seed(42)
indices = np.arange(len(frames))
np.random.shuffle(indices)
split = int(0.8 * len(indices))
train_frames = [frames[i] for i in indices[:split]]
val_frames   = [frames[i] for i in indices[split:]]

# Generators
train_gen = FeatureDataGenerator(train_frames, feature_dir, batch_size=BATCH_SIZE, num_keypoints=NUM_KPTS)
val_gen   = FeatureDataGenerator(val_frames,   feature_dir, batch_size=BATCH_SIZE, num_keypoints=NUM_KPTS)

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
    #x = Conv2D(num_keypoints, 1, activation=None, name='heatmap_conv')(x)
    #x = layers.ReLU()(x)
    #x = SpatialSoftmax(name='spatial_softmax')(x)
    #x = layers.Dropout(0.3)(x)
    kp = SoftArgMaxConv(name='softargmax')(x)
    out = Reshape((num_keypoints * 2,), name='reshape_output')(kp)
    return models.Model(inputs=inp, outputs=out, name='keypoint_head')

def build_head_model(input_shape=(224, 224, 3), num_keypoints=16):
    model = models.Sequential()
    
    # Convolutional layers
    model.add(layers.Conv2D(151, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(160, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(num_keypoints, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(SoftArgMaxConv())
    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_keypoints * 2))  # 2 coordinates (x, y) for each keypoint
    
    return model

model = build_head_model((H, W, C), NUM_KPTS)
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

np.savez(general_path + "training_history.npz",
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
plt.savefig(general_path + "learning_curve.png")
plt.close()

# Save the trained head model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)
print(f"Head model training complete and saved at {model_path}")
