from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import json

class KeypointDataGenerator(Sequence):
    def __init__(self, frames, image_dir, batch_size=8, input_size=(512, 512), num_keypoints=16, shuffle=True):
        self.frames = frames
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_keypoints = num_keypoints
        self.shuffle = shuffle
        self.indices = np.arange(len(self.frames))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.frames) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_frames = [self.frames[k] for k in batch_indices]

        X, Y = self.__data_generation(batch_frames)

        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_frames):
        X_batch = []
        Y_batch = []

        for frame in batch_frames:
            try:
                image_path = os.path.join(self.image_dir, frame["image"]["name"])
                img = load_img(image_path, target_size=self.input_size)
                img_array = img_to_array(img) / 255.0

                X_batch.append(img_array)

                points = frame["annorect"][0]["annopoints"]["point"]
                joints = []
                for p in points:
                    joints.extend([p["x"], p["y"]])
                joints = np.array(joints, dtype=np.float32) / 512.0

                Y_batch.append(joints)

            except Exception as e:
                print(f"Skipping frame due to error: {e}")

        return np.array(X_batch, dtype=np.float32), np.array(Y_batch, dtype=np.float32)

