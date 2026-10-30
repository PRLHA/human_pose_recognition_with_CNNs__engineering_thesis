import os
import numpy as np
from tensorflow.keras.utils import Sequence  # ensure Sequence is imported

class FeatureDataGenerator(Sequence):
    def __init__(self,
        frames,
        feature_dir,
        batch_size=16,
        num_keypoints=16,
        shuffle=True
    ):
        self.frames        = frames
        self.feature_dir   = feature_dir
        self.batch_size    = batch_size
        self.num_keypoints = num_keypoints
        self.shuffle       = shuffle
        self.indices       = np.arange(len(frames))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.frames) / self.batch_size))

    def __getitem__(self, idx):
        batch_inds   = self.indices[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_frames = [self.frames[i] for i in batch_inds]
        X, Y = self.__data_generation(batch_frames)
        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_frames):
        X_batch = []
        Y_batch = []
        for frame in batch_frames:
            base = os.path.splitext(frame['image']['name'])[0]

            # try loading .npy or .npz feature files
            npy_path = os.path.join(self.feature_dir, base + ".png" + '.npy')
            npz_path = os.path.join(self.feature_dir, base + ".png" + '.npz')
            if os.path.exists(npy_path):
                feat = np.load(npy_path)
            elif os.path.exists(npz_path):
                data = np.load(npz_path)
                feat = data.get('features', None)
                if feat is None:
                    raise ValueError(f".npz file found but no 'features' key: {npz_path}")
            else:
                raise FileNotFoundError(f"Feature file not found for frame {base}: looked for {npy_path} or {npz_path}")

            feat = feat.astype(np.float32)

            points = frame['annorect'][0]['annopoints']['point']
            joints = []
            for p in points:
                joints.extend([p['y'] / 512.0, p['x'] / 512.0])#Modyfikacja zamieniono x i y joints.extend([p['x'] / 512.0, p['y'] / 512.0])
            joints = np.array(joints, dtype=np.float32)

            X_batch.append(feat)
            Y_batch.append(joints)

        X = np.stack(X_batch, axis=0)
        Y = np.stack(Y_batch, axis=0)
        return X, Y
      
    def get_file_name(self, index):
        """Retrieve the file name corresponding to the given index."""
        if index < 0 or index >= len(self.frames):
            raise IndexError("Index out of bounds.")
        
        frame = self.frames[self.indices[index]]
        base = os.path.splitext(frame['image']['name'])[0]
        return base
