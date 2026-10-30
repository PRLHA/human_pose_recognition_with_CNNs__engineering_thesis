import tensorflow as tf
import tensorflow_hub as hub
import os

general_path = "./"
hrnet_url = "https://tfhub.dev/google/HRNet/ade20k-hrnetv2-w48/1"
saved_hrnet_path = general_path + "hrnet_saved"

# Download HRNet module from TF Hub
hrnet = hub.load(hrnet_url)

# Save it as a TF SavedModel to disk
tf.saved_model.save(hrnet, saved_hrnet_path)

print(f"✅ HRNet downloaded and saved to: {saved_hrnet_path}")
