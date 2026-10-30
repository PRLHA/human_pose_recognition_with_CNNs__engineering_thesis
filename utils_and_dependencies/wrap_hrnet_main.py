from tensorflow import keras
from hrnet_wrapper import HRNetBackboneLayer

LEGACY_SM_DIR = "./hrnet_saved"
OUT_PATH      = "hrnet_converted.keras"

inputs = keras.Input((512,512,3))
x = HRNetBackboneLayer(LEGACY_SM_DIR)(inputs)
model = keras.Model(inputs, x)
model.save(OUT_PATH)
print("saved ✅")

