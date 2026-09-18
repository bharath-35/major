from tensorflow.keras.models import load_model

model = load_model(r"D:\fraud-detection-federated\docs\global_model.h5")
model.summary()