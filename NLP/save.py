from mnist import MNIST
import numpy as np

# Load EMNIST (only once)
mndata = MNIST("C:/study/NLP/NLP/emnist/emnist_source_files")
mndata.select_emnist('digits')
images, labels = mndata.load_training()

# Reshape and convert
images = np.array(images).reshape(-1, 28, 28)
labels = np.array(labels)

# Save as .npy
np.save("train_images.npy", images)
np.save("train_labels.npy", labels)

print("Preprocessed and saved data ✅")
