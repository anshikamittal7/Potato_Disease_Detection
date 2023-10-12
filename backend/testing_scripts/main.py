import tensorflow as tf
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
print(os.getcwd())


load_model = tf.keras.models.load_model("potatoes.h5")
# load_model.summary()
# test_dir= "testDir"
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 1
# test_data = tf.keras.preprocessing.image_dataset_from_directory(
#     directory = test_dir,
#     image_size = IMG_SIZE,
#     label_mode = 'categorical',
#     batch_size = BATCH_SIZE
# ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# load_model.evaluate(test_data)

test_image_path = "testDir/healthy_potato.jpg"  # Replace with the actual path to your image
img = Image.open(test_image_path)
img = img.resize((256, 256))  # Resize to the same dimensions as your model's input
img = np.array(img) / 255.0  # Convert to NumPy array and normalize

# Add a batch dimension to the image
img = tf.convert_to_tensor(img)
img = tf.expand_dims(img, axis=0)

# Make predictions on the test image
predictions = load_model.predict(img)
print(predictions)
class_names = ["EARLY BLIGHT", "LATE BLIGHT", "HEALTHY"]  # Replace with your class names

# Display the image and prediction
plt.figure(figsize=(6, 8))
plt.imshow(img[0])  # Display the image
true_class = class_names[np.argmax(predictions)]
title = f"Predicted class: {true_class}"
plt.title(title)
plt.axis("off")
plt.show()
print(predictions)