from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Define constants
IMAGE_SIZE = 256
BATCH_SIZE = 5

# Specify the paths to your training and testing datasets
train_data_dir = 'Data/training'
test_data_dir = 'Data/testing'

# Data Augmentation for the dataset
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create a DataFrame with file paths and labels for training data
train_file_paths = []
train_labels = []

for class_folder in os.listdir(train_data_dir):
    class_path = os.path.join(train_data_dir, class_folder)
    class_label = class_folder
    class_files = [os.path.join(class_path, file) for file in os.listdir(class_path)]
    train_file_paths.extend(class_files)
    train_labels.extend([class_label] * len(class_files))

train_df = pd.DataFrame({'file_paths': train_file_paths, 'labels': train_labels})

# Create a DataFrame with file paths and labels for testing data
test_file_paths = []
test_labels = []

for class_folder in os.listdir(test_data_dir):
    class_path = os.path.join(test_data_dir, class_folder)
    class_label = class_folder
    class_files = [os.path.join(class_path, file) for file in os.listdir(class_path)]
    test_file_paths.extend(class_files)
    test_labels.extend([class_label] * len(class_files))

test_df = pd.DataFrame({'file_paths': test_file_paths, 'labels': test_labels})  # Make sure 'labels' is a valid column name

# Create the generators for training and testing data
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='file_paths',
    y_col='labels',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=True
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='file_paths',
    y_col='labels',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

model = tf.keras.models.load_model("./Valid Models/vgg16_model.h5")

train_accuracy = model.evaluate(train_generator)[1]
test_accuracy = model.evaluate(validation_generator)[1]

print(f"InceptionV3 Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"InceptionV3 Testing Accuracy: {test_accuracy * 100:.2f}%")


validation_steps = len(validation_generator)
validation_generator.reset()  # Reset the generator to start at the beginning of the validation set
predictions = model.predict(validation_generator, steps=validation_steps, verbose=1)

# Convert predictions to class labels
predicted_classes = predictions.argmax(axis=1)

# Get the true labels
true_classes = validation_generator.classes

# Define class names
class_names = list(validation_generator.class_indices.keys())
# Compute the confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Print a classification report
class_names = list(validation_generator.class_indices.keys())
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_names))
