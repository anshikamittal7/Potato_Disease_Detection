import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

IMAGE_SIZE = 256
CHANNELS = 3

# Data Augmentation with validation split
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True,
    validation_split=0.2  # 80% training, 20% validation
)

# Create training generator
train_generator = train_datagen.flow_from_directory(
    'Data/training',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode="sparse",
    subset='training'  # Specify training subset
)

# Create validation generator
validation_generator = train_datagen.flow_from_directory(
    'Data/training',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode="sparse",
    subset='validation'  # Specify validation subset
)

# Create InceptionV3 base model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

n_classes = 3

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Build your model using functional API
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(n_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=predictions)

model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with validation data
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    verbose=1,
    validation_data=validation_generator,  # Add validation data
    validation_steps=len(validation_generator)  # Add validation steps
)

# Save the model
model.save("inception_model.h5")

# Plot training and validation history
plt.figure(figsize=(12, 4))

# Plot training accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
