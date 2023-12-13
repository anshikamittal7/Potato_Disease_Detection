import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import numpy as np

# Load your models
model_names = ['./Valid Models/resnet50_model.h5', './Valid Models/inception_model.h5', './Valid Models/vgg16_model.h5' , './Valid Models/sequential_15layer.h5']
# , './Valid Models/sequential_15layer.h5'
models = []
for model_name in model_names:
    model = tf.keras.models.load_model(model_name)
    models.append(model)

# Load and preprocess the image

c=0
tot=0
for i in range(910, 920):
    
    img_path = f'./Data/testing/Potato___Healthy/image ({i}).JPG'
    img = image.load_img(img_path, target_size=(256, 256))  # Adjust target_size based on your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Plot the original image
    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, len(models)+1, 1)
    # plt.imshow(img)
    # plt.title('Original Image')

    # Make predictions for each model
    for i, model in enumerate(models, start=1):
        predictions = model.predict(img_array)

        # Decode predictions (for example, if using a model pre-trained on ImageNet)
        # Modify this part based on your model's output format
        classes = range(1, len(predictions[0]) + 1)  # Assuming indices are class labels
        probs = predictions[0]
        # Extract class labels and probabilities
        # labels, probs = zip(*decoded_predictions[0])  # Assuming top-1 prediction

        # Plot the results
        # plt.subplot(1, len(models)+1, i+1)
        # print("classes ", classes , "probs ", probs)
        # plt.barh(classes, probs, color='skyblue')
        # plt.xlim([0, 1])
        # plt.title(f'Model {i} Predictions')
        class_dict = {0: 'early blight', 1: 'late blight', 2: 'healthy'}
        pred_class = class_dict[np.argmax(probs)]

        # Output the prediction
        print(f'Model {i} Predicted:', pred_class)
        print(f'Class Probabilities: {probs}')

        # Keep track of class counts
        tot += 1
        if np.argmax(probs) == 1:
            c += 1

    # Print overall statistics

print(f'Total Healthy Predictions: {c}/{tot}')
plt.tight_layout()
plt.show()
