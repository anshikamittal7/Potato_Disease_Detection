import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import numpy as np

# Load your models
model_names = ['./Valid Models/resnet50_model.h5', './Valid Models/inception_model.h5', './Valid Models/vgg16_model.h5' , './Valid Models/sequential_15layer.h5']
models = [tf.keras.models.load_model(model_name) for model_name in model_names]

# Load and preprocess the image
c = 0
tot = 0
ensemble_predictions = []
arr= ["Early_blight", "Late_blight", "Healthy"]
x= 1
strx= "Potato___"+arr[x]

for i in range(910, 920):
    img_path = f'./Data/testing/{strx}/image ({i}).JPG'
    img = image.load_img(img_path, target_size=(256, 256))  # Adjust target_size based on your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions for each model
    model_predictions = []

    for model in models:
        predictions = model.predict(img_array)
        model_predictions.append(np.argmax(predictions))

    # Decide the majority prediction
    majority_prediction = np.bincount(model_predictions).argmax()

    # Output the majority prediction
    class_dict = {0: 'early blight', 1: 'late blight', 2: 'healthy'}
    pred_class = class_dict[majority_prediction]
    print(f'Majority Predicted:', pred_class)

    # Keep track of class counts
    tot += 1
    if majority_prediction == x:  # Assuming 2 is the index for 'healthy' class
        c += 1

    ensemble_predictions.append(majority_prediction)

# Print overall statistics
print(f'Total {arr[x]} Predictions: {c}/{tot}')
print(f'Ensemble Predictions: {ensemble_predictions}')
