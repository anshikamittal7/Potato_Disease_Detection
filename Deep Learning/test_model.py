
# Load the image file
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# Load the pre-trained model
model = load_model('./Valid Models/vgg16_model.h5')

# Load the image file
c=0
tot=0
script_dir = os.path.dirname(__file__)  # Get the directory of the script
for i in range(901, 980) :

    # image_path = os.path.join(script_dir, 'Data', 'training', 'Potato___Early_blight', f'image ({i}).JPG')
    # image_path = os.path.join(script_dir, 'Data', 'training', 'Potato___Late_blight', f'image ({i}).JPG')
    image_path = os.path.join(script_dir, 'Data', 'training', 'Potato___healthy', f'image ({i}).JPG')

    img = image.load_img(image_path, target_size=(256, 256))

    # Convert the image to a numpy array
    x = image.img_to_array(img)

    # Add a fourth dimension (since Keras expects a list of images)
    x = np.expand_dims(x, axis=0)

    # Scale the input image to the range used in the trained network
    x = x / 255.0


    preds = model.predict(x)
    print(str(i)+ " : "+ str(preds))
    # Map the predicted values to the corresponding classes
    class_dict = {0: 'early blight', 1: 'late blight', 2: 'healthy'}
    pred_class = class_dict[np.argmax(preds)]

    # Output the prediction
    if(np.argmax(preds) == 2):
         c= c+1
    tot= tot+1
    print('Predicted:', pred_class)

print(c,"/", tot )