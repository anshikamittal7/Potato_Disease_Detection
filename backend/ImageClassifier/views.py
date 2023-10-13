import tensorflow as tf
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt



# Define the downloadImage function
# views.py


def process_and_save_image(image):
    # Open the uploaded image using Pillow
    img = Image.open(image)

    # Perform image processing using Pillow
    # For example, you can resize the image to a specific size
    img = img.resize((256, 256), Image.ANTIALIAS)

    # Save the processed image
    processed_image_path = 'processed_image.jpg'
    img.save(processed_image_path)

    return processed_image_path

def downloadImage(request):
    if request.method == 'POST':
        image = request.FILES['image']
        processed_image_path = process_and_save_image(image)

        return JsonResponse({'message': 'Image uploaded successfully.'})
    return JsonResponse({'error': 'No image provided in the request.'})


def runImageOnModel():

    # load_model = tf.keras.models.load_model("potatoes.h5")
    # test_image_path = "testDir/healthy_potato.jpg" 
    # img = Image.open(test_image_path)
    # img = img.resize((256, 256))  
    # img = np.array(img) / 255.0 

    # img = tf.convert_to_tensor(img)
    # img = tf.expand_dims(img, axis=0)

    # predictions = load_model.predict(img)
    # class_names = ["EARLY BLIGHT", "LATE BLIGHT", "HEALTHY"]  

    # plt.figure(figsize=(6, 8))
    # plt.imshow(img[0])  
    # true_class = class_names[np.argmax(predictions)]
    # return true_class
    pass

# Create your views here.
def hello(request):
    return HttpResponse("Hey")

@csrf_exempt
def getDiseaseStatus(request):
    if request.method== "POST":
        downloadImage(request)
        runImageOnModel()
        res= {
            "hey": "bro",
            "status": 200
        }

        return JsonResponse(res)

    return render(request, 'index.html')
