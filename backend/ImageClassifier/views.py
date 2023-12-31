import tensorflow as tf
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from PotatoDiseaseBackend import settings



def runImageOnModel(image_filename):
    models_dir= settings.MODELS_DIR
    potato_model_path= os.path.join(models_dir, "potatoes.h5")
    load_model = tf.keras.models.load_model(potato_model_path)
    save_dir = settings.UPLOADS_DIR
    test_image_path = os.path.join(save_dir, image_filename)
    img = Image.open(test_image_path)
    img = img.resize((256, 256))  
    img = np.array(img) / 255.0 

    img = tf.convert_to_tensor(img)
    img = tf.expand_dims(img, axis=0)

    predictions = load_model.predict(img)
    class_names = ["EARLY BLIGHT", "LATE BLIGHT", "HEALTHY"]  

    # plt.figure(figsize=(6, 8))
    # plt.imshow(img[0])  
    true_class = class_names[np.argmax(predictions)]
    return true_class
    

# Create your views here.
def hello(request):
    return HttpResponse("Hey")

@csrf_exempt
def getDiseaseStatus(request):
    if request.method == "POST":
        uploaded_image = request.FILES.get('image')

        if uploaded_image:
            save_dir = settings.UPLOADS_DIR
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)

            image_filename = os.path.join(save_dir, uploaded_image.name)

            with open(image_filename, 'wb+') as destination:
                for chunk in uploaded_image.chunks():
                    destination.write(chunk)
                    

            diseaseType = runImageOnModel(image_filename)
            res = {
                "message": "Image uploaded and saved successfully",
                "predictedDiseaseType": diseaseType,
                "status": 200
            }
        else:
            res = {
                "message": "No image uploaded",
                "predictedDiseaseType": "Error",
                "status": 400
            }
        print(res)

        return JsonResponse(res)

    return render(request, 'index.html')