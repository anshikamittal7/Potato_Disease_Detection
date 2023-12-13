from flask import Blueprint, request, url_for
from flask import render_template
from werkzeug.utils import secure_filename
from . import app
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

from . import app
plt.switch_backend('agg')
views = Blueprint("views", __name__)
@views.route('/', methods=['GET','POST'])
def home():
    if request.method == "POST":
        file = request.files['file']
        input_img = secure_filename(file.filename)
        file.save(app.config['IMAGE_UPLOADS']+input_img)

        pred=predict_save(input_img)
        return render_template('home.html', pred=pred, input_img=input_img, pred_img='pred_img.png', pred_img_seq='pred_img_seq.png', pred_img_inc='pred_img_inc.png', pred_img_vgg='pred_img_vgg.png', pred_img_res='pred_img_res.png')
    return render_template('home.html')


@views.route('/about')
def about():
    return render_template('about.html')

##############################################
model = load_model(app.config['MODEL'])
seq_model = load_model(app.config['SEQ_MODEL'])
inc_model = load_model(app.config['INC_MODEL'])
vgg_model = load_model(app.config['VGG_MODEL'])
res_model = load_model(app.config['RES_MODEL'])

class_names = ['Early_blight', 'Healthy', 'Late_blight']
class_names_x = ['Early_blight', 'Late_blight', 'Healthy']


def predict_save(img):
    tiny_image = load_img(app.config['IMAGE_UPLOADS']+img, target_size=(128, 128))
    tiny_image = img_to_array(tiny_image)
    tiny_image = np.expand_dims(tiny_image, axis=0)
    tiny_image= preprocess_input(tiny_image)
    
    createImg(model, tiny_image, 'norm', class_names)

    my_image = load_img(app.config['IMAGE_UPLOADS']+img, target_size=(256, 256))
    my_image = img_to_array(my_image)
    my_image = np.expand_dims(my_image, axis=0)
    my_image= preprocess_input(my_image)
    createImg(seq_model, my_image, 'seq', class_names_x)
    createImg(inc_model, my_image, 'inc', class_names_x)
    createImg(vgg_model, my_image, 'vgg', class_names_x)
    createImg(res_model, my_image, 'res', class_names_x)

def createImg(model, img, nam, class_names_var):
    out = np.round(model.predict(img)[0], 2)
    fig = plt.figure(figsize=(8, 5))
    plt.barh(class_names_var, 
            [1,1,1], 
            edgecolor='gray',
            linewidth=2,
            color='white',
            height=0.5)
    plt.barh(class_names_var,
             out, 
             color='lightgray', 
             height=0.5)
    
    for index, value in enumerate(out):
        plt.text(value/2, index, f"{100*value:.2f}%",fontsize=13, fontweight='bold')
        
    plt.xticks([])
    plt.yticks([0, 1, 2], labels=class_names_var, fontweight='bold', fontsize=14)
    name = app.config['IMAGE_UPLOADS']+'pred_img_'+nam+'.png'
    fig.savefig(name, bbox_inches='tight')
    print(class_names_var[np.argmax(out)])
    return class_names_var[np.argmax(out)]

