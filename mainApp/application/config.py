class Config:
    DEBUG = True
    IMAGE_UPLOADS = "application/static/uploads/"
    SESSION_COOKIE_SECURE = False
    SECRET_KEY = "secretkeysuperkey"
    MODEL = "application/static/models/model_v1.h5"
    SEQ_MODEL = "application/static/models/sequential_15layer.h5"
    INC_MODEL = "application/static/models/inception_model.h5"
    VGG_MODEL = "application/static/models/vgg16_model.h5"
    RES_MODEL = "application/static/models/resnet50_model.h5"

