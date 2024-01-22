from ultralytics import YOLO
from torchvision import transforms as T
from torchvision import io

model = YOLO('flask_app/weights/best.pt')
trnsform = T.Compose([
    T.ToTensor(),
    T.Resize((320, 320)),
    T.ToPILImage()
])

def load_model():

    return model

def load_transform():

    return trnsform