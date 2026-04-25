from fastapi import FastAPI, HTTPException, UploadFile, File
from torchvision import transforms
# from pydantic import BaseModel
import streamlit as st
from PIL import Image
import torch.nn as nn
import uvicorn
import torch
import io



classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class CifarmClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, image):
        image = self.first(image)
        image = self.second(image)
        return image

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CifarmClassification()
model.load_state_dict(torch.load('cifar_10_model.pth', map_location=device))
model.to(device)
model.eval()

# app = FastAPI()
#
# @app.post('/predict')
# async def check_image(file:UploadFile = File(...)):
#     try:
#         data = await file.read()
#         if not data:
#             raise HTTPException(status_code=400, detail='File not Found')
#
#         img = Image.open(io.BytesIO(data))
#         img_tensor = transform(img).unsqueeze(0).to(device)
#
#         with torch.no_grad():
#             prediction = model(img_tensor)
#             result = prediction.argmax(dim=1).item()
#             return {f'class': classes[result]}
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f'{e}')
#
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)

st.title('Cifar-10 Model')
st.text('Загрузите изображение, и модель попробует её распознать.')

mnist_image = st.file_uploader('Выберите изображение', type=['PNG', 'JPG', 'JPEG', 'SVG'])

if not mnist_image:
    st.info('Загрузите изображение')
else:
    st.image(mnist_image, caption='Загруженное изображение')

    if st.button('Распознать'):
        try:
            image = Image.open(mnist_image)
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                y_prediction = model(image_tensor)
                prediction = y_prediction.argmax(dim=1).item()
            st.success(f'Модель думает, что это: {classes[prediction]}')

        except Exception as e:
            st.error(f'Ошибка: {str(e)}')
