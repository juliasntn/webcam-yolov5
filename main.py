from PIL import Image
import torch
from pathlib import Path
import cv2
import numpy as np

# Carregar o modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 's' refere-se ao tamanho do modelo (pequeno, médio, grande, etc.)

# Definir a webcam como fonte de entrada
cap = cv2.VideoCapture(0)

# Loop principal
while True:
    # Capturar frame da webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Converter o frame para RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Converter o frame para uma imagem PIL
    pil_img = Image.fromarray(rgb)

    # Fazer a detecção de objetos
    results = model(pil_img)

    # Desenhar caixas delimitadoras e rótulos nos objetos detectados
    results.render()

    # Exibir o frame com as caixas delimitadoras e rótulos
    frame = cv2.cvtColor(np.array(results.ims[0]), cv2.COLOR_RGB2BGR)
    cv2.imshow('Detecção de Objetos', frame)

    # Sair da tela se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a webcam
cap.release()

# Destruir todas as janelas
cv2.destroyAllWindows()
