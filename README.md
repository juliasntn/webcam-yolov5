### README

# Detecção de Objetos com YOLOv5

Este projeto implementa um sistema de detecção de objetos em tempo real usando a webcam do computador e o modelo YOLOv5. O modelo é carregado utilizando a biblioteca `torch` e a detecção é feita em cada frame capturado da webcam.

## Requisitos

- Python 3.8 ou superior
- Bibliotecas Python:
  - `torch`
  - `opencv-python`
  - `numpy`
  - `Pillow`

## Instalação

1. Clone o repositório:

    ```bash
    git clone https://github.com/seu-usuario/seu-repositorio.git
    ```

2. Navegue até o diretório do projeto:

    ```bash
    cd seu-repositorio
    ```

3. Crie um ambiente virtual (opcional, mas recomendado):

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    .\venv\Scripts\activate  # Windows
    ```

4. Instale as dependências:

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install opencv-python numpy pillow
    ```

## Uso

Execute o script para iniciar a detecção de objetos:

```bash
python detectar_objetos.py
```

O script irá:

1. Carregar o modelo YOLOv5.
2. Capturar frames da webcam.
3. Realizar a detecção de objetos em cada frame.
4. Desenhar caixas delimitadoras e rótulos nos objetos detectados.
5. Exibir o frame com as detecções em uma janela.

Pressione a tecla 'q' para sair da aplicação.

## Código

```python
from PIL import Image
import torch
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
```

## Contribuição

Sinta-se à vontade para fazer um fork do projeto, criar uma nova branch e enviar pull requests. Agradecemos por suas contribuições!

## Contato
linkedin: https://www.linkedin.com/in/julia-santana-040a12180/
