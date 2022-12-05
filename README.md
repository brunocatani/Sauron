# Sauron

--PT-BR--
 
Esse projeto é parte do "Projeto Integrador 2022/2" supervisionado pelo professor Rafael Barbosa

Este projeto tem como objetivo construir um sistema de tomada de presença utilizando um Nvidia Jetson Nano 4GB

  Projetado pelos integrantes:

- Bruno Henrique Catani : https://github.com/brunocatani/

- Maria Eduarda Martinelli : https://github.com/DudamaMartinelli

- Eduardo Dallarosa : https://github.com/EduardoDallarosa

- Emanuelli Graff : https://github.com/manugraff


Hardware necessario:

- Nvidia Jetson Nano
- Camera CSI/USB
- Dongle WI-FI


## Requerimentos 

- OpenCV 4.6.0 Compilado para CUDA & CUDNN: 
https://github.com/opencv/opencv/releases

- Jetpack SDK 4.6.2: 
https://developer.nvidia.com/embedded/jetpack-sdk-462

- Dlib 19.17: 
https://github.com/davisking/dlib

- Face Recognition 1.3.0: 
https://github.com/ageitgey/face_recognition

- Python-Telegram-Bot 13.14:
https://github.com/python-telegram-bot/python-telegram-bot



## Compilando OpenCV com CUDA na Jetson Nano

Tutorial por completo por QENGINEERING
https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html

## Compilando DLIB com CUDA + Face-Recognition

- Instalando dependências base

CTRL+T para abrir terminal

```
sudo apt-get update
sudo apt-get install python3-pip cmake libopenblas-dev liblapack-dev libjpeg-dev
```
- Dependência Numpy
```
pip3 install numpy
```
- Baixar e descompactar Dlib
```
wget http://dlib.net/files/dlib-19.17.tar.bz2 
tar jxvf dlib-19.17.tar.bz2
cd dlib-19.17
```
- Editar cudnn_dlibapi.cpp 
```
gedit dlib/cuda/cudnn_dlibapi.cpp
```

- CTRL+F para encontrar 
```
forward_algo = forward_best_algo;
```

- Editar linha
```
//forward_algo = forward_best_algo;

```
- Compilar e instalar Dlib (30-60 minutos na Jetson Nano)
```
sudo python3 setup.py install
```

- Instalar biblioteca Face_Recognition
```
sudo pip3 install face_recognition
```


--US-EN--

This project is part of "Projeto Integrador 2022/2" supervised by the profesor Rafael Barbosa
