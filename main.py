import cv2                      #OpenCV
import face_recognition         #Face-Recognition
import numpy as np              #Numpy
import csv                      #Biblioteca CSV
import os                       #Biblioteca OS, lê parametros sistema operacional
import dlib                     #DLIB
from datetime import datetime   #Lê valores de data e hora para csv

#Telegram Bot
import requests



def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=960,
    display_height=540,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

#Faz a captura da camera usando o OpenCV, Utilizando uma Pipeline pelo GSTREAMER

vcap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)


#Carrega as imagens individualmente e faz a encodificação dela, armazenando dentro do Array

bruno_image = face_recognition.load_image_file("database/bruno.jpeg")
bruno_encoding = face_recognition.face_encodings(bruno_image)[0]

manu_image = face_recognition.load_image_file("database/manu.jpeg")
manu_encoding = face_recognition.face_encodings(manu_image)[0]

duda_image = face_recognition.load_image_file("database/duda.jpeg")
duda_encoding = face_recognition.face_encodings(duda_image)[0]

du_image = face_recognition.load_image_file("database/du.jpeg")
du_encoding = face_recognition.face_encodings(du_image)[0]

roni_image = face_recognition.load_image_file("database/roni.jpeg")
roni_encoding = face_recognition.face_encodings(roni_image)[0]


kface_enconding = [
    bruno_encoding,
    manu_encoding,
    duda_encoding,
    du_encoding,
    roni_encoding,
]

kface_names = [
    "Bruno Catani",
    "Emanuelli Graff",
    "Maria Eduarda Martinelli",
    "Eduardo Dalarosa",
    "Roni",
]

students = kface_names.copy()

face_locations = []
face_encodings = []
face_names = []
processing=True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

directory = "confirmation"

try:
    os.mkdir(directory)
except FileExistsError:
    pass

date_directory = "confirmation/" + current_date

try:
    os.mkdir(date_directory)
except FileExistsError:
    pass

f = open("confirmation/" +current_date+".csv",'w+',newline = '')
lnwriter = csv.writer(f)

def send_to_telegram(message):

    apiToken = '5881673980:AAHB5N8l0fK-P-fvOpYncrrNYK2V3sVOg7c'
    chatID = '-1001812740148'
    apiURL = f'https://api.telegram.org/bot5881673980:AAHB5N8l0fK-P-fvOpYncrrNYK2V3sVOg7c/sendMessage'

    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
        print(response.text)
    except Exception as e:
        print(e)


send_to_telegram("Chamada iniciada")


while True:

    ret, frame = vcap.read()

    small_frame = cv2.resize(frame, (0,0), None, fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if processing:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encodings in face_encodings:
            matches = face_recognition.compare_faces(kface_enconding, face_encodings)
            name = ""
            face_distance = face_recognition.face_distance(kface_enconding, face_encodings)
            best_index = np.argmin(face_distance)

            if matches[best_index]:
                name = kface_names[best_index]

            face_names.append(name)

            if name in kface_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name,current_time])
                    print(name)

                    cv2.imwrite(os.path.join(date_directory, name+' '+current_time+".jpg"), frame)

                    send_to_telegram('Bem vindo(a) '+name+" - Horario: "+current_time)


            cv2.putText(frame, name, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))

    cv2.imshow("Sauron", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vcap.release()
cv2.destroyAllWindows()
f.close()




