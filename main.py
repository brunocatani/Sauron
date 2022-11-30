import cv2                      #OpenCV
import face_recognition         #Face-Recognition
import numpy as np              #Numpy
import csv                      #Biblioteca CSV
import os                       #Biblioteca OS, lê parametros sistema operacional
import dlib                     #DLIB
from datetime import datetime   #Lê valores de data e hora para csv


print(dlib.cuda.get_num_devices())

#Faz a captura da camera usando o OpenCV, Utilizando uma Pipeline pelo GSTREAMER

vcap = cv2.VideoCapture(0)

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


print(bruno_encoding)


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

directory = "confirmation/" + current_date

try:
    os.mkdir(directory)
except FileExistsError:
    pass

f = open("confirmation/" +current_date+".csv",'w+',newline = '')
lnwriter = csv.writer(f)


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

                    cv2.imwrite(os.path.join(directory, name+' '+current_time+".jpg"), frame)

            cv2.putText(frame, name, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))

    cv2.imshow("Sauron", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vcap.release()
cv2.destroyAllWindows()
f.close()




