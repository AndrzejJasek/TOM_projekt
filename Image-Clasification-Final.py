#Ten program jest tylko poczatkowa wersja fragmentu projektu stosujacego uczenie maszynowe do rozpoznania na zdjeciu testowym typu danych(nerka zdrowa, nerka z nowotworem, cos innego)
#Korzystanie z tego kodu w testach powodowalo przeciazenie pamieci komputera, dlatego czesc jest nie dokonczona
#Kod bierze pod uwage dwa "labels" danych: testowano na 60000 zdjaciach z danymi typu koty i psy
#Obrobke zdjec mozna dokonac przed przepuszczeniem danych przez ponizszy kod, ale zostala ona uwzgledniona
#dane otrzymane zostana manualnie podzielone na:
#20% danych testowych(losowe)
#40% zdjecia z nerkami chorymi
#40% zdjecia z nerkami zdrowymi
#dodatkowo zdjecia z obszarami nie bedacymi nerkami(np tomografia mozgu, przeswietlenie glowy itp.)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

print(tf.__version__)#TUTAJ WYPISUJEMY WERSIE TENSORFLOW'A DLA NASZEJ INFORMACJI

DATADIR=r"C:\Users\micro\OneDrive\Pulpit\DaneTreningowe"#FOLDER Z DWOMA MNIEJSZYMI FOLDERAMI(KAZDY ZAWIERAJACY ROWNA ILOSC ZDJEC
TESTDIR=r"C:\Users\micro\OneDrive\Pulpit\DaneTestowe"#FOLDER Z DANYMI KTORE BEDZIEMY TESTOWAC(OK 20% CALOSCI ZDJEC)

CATEGORIES = ['A', 'B']#NA POTRZEBY TEGO PROGRAMU ZASTOSOWANO KATEGORIE A I B.
#W KONCOWYM KODZIE BEDA 3 KATEGORIE:NERKA ZDROWA, NERKA Z NOWOTWOREM, OBIEKT NIE JEST NERKA


#PONIZSZY FRAGMENT TWORZY MACIERZ DANYCH PRZETWORZONYCH NA SKALE SZAROSCI
for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()

        break
    break

#TUTAJ OBROBKA ZDJEC: ROZMIAR, KSZTALT, ROZDZIELCZOSC
IMG_SIZE = 300
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

#W TRENINGDATA BEDZIEMY ZAMIESZCZAC DANE SLUZACE DO TRENINGU PROGRAMU
training_data = []
def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)): 
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()

print(len(training_data))#PODZIELIC PO ROWNO-DANE TRENINGOWE POWINNY ZACHOWAC ROWNOWAGE(TAKA SAMA ILOSC W FOLDERACH A I B)

import random#TA CZESC MIESZA DANE RANDOMOWO
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

#TUTAJ ZAMIESZCZAMY WYMIESZANE OBRAZY WRAZ Z ICH KATEGORIAMI
train_images = []
train_labels = []

for features,label in training_data:
    train_images.append(features)
    train_labels.append(label)
#KORZYSTAMY Z PICKLE ABY ZAPISAC DANE UMOZLIWIAJAC POZNIEJSZE POPRAWKI W PROGRAMIE I OSZCZEDZENIE CZASU
pickle_in = open("train_images.pickle","rb")
train_images = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
train_labels = pickle.load(pickle_in)

#ZMIANA ZAKRESU WARTOSCI W OBRAZACH
train_images = train_images / 255.0
test_images = test_images / 255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=train_images.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=32, epochs=10, validation_split=0.6)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)#OCENA PRACY MODELU
print('\nTest accuracy:', test_acc)#POPRAWNOSC PRACY KODU


#NASTEPNA CZESC POLEGA NA OBLICZENIU PRZWDOPODOBIENSTWA POPRAWNOSCI ODPOWIEDZI MODELU
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)


np.argmax(predictions[5])#JAKI TO TYP OBRAZU(PRZEWIDYWANY PRZEZ PROGRAM)
test_labels[5]#JAKI TO FAKTYCZNIE OBRAZ(0 LUB 1)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('green')



#WYSWIETLENIE DANEJ OBRAZOWEJ 5 I POKAZANIE NA WYKRESIE SLUPKOWYM PRZEWIDZIANEGO TYPU
i = 5
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

#TO SAMO CO WYZEJ TYLKO DLA 15 OBRAZOW POKAZANE NA MACIERZY
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()



img = test_images[0]
img = (np.expand_dims(img,0))
#PRZEWIDYWANIE TYPU OBRAZU
predictions_single = probability_model.predict(img)

print(predictions_single)#WYSWIETLENIE LICZB


#WYKRES SLUPKOWY PRZEWIDYWANIA TYPU OBRAZU
plot_value_array(0, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])#PRZEWIDZIANY TYP OBRAZU(0 LUB 1)

