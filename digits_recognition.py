import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the trained model
model = load_model('E:/pycharmik/number recognition/trained_model.h5')
#model.summary()

# Image preprocessing
def Preprocessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgHist = cv2.equalizeHist(imgGray)
    imgHist = imgHist/255
    #imgHist = cv2.resize(imgHist,(100,100))
    #cv2.imshow("Gray", imgGray)
    #cv2.imshow("Preprocesssed", imgHist)
    return imgHist

# Prediction
print('----------PREDYKCJA-----------')


def plot_image(i, predictions_array, true_label, img):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(predictions_array, true_label):
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')




class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five',
               'Six', 'Seven', 'Eight', 'Nine']
class_labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
imgPath = 'E:/pycharmik/number recognition/test images'
images = []


for img in os.listdir(imgPath):
    img = cv2.imread(os.path.join(imgPath,img))
    if img is not None:
        images.append(img)
print(f"Loaded {len(images)} images")


# SHOW TEST IMAGES
#plt.figure(figsize=(5,5))
#for i in range(10):
#    images[i] = cv2.resize(images[i],(32,32))
#    images[i] = Preprocessing(images[i])
#    plt.subplot(2,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(images[i])

#plt.show()



num_rows = 4
num_cols = 5
num_images = num_rows*num_cols


plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    img = np.asarray(images[i])
    img = cv2.resize(img,(32,32))
    img = Preprocessing(img)
    img = img.reshape(1,32,32,1)
    prediction = model.predict(img)#[0][0]
    print(np.argmax(prediction))
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, prediction, class_labels[i], images[i])
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(prediction[0], class_labels[i])
plt.tight_layout()
plt.show()

'''

# CAPTURE CAMERA

cap = cv2.VideoCapture(0)
cap.set(4,640)
cap.set(3,480)


while True:
    ret, frame = cap.read()
    img = np.asarray(frame)
    img = cv2.resize(img,(32,32))
    img = Preprocessing(img)
    img = img.reshape(1,32,32,1)
    prediction = model.predict(img)#[0][0]
    #print(prediction[0])

    classes = np.argmax(prediction)
    val = np.round(100*np.max(prediction),2)
    print(val)
    #print("NUMBER: ", classes, " PROPABILITY: ", val)
    if val > 50:
        cv2.putText(frame, "Number: " + str(classes) + "   "+str(val) + "%",
                        (50,50),cv2.FONT_HERSHEY_COMPLEX,
                        1,(0,0,255),1)

    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()


'''