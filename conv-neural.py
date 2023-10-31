import numpy as np
import cv2
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TRAIN_DIR = 'F:/neural-network/All-700-1'
TEST_DIR = 'F:/saf/test_dir'
IMG_SIZE_x = 320
IMG_SIZE_y = 240
LR = 1e-3

MODEL_NAME = 'All-700-1-{}-{}.model'.format(LR, '2conv-basic-video-1')

def label_img(img):
    word_label = img.split('.')[0]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'A':
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #                             [no cat, very doggo]
    elif word_label == 'B':
        return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'C':
        return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'D':
        return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'E':
        return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'F':
        return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'G':
        return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'H':
        return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'I':
        return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'K':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'L':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'M':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'N':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'O':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'P':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'Q':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'R':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'S':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'T':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'U':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'V':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'W':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'X':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'Y':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'J':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'Z':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'My':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    elif word_label == 'Name':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    elif word_label == 'Yes':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    elif word_label == 'No':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    elif word_label == 'Your':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    elif word_label == 'Hello':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

    elif word_label == 'Undo':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    elif word_label == 'Gar':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        #print(label)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE_x, IMG_SIZE_y))
        #print(training_data)
        training_data.append([np.array(img), np.array(label)])

    tqdm(os.listdir(TRAIN_DIR)).close()
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE_x, IMG_SIZE_y))
        testing_data.append([np.array(img), img_num])

    tqdm(os.listdir(TEST_DIR)).close()
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data



def create_or_load_model(IMG_SIZE_x, IMG_SIZE_y, train_data):


    convnet = input_data(shape=[None, IMG_SIZE_x, IMG_SIZE_y, 1], name='input')


    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)


    convnet = fully_connected(convnet, 1024, activation='relu')
    #convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 34, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')
        return model
    else:
        train = train_data
        test = train_data[-8000:]

        X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE_x, IMG_SIZE_y, 1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE_x, IMG_SIZE_y, 1)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}),
                  snapshot_step=500, show_metric=True, run_id=MODEL_NAME)



        model.save(MODEL_NAME)

        return model



train_data = create_train_data()


# If you have already created the dataset:
#train_data = np.load('train_data.npy', allow_pickle= True)




model = create_or_load_model(IMG_SIZE_x, IMG_SIZE_y, train_data)

# cap = cv2.VideoCapture(0)
#
# x1 = 0
# x2 = 240
#
# y1 = 70
# y2 = 410
#
# font = cv2.FONT_HERSHEY_SIMPLEX
# color = (0, 0, 255)
# stroke = 1
#
#
# while (True):
#     _, frame = cap.read()
#
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # lower_blue = np.array([90, 80, 80])
#     # upper_blue = np.array([110, 255, 255])   # BASA
#
#     lower_blue = np.array([0, 70, 70])
#     upper_blue = np.array([130, 255, 255])
#
#
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
#     res = cv2.bitwise_and(frame, frame, mask=mask)
#
#     median = cv2.medianBlur(res, 3)
#
#     median = median[y1:y2, x1:x2]
#
#     gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
#
#     gray = gray.reshape(1, IMG_SIZE_x, IMG_SIZE_y, 1)
#
#     model_out = model.predict(gray)[0]
#     # print(model_out)
#
#     if np.argmax(model_out) == 0:
#         str_label = 'A'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 1:
#         str_label = 'B'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 2:
#         str_label = 'C'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 3:
#         str_label = 'D'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 4:
#         str_label = 'E'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 5:
#         str_label = 'F'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 6:
#         str_label = 'G'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 7:
#         str_label = 'H'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 8:
#         str_label = 'I'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 9:
#         str_label = 'K'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 10:
#         str_label = 'L'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 11:
#         str_label = 'M'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 12:
#         str_label = 'N'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 13:
#         str_label = 'O'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 14:
#         str_label = 'P'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 15:
#         str_label = 'Q'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 16:
#         str_label = 'R'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 17:
#         str_label = 'S'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 18:
#         str_label = 'T'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 19:
#         str_label = 'U'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 20:
#         str_label = 'V'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 21:
#         str_label = 'W'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 22:
#         str_label = 'X'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     elif np.argmax(model_out) == 23:
#         str_label = 'Y'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#     else:
#         str_label = 'nothing'
#         cv2.putText(median, str_label, (100, 100), font, stroke, color, 1, cv2.LINE_AA)
#
#
#     # cv2.waitKey(50)
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 0)
#     cv2.imshow('Median Blur', median)
#     cv2.imshow('frame', frame)
#
#     if cv2.waitKey(100) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()
# cap.release()