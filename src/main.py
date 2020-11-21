#The code is inspired from https://github.com/keshavoct98/DANCING-AI

import os
from display_output import displayResults
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf 
import cv2
import time
import numpy as np
import pandas as pd
import librosa
from absl import app, flags
from absl.flags import FLAGS

flags.DEFINE_string('video', 'data/0.mp4', 'Input video path')
flags.DEFINE_string('audio', 'data/0.wav', 'Input video path')
flags.DEFINE_string('background', 'data/bg0.jpg', 'path to background image')
protoFile = "models/pose_deploy_linevec.prototxt"
weightsFile = "models/pose_iter_440000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
inWidth, inHeight, threshold = 256, 256, 0.3
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13]]


''' Missing values in pose coordinates are replaced using forward and 
    backward filling method. Rows left with missing values after applying mentioned methods
    are deleted. Last few rows are dropped to match lengths of X and Y.
    '''
def XY(df, audio_input):
   
    min_length = min(audio_input.shape[0], df.shape[0])
    X = audio_input[:min_length, :]
    Y = df.iloc[:min_length, :]
    
    Y = Y.astype('float64')
    Y.replace(to_replace=-1, value=np.nan, inplace = True)
    Y.fillna(method='ffill',axis = 1, inplace = True)
    Y.fillna(method='bfill',axis = 1, inplace = True)
    Y = Y.dropna()
    
    X = X[Y.index]
    Y = Y.values
    return X, Y


''' Displays video with estimated pose. Returns pose coordinates and audio tempogram as numpy array.'''
    
def video(vid_path, aud_path):
    cap = cv2.VideoCapture(vid_path)
    hasFrame, frame = cap.read()
    x,y,w = frame.shape
    bg = cv2.imread(r"C:\Users\Devashi Jain\Desktop\IIIT-D\fun\DANCING-AI-master\data\bg1.jpg")
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    df = pd.DataFrame(columns = range(28))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    print('Video Processed(In sec):-', end = ' ')
    while(1):
        black_image = cv2.resize(bg, (frameWidth, frameHeight),interpolation = cv2.INTER_NEAREST)
        frame_count = frame_count + 1
        if frame_count % int(fps) == 0:
            print(int(frame_count / fps), end=', ', flush=True)
        t = time.time()  
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0/255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        H, W = output.shape[2], output.shape[3]

        points = []
        list_coordinates = []
        for i in range(14):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x, y = (frameWidth * point[0]) / W, (frameHeight * point[1]) / H

            if prob > threshold :
                points.append((int(x), int(y)))
                list_coordinates.extend([int(x), int(y)])                  
            else :
                points.append(None)
                list_coordinates.extend([-1, -1])
        df.loc[len(df)] = list_coordinates
        
        for pair in POSE_PAIRS:
            partA, partB = pair[0], pair[1]
            if points[partA] and points[partB]:
                cv2.line(black_image, points[partA], points[partB], (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(black_image, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(black_image, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), 
                    cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        cv2.imshow("original", frame)
        cv2.imshow('Output-Skeleton', black_image)
        if cv2.waitKey(1) == 27: break
    
    cv2.destroyAllWindows()
    print()
    y, sr = librosa.load(aud_path)
    audio_input = np.transpose(librosa.feature.tempogram(y, sr, hop_length = int(sr/fps), win_length = 36))

    return XY(df, audio_input)

''' Scaling and Train-test split of data.'''
def preprocess(X, Y):
    test_data = X.shape[0] - int(X.shape[0]/20) # 5% data is used for predictions
    X_train = X[:test_data, :]
    Y_train = Y[:test_data, :]
    X_test = X[test_data:, :]
    Y_test = Y[test_data:, :]
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    Y_train = scaler.fit_transform(Y_train)
    Y_test = scaler.transform(Y_test)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, X_test, Y_train, Y_test, scaler

''' Splits data into train-test, trains lstm on training split
    and return predictions on test data.'''
def train(X, Y):
   
    
    X_train, X_test, Y_train, Y_test, scaler = preprocess(X, Y)
    
    keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(units = 72, input_shape = (36,1), return_sequences = True))
    model.add(LSTM(54, activation = 'tanh'))
    model.add(Dense(28))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(X_train, Y_train, batch_size = 16, epochs = 400)
    
    return scaler.inverse_transform(model.predict(X_test))


def main(_argv):
    
    head, tail = os.path.split(FLAGS.video)
    csv_xpath = 'data/'+tail.split('.')[0]+'X.csv'
    csv_ypath = 'data/'+tail.split('.')[0]+'Y.csv' 
    if os.path.isfile(csv_xpath) and os.path.isfile(csv_ypath):
        ''' If csv file is found, pose estimation part is skipped and 
            lstm is directly trained on the inputs from csv file.'''
        X, Y = pd.read_csv(csv_xpath, header = None), pd.read_csv(csv_ypath, header = None)
        X, Y = X.values, Y.values
    else:
        X, Y = video(FLAGS.video, FLAGS.audio) # Returns pose estimation coordinates of video and tempogram of input audio.
        pd.DataFrame(X).to_csv(csv_xpath, header = None, index = None)
        pd.DataFrame(Y).to_csv(csv_ypath, header = None, index = None)
        
    predictions = train(X, Y) # Split data, train lstm and return predictions.
    displayResults(predictions, FLAGS.background) # display and save output video
    print('video saved at "output/output.avi"')
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass