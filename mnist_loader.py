import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split


def get_data_wrapper():
    (X,y) , (test_X, test_y) = mnist.load_data()
    # print(X.shape)
    # train_X = X.reshape(60000,784)
    # print(train_X.shape)
    # np.random.shuffle(train_X)
    # valid_X = train_X[50000:,::]
    # train_X = train_X[:50000,::]
    # print(valid_X.shape)
    # print(train_X.shape)
    # print(y.shape)
    Y = np.concatenate((y,test_y),axis=0)
    X = np.concatenate((X,test_X),axis=0)
    X = X.reshape(70000,784)
    print(X.shape)
    print(Y.shape)

    # X_train,y_train,X_test,y_test = train_test_split(X,Y,random_state=104,train_size=0.7,shuffle=True)
    # X_val,y_val,X_test,y_test = train_test_split(X_test,y_test,random_state=48,test_size=0.5,shuffle=True)

    X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=104,train_size=0.7,shuffle=True)
    X_val,X_test,y_val,y_test = train_test_split(X_test,y_test,random_state=48,test_size=0.5,shuffle=True)

    train_data = zip(X_train,y_train)
    val_data = zip(X_val,y_val)
    test_data = zip(X_test,y_test)

    
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_test.shape)
    print(y_test.shape)

    return train_data,val_data,test_data

