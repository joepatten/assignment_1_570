import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wget
import os
from tensorflow.examples.tutorials.mnist import input_data


def view_image(obs):
    square_obs = np.reshape(obs,(28,28))
    plt.imshow(square_obs, cmap="gray")
    plt.show()


def plot_data(df, y_lst, filename):
    for y_var in y_lst:
        x_name = df.index.name
        df.reset_index().plot(x=x_name, y=y_var, legend=False)
        plt.ylabel(y_var.split()[-1])
        f = '_{}'.format(y_var.split()[0]).join(filename.rsplit('.', 1))
        plt.savefig(f, bbox_inches="tight")
        plt.title(y_var)
        plt.tight_layout()
        


def g_learning_curve(func, X_train, X_test, y_train, y_test, binary=True, iterations=20):
    d = []
    for s_size in range(5000, 55000+1, 5000):
        w, mistakes = func(X_train, y_train, obs=s_size, iterations=iterations)
        score = calc_acc_score(X_test, y_test, w, binary)
        d.append([s_size, score])
    df = pd.DataFrame(d).rename(columns={0:'observation size', 1:'accuracy'}).set_index('observation size')
    return df


def classify(X, w, binary, classes=None, b=0):
    if binary == True:
        preds = np.sign(X@w)\
                  .reshape(X.shape[0],)
    else:
        preds = []
        for i in range(X.shape[0]):
            x = X[i]
            
            idx = np.argmax([np.dot(w.T, make_F(x, k, classes)) for k in classes])
            yhat = classes[idx]
            
            preds.append(yhat)
    return preds


def calc_acc_score(X, y, w, binary):
    yhat = classify(X, w, binary, classes=np.unique(y))
    return np.mean(yhat == y)

def acc_scores(iterations, func, X_train, X_test, y_train, y_test, binary=True):   
    train_scores = []
    test_scores = []
    w = None
    for i in range(1, iterations + 1):
        w, _ = func(X_train, y_train, w=w)
        train_scores.append(calc_acc_score(X_train, y_train, w, binary))
        test_scores.append(calc_acc_score(X_test, y_test, w, binary))
    all_scores = pd.DataFrame([train_scores, test_scores]).T
    all_scores.columns = ['training scores', 'test scores']
    all_scores.index.name = 'iterations'
    all_scores.index += 1
    return all_scores

def make_F(x, k, classes):
    # k is the desired class
    F = []
    for c in classes:
        if k == c:
            F = np.concatenate([F, x], axis=0)
        else:
            F = np.concatenate([F, np.zeros(len(x))], axis=0)
    return F.reshape(-1, 1)


def plot_curves(df, x, y_lst):
    for y_var in y_lst:
        df.plot(x=x, y=y_var)
        plt.ylabel('accuracy')
        plt.title(y_var)
        plt.show()
        plt.savefig('/figures/' + y_var.replace(' ','_') + '_accuracy.png')


def load_data():
    filenames = ['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz']
    link = r'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/'


    for filename in filenames:
        print(f'Downloading {filename} from github.')
        f = os.path.join(link, filename)
        wget.download(f, './data/fashion')
    
    
    data = input_data.read_data_sets('./data/fashion')
    
    # training labels
    labels_raw = data.train.labels
    labels = [-1 if x%2 == 0 else 1 for x in labels_raw]
    labels = np.array(labels)
    
    # test labels
    test_labels_raw = data.test.labels
    test_labels = [-1 if x%2 == 0 else 1 for x in test_labels_raw]
    test_labels = np.array(test_labels)
    
    # training features
    images = data.train.images
    
    # test feaures
    test_images = data.test.images
    
    return images, labels, labels_raw, test_images, test_labels, test_labels_raw