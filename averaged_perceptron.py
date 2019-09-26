import numpy as np
import pandas as pd


def avg_perceptron(X, labels, iterations=1, obs=None, w=None, u=None, mistakes=None):
    if w is None:
        w = np.random.random(len(X[0])).reshape(-1,1)
    if u is None:
        u = np.random.random(len(X[0])).reshape(-1,1)
    if mistakes is None:
        mistakes = []
    if obs is None:
        obs = X.shape[0]
    indexes = np.random.permutation(range(X.shape[0]))[:obs]
    c = 1

    for iters in range(iterations):
        # shuffle the data
        indexes = np.random.permutation(indexes)
        m = 0
        
        for i in indexes:
            x = X[i].reshape(-1,1)
            y = labels[i]

            yhat = np.sign(np.dot(w.T,x))
            if yhat != y: 
                w += y*x
                u += c*y*x
                
                m += 1
            c += 1              
            
        mistakes.append((iters, m))
        
        if m == 0:
            break
    
    w = w.reshape(w.shape[0],1) - 1/c*u
    
    mistakes = pd.DataFrame(mistakes)
    mistakes.columns = ['iterations', 'average perceptron mistakes']
    mistakes.set_index('iterations', inplace=True)
    
    return w, mistakes

#need to correct this
def avg_perceptron_MC(X, labels, iterations=1, obs=None, w=None, u=None, mistakes=None):
    if mistakes is None:
        mistakes = []
    if obs is None:
        obs = X.shape[0]
    indexes = np.random.permutation(range(X.shape[0]))[:obs]
    classes = np.unique(labels)
    if w is None:
        w = np.random.random(len(X[0]) * len(classes)).reshape(-1,1)
    if u is None:
        u = np.random.random(len(X[0]) * len(classes)).reshape(-1,1)
    c = 1

    for iters in range(iterations):
        # shuffle the data
        indexes = np.random.permutation(indexes)
        m = 0
        
        for i in indexes:
            x = X[i]
            y = labels[i]
            
            idx = np.argmax([np.dot(w.T, make_F(x, k, classes)) for k in classes])
            yhat = classes[idx]
            F_yhat = make_F(x, yhat, classes)
            F_y = make_F(x, y, classes)
            
            if y != yhat: 
                w += (F_y - F_yhat)
                u += c*(F_y - F_yhat)
                
                m += 1
            c += 1            
            
        mistakes.append((iters, m))
        
        if m == 0:
            break
    
    w = w.reshape(w.shape[0],1) - 1/c*u
    
    mistakes = pd.DataFrame(mistakes)
    mistakes.columns = ['iterations', 'average perceptron mistakes']
    mistakes.set_index('iterations', inplace=True)
    
    return w, mistakes


def make_F(x, k, classes):
    # k is the desired class
    F = []
    for c in classes:
        if k == c:
            F = np.concatenate([F, x], axis=0)
        else:
            F = np.concatenate([F, np.zeros(len(x))], axis=0)
    return F.reshape(-1, 1)

