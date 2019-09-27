import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import os
    

def pa(X, labels, iterations=1, obs=None, w=None, mistakes=None):
    # X = np.append(X,np.ones([len(X),1]),1)
    if w is None:
        w = np.random.random(len(X[0])).reshape(-1,1)
    if mistakes is None:
        mistakes = []
    if obs is None:
        obs = X.shape[0]
    indexes = np.random.permutation(range(X.shape[0]))[:obs]

    for iters in range(iterations):
        # shuffle the data
        indexes = np.random.permutation(indexes)
        m = 0
        
        for i in indexes:
            x = X[i].reshape(-1,1)
            y = labels[i]

            yhat = np.sign(np.dot(w.T,x))
            tau = max(0,(1-y*(np.dot(w.T,x)))/np.dot(x.T,x)**2)
            
            if yhat != y: #correct for perceptron, but need to consider pa
                m += 1
                
            w += tau*y*x
            
        mistakes.append((iters, m))
        
        if m == 0:
            break
        
    mistakes = pd.DataFrame(mistakes)
    mistakes.columns = ['iterations', 'passive aggressive mistakes']
    mistakes.set_index('iterations', inplace=True)
    
    w = w.reshape(w.shape[0],1)
    
    return w, mistakes


def pa_MC(X, labels, iterations=1, obs=None, w=None, mistakes=None):
    # X = np.append(X,np.ones([len(X),1]),1)
    if mistakes is None:
        mistakes = []
    if obs is None:
        obs = X.shape[0]
    indexes = np.random.permutation(range(X.shape[0]))[:obs]
    classes = np.unique(labels)
    if w is None:
        w = np.random.random(len(X[0]) * len(classes)).reshape(-1,1)

    for iters in range(iterations):
        # shuffle the data
        indexes = np.random.permutation(indexes)
        m = 0
        
        for i in indexes:
            x = X[i]#.reshape(-1,1)
            y = labels[i]
            
            # first calculate loss
            
            idx = np.argmax([np.dot(w.T, make_F(x, k, classes)) for k in classes])
            yhat = classes[idx]
            F_yhat = make_F(x, yhat, classes)
            F_y = make_F(x, y, classes)
            F_y_F_yhat = F_y - F_yhat
            
            
            
            if yhat != y: #correct for perceptron, but need to consider pa
                m += 1
                tau = max(0,(1-y*(np.dot(w.T, (F_y_F_yhat))))/np.dot((F_y_F_yhat).T, (F_y_F_yhat))**2)
                w += tau*(F_y_F_yhat)
                
            
        mistakes.append((iters, m))
        
        if m == 0:
            break
    
    mistakes = pd.DataFrame(mistakes)
    mistakes.columns = ['iterations', 'passive aggressive mistakes']
    mistakes.set_index('iterations', inplace=True)
    
    w = w.reshape(w.shape[0],1)
    
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