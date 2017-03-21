# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:50:48 2017

@author: 521-hui
"""
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt 
import numpy as np

if __name__ == "__main__":
    

    fpr, tpr, thresholds = roc_curve(testlabel, score_svm[0])
    roc_auc = auc(fpr, tpr)
    print 'AUC:', roc_auc
    plt.plot(fpr, tpr, 'b-', label = 'SVM')
    
    fpr, tpr, thresholds = roc_curve(testlabel, score_rf[0])
    roc_auc = auc(fpr, tpr)
    print 'AUC:', roc_auc
    plt.plot(fpr, tpr, 'g-.', label= 'random forest')
    
    score_all = (np.array(score_svm[0]) + np.array(score_rf[0]))*0.5
    fpr, tpr, thresholds = roc_curve(testlabel, score_all)
    roc_auc = auc(fpr, tpr)
    print 'AUC:', roc_auc
    plt.plot(fpr, tpr, 'r', label = 'ensemble')
    
    plt.plot([1,0],[0,1], 'k-.')

    
    plt.ylabel('FPR')
    plt.xlabel('TPR')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    
    plt.savefig('ROC.png', dpi =1000)