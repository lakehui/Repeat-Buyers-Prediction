# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:09:06 2016

@author: 521-hui
"""

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pickle
from collections import Counter
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_auc_score
from sknn import mlp
import csv
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt 

from test_hh import Sample

def sortSampCate(sample):
    Psamp = []
    Nsamp = []
    for row in sample:
        tmprow = []
        for i in range(10):
            x = row.feature[i]
            if x == '':
                tmprow.append(0)
            else:
                tmprow.append(x)
            
        if row.label == 1:
            Psamp.append(tmprow)
        else:
            Nsamp.append(tmprow)
            
    return Psamp, Nsamp
    
def sortClass(sample, label, select):
    Psamp = []
    Nsamp = []
    dict_select = Counter(select)
    
    for i in range(len(sample)):  
        row = []
        for j in range(len(sample[i])):
            if not dict_select.has_key(j):
                row.append(sample[i][j])
                
        if label[i] == 1:
            Psamp.append(row)
        else:
            Nsamp.append(row)
            
    return Psamp, Nsamp
    

def genRandomSamp(feature, label):
    
    POSsamp, NEGsamp = sortClass(feature, label, [])
    
    index_random = np.random.permutation(len(POSsamp)).tolist()   
    new_samp_p = [POSsamp[x] for x in index_random]
    
    index_random = np.random.permutation(len(NEGsamp)).tolist()   
    new_samp_n = [NEGsamp[x] for x in index_random]
    
    new_train_p = [new_samp_p[i] for i in range(len(new_samp_p)) if i < 8000]
    new_train_n = [new_samp_n[i] for i in range(len(new_samp_n)) if i < 8500]
    
    new_test_p = [new_samp_p[i] for i in range(len(new_samp_p)) if i >= 8000]
    new_test_n = [new_samp_n[i] for i in range(len(new_samp_n)) if i >= 8500]
    
    return new_train_p, new_train_n, new_test_p, new_test_n
    
def getTrainSet(samp_P, samp_N):
    
    num_samp = int(len(samp_P)*3 / 4)
    
    index = np.random.permutation(len(samp_P))[:num_samp].tolist()    
    train_P = [samp_P[x] for x in index]
    
    index = np.random.permutation(len(samp_N))[60000:60000+num_samp].tolist()    
    train_N = [samp_N[x] for x in index]
    
    return train_P, train_N

def RfModelDecision(train_x, label, Ptest_x, Ntest_x):
    
    print 'training random forest model...'
    rf_model= RandomForestClassifier(n_estimators = 60)
    rf_model.fit(train_x, label)
    
    Ppred_rf = rf_model.predict_proba(Ptest_x)
    Npred_rf = rf_model.predict_proba(Ntest_x)

    test_label = [1]*len(Ppred_rf) + [0]*len(Npred_rf)
    score = Ppred_rf.tolist() + Npred_rf.tolist()
    score_rf = [x[1] for x in score]
    auc = roc_auc_score(test_label,score_rf)
    print 'rf_AUC:', auc
    
    return score_rf, rf_model
    
def SVMModelDecision(train_x, label, Ptest_x, Ntest_x):
    
    print 'training svm model...'
    svm_model= svm.SVC(C=35, gamma=0.005, kernel='rbf', probability=True)
    svm_model.fit(train_x, label)
    
    Ppred_svm = svm_model.predict_proba(Ptest_x)
    Npred_svm = svm_model.predict_proba(Ntest_x)

    test_label = [1]*len(Ppred_svm) + [0]*len(Npred_svm)
    score = Ppred_svm.tolist() + Npred_svm.tolist()
    score_svm = [x[1] for x in score]
    auc = roc_auc_score(test_label,score_svm)
    print 'svm_AUC:', auc
    
    return score_svm, svm_model
    

def ROC_Plot(predict_vec, label):
    fpr, tpr, thresholds = roc_curve(label, predict_vec)
    roc_auc = auc(fpr, tpr)
    print 'AUC:', roc_auc
    plt.plot(fpr, tpr, lw=1)  


if __name__ == "__main__":
    
    operation = "train"
    
    if operation == "train":
        feature_all = pickle.load(open("../data/feature_train/feat_final.pytmp"))
        label = pickle.load(open("../data/feature_train/label.pytmp"))
        
        open_scaler = open("../data/model/scaler.pytmp", 'r')
        scaler = pickle.load(open_scaler)
        open_scaler.close()
        
        feat_scale = scaler.transform(feature_all)
        
        feat_scale = SelectKBest(chi2, k=23).fit_transform(feat_scale, label)
    #    select_model = SelectKBest(f_classif, k=15).fit(feature_all, label)
    #    feature_all_new = select_model.transform(feature_all)
        '''
        scaler = preprocessing.StandardScaler().fit(feature_all)
        scaler_file = open("../data/model/scaler.pytmp", 'wb')
        pickle.dump(scaler, scaler_file)
        scaler_file.close()
        
        train_set_P, train_set_N, test_x_P, test_x_N = genRandomSamp(feature_all, label)
        
        train_P, train_N = getTrainSet(train_set_P, train_set_N)
        
        train_x_org =  train_P + train_N
        train_y = [1]*len(train_P) + [0]*len(train_N)
        
        # use function preprocessing.scale to standardize X  
        train_x = scaler.transform(train_x_org)
        Ptest_x = scaler.transform(test_x_P)
        Ntest_x = scaler.transform(test_x_N)
        '''
        
    #    min_max_scaler = preprocessing.MinMaxScaler()  
    #    feat_minMax = min_max_scaler.fit_transform(feat)
    #    X_old = np.array(feat)
        
        # 训练svm
        #test_label = [1]*len(Ptest_x) + [0]*len(Ntest_x)
        #score_total_svm = np.array([0]*len(test_label))
        svm_models = []
        score_svm = []
        for i in range(5):
            train_P, train_N, test_P, test_N = genRandomSamp(feat_scale, label)
            #train_P, train_N = getTrainSet(train_set_P, train_set_N)
            train_x =  train_P + train_N
            train_y = [1]*len(train_P) + [0]*len(train_N)
            #train_x = scaler.transform(train_x_org)
            
            tmp_score, svm_model = SVMModelDecision(train_x, train_y, test_P, test_N)
            score_svm.append(tmp_score)
            #score_total_svm = score_total_svm + np.array(tmp_score)/float(5)
            
            
            #check training set accuracy
            sum_Ptrain = sum(svm_model.predict(train_P))
            sum_Ntrain = len(train_N) - sum(svm_model.predict(train_N))
            print 'training set acc:', sum_Ptrain, sum_Ntrain
            
            svm_models.append(svm_model)
        
        

        '''
        open_file = open("../data/model/svm_model_files.pytmp", 'r')
        svm_models = pickle.load(open_file)
        open_file.close
        
        open_file = open("../data/model/rf_model_files.pytmp", 'r')
        rf_models = pickle.load(open_file)
        open_file.close
        
        score_svm = []
        for svm_model in svm_models:
            predictions = svm_model.predict_proba(test_feat)
            tmp_score = [x[1] for x in predictions]
            score_svm.append(tmp_score)
            ROC_Plot(tmp_score, label)
        
        score_rf = []
        for rf_model in rf_models:
            predictions = rf_model.predict_proba(test_feat)
            tmp_score = [x[1] for x in predictions]
            score_rf.append(tmp_score)
            ROC_Plot(tmp_score, label)
        '''
            
        
        
        # 训练random forest
        #test_label = [1]*len(Ptest_x) + [0]*len(Ntest_x)
        #score_total_rf = np.array([0]*len(test_label))
        rf_models = []
        score_rf = []
        for i in range(5):
            train_P, train_N, test_P, test_N = genRandomSamp(feat_scale, label)
            #train_P, train_N = getTrainSet(train_set_P, train_set_N)
            train_x =  train_P + train_N
            train_y = [1]*len(train_P) + [0]*len(train_N)
            #train_x = scaler.transform(train_x_org)
            
            tmp_score, rf_model = RfModelDecision(train_x, train_y, test_P, test_N)
            score_rf.append(tmp_score)
            #score_total_rf = score_total_rf + np.array(score_rf)/float(5)      
            
            #check training set accuracy
            sum_Ptrain = sum(rf_model.predict(train_P))
            sum_Ntrain = len(train_N) - sum(rf_model.predict(train_N))
            print 'training set acc:', sum_Ptrain, sum_Ntrain
            
            rf_models.append(rf_model)
        
        #auc = roc_auc_score(test_label,score_total_rf)
        #print 'total_rf_AUC:', auc
        
        # plot ROC curve
        
        
        '''
        score_sum=[]
        for i in range(len(score_rf)):
            score_sum.append((score_total_rf[i] + score_total_svm[i])/2)
        auc = roc_auc_score(test_label,score_sum)
        print 'total_AUC:', auc
        
        save_file = open("../data/model/svm_model_files.pytmp", 'wb')
        pickle.dump(svm_models, save_file)
        save_file.close()
        
        save_file = open("../data/model/rf_model_files.pytmp", 'wb')
        pickle.dump(rf_models, save_file)
        save_file.close()
        '''
        
        '''
        # 训练SVM模型       
        print 'training svm model...'
        svm_model = svm.SVC(C=35, gamma=0.005, kernel='rbf', probability=True)
    
        svm_model.fit(train_x, train_y)
        Ppred_svm = svm_model.predict_proba(Ptest_x)
        Npred_svm = svm_model.predict_proba(Ntest_x)
    
        test_label = [1]*len(Ppred_svm) + [0]*len(Npred_svm)
        score = Ppred_svm.tolist() + Npred_svm.tolist()
        score_svm = [x[1] for x in score]
        auc = roc_auc_score(test_label,score_svm)
        print 'svm_AUC:', auc
        
        #check training set accuracy
        sum_Ptrain = sum(svm_model.predict(train_P))
        sum_Ntrain = len(train_N) - sum(svm_model.predict(train_N))
        print 'training set acc:', sum_Ptrain, sum_Ntrain
        '''
        
        '''
        #train MLP classifier
        print 'training mlp classifir'
        layers = [mlp.Layer('Rectifier', units=1000, weight_decay=0.0005, dropout=0.3, normalize=True),\
                  mlp.Layer('Rectifier', units=50, weight_decay=0.0005, dropout=0.3, normalize=True),\
                  mlp.Layer('Softmax')]
        
        mlp_model = mlp.Classifier(layers, batch_size=40, valid_size=0.01,loss_type='mcc')
        
        mlp_model.fit(np.array(train_x), np.array(train_y))
        
        Ppred_mlp = mlp_model.predict_proba(np.array(Ptest_x))
        Npred_mlp = mlp_model.predict_proba(np.array(Ntest_x))
        
        test_label = [1]*len(Ppred_mlp) + [0]*len(Npred_mlp)
        score = Ppred_mlp.tolist() + Npred_mlp.tolist()
        score_mlp = [x[1] for x in score]
        auc = roc_auc_score(test_label,score_mlp)
        print 'mlp_AUC:', auc
        
        sum_Ptrain = mlp_model.predict(np.array(train_P))
        sum_Ntrain = mlp_model.predict(np.array(train_N))
        print 'training set acc:', sum_Ptrain.sum(), sum_Ntrain.sum()
        '''
    
#===================================predict test sample=====================================#
    if operation == "test":    
        test_file = open("../data/feature/feat_final.pytmp")
        test_feat_org = pickle.load(test_file)
        test_file.close()
        
        open_scaler = open("../data/model/scaler.pytmp", 'r')
        scaler = pickle.load(open_scaler)
        open_scaler.close()
        
        test_feat = scaler.transform(test_feat_org)
        
        open_file = open("../data/model/svm_model_files.pytmp", 'r')
        svm_models = pickle.load(open_file)
        open_file.close
        
        open_file = open("../data/model/rf_model_files.pytmp", 'r')
        rf_models = pickle.load(open_file)
        open_file.close
        
        print "svm predict..."
        score_total_svm = np.array([[0,0]]*len(test_feat))
        for svm_model in svm_models:
            score_tmp = svm_model.predict_proba(test_feat)
            score_total_svm = score_total_svm + score_tmp/float(10)
        
        print "random forest predict..."
        score_total_rf = np.array([[0,0]]*len(test_feat))
        for rf_model in rf_models:
            score_tmp = rf_model.predict_proba(test_feat)
            score_total_rf = score_total_rf + score_tmp/float(10)
        
        score_tmp = (score_total_svm +score_total_rf)/2
        score_final = [x[1] for x in score_tmp]
        
        #write result score into submit file
        test_file = open("../data/test_format1.csv", 'r')
        test_reader = csv.reader(test_file)
        
        result_file = open("../data/submit_result_allsample.csv", 'wb')
        result_writer = csv.writer(result_file)
        
        result_writer.writerow(test_reader.next())
        i = 0
        for row in test_reader:
            row[2] = str(score_final[i])
            result_writer.writerow(row)
            i = i + 1
        
        test_file.close()
        result_file.close()
        
        