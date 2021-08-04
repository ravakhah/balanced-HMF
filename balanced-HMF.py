# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 12:51:28 2020

@author: Mehdi
"""
import time
import numpy as np
import scipy.io
from sklearn.svm import LinearSVC
from sklearn.svm import OneClassSVM
no_item = 0
no_user = 0
maes = np.zeros(10)
rmses = np.zeros(10)
C1 = 0.02
C2 = 0.02
K = 10
t0 = time.time()
for fold_id in range(1):
    print('fold number = ', fold_id)
    Y = np.loadtxt("..\ml_1m_10_fold_dataset/train" + str(fold_id+1) + ".txt", delimiter=',')    
    n = Y.shape[0] # number of users
    m = Y.shape[1]  # number of items    
    Test_rates = np.loadtxt("..\ml_1m_10_fold_dataset/test" + str(fold_id+1) + ".txt", delimiter=',')    
    tests = np.where(Test_rates>0)
    Final_Predict = np.zeros((n,m))
    
    U = np.random.random((n,K))
    V = np.random.random((m,K))
    U_biases = np.zeros(n)
    V_biases = np.zeros(m)
 
    step = 1
    while (step < 5):
        if (step == 1):
            current_rate = 3
        
        print('current_rate = ', current_rate)
        Y1 = np.copy(Y)
        if (current_rate == 3):
            Y1[Y1==1] = -1
            Y1[Y1==2] = -1
            Y1[Y1==3] = -1
            Y1[Y1>3] = 1
        
        elif (current_rate == 2):
            Y1[Y1==1] = -1
            Y1[Y1==2] = -1
            Y1[Y1==3] = 1
            Y1[Y1>3] = 0
        
        elif (current_rate == 1):
            Y1[Y1==1] = -1
            Y1[Y1==2] = 1
            Y1[Y1>2] = 0
            
        elif (current_rate == 4):
            Y1[Y1<4] = 0
            Y1[Y1==4] = -1
            Y1[Y1==5] = 1
        #####################prepare itemlist and ratelist for users
        items_for_users = []
        rates_for_users = []       
        for i in range(n):           
            hh = np.asarray(np.where(Y1[i,:] != 0))
            items_user_i = list(hh.flatten())
            hh2 = Y1[i,items_user_i]
            rates_user_i = list(hh2.flatten())                       
            items_for_users.append(items_user_i)
            rates_for_users.append(rates_user_i)
        #############################end of prepare itemlist and ratelist for users  
       
        #####################prepare userlist and ratelist for items
        users_for_items = []
        rates_for_items = []      
        for j in range(m):          
            hh = np.asarray(np.where(Y1[:,j] != 0))
            users_item_j = list(hh.flatten())
            hh2 = Y1[users_item_j,j]
            rates_item_j = list(hh2.flatten())                      
            users_for_items.append(users_item_j)
            rates_for_items.append(rates_item_j)
        #############################end of prepare userlist and ratelist for items
        f = 0
        while (True):
            print('step number: ',f)
            for ucnt in range(n):
                #print(ucnt)
                Vindexes = items_for_users[ucnt]              
                r = rates_for_users[ucnt]
                v = V[Vindexes]
                
                num_items = len(r) 
                if (num_items == 0):
                    no_item += 1
                    continue
                
                if (all(elem == 1 for elem in r) or all(elem == -1 for elem in r)): 
                    clf = OneClassSVM(kernel = 'linear')
                else:
                    if (num_items > K):
                        clf = LinearSVC(dual = False, C = C1)
                    else:
                        clf = LinearSVC(C = C1, loss='hinge')
                clf.fit(v, r)
                U[ucnt] = clf.coef_
                
                U_biases[ucnt] = clf.intercept_

            for vcnt in range(m):
                Uindexes = users_for_items[vcnt]              
                r = rates_for_items[vcnt]
                u = U[Uindexes]
                
                num_users = len(r)
                if (num_users == 0):
                    no_user +=1
                    continue
                
                if (all(elem == 1 for elem in r) or all(elem == -1 for elem in r)): 
                    clf = OneClassSVM(kernel = 'linear')
                else:
                    if (num_users > K):
                        clf = LinearSVC(dual = False, C = C2)
                    else:
                        clf = LinearSVC(C = C2, loss='hinge')
                clf.fit(u, r)
                V[vcnt] = clf.coef_
                
                V_biases[vcnt] = clf.intercept_
            f += 1
            if (f>4):
                break
        
        if (current_rate == 3):
            U3 = np.copy(U)
            V3 = np.copy(V)
            U_biases3 = np.copy(U_biases)
            V_biases3 = np.copy(V_biases)
            current_rate = 2
        elif (current_rate == 2):
            U2 = np.copy(U)
            V2 = np.copy(V)
            U_biases2 = np.copy(U_biases)
            V_biases2 = np.copy(V_biases)
            current_rate = 1
        elif (current_rate == 1):
            U1 = np.copy(U)
            V1 = np.copy(V)
            U_biases1 = np.copy(U_biases)
            V_biases1 = np.copy(V_biases)
            current_rate = 4
        elif (current_rate == 4):
            U4 = np.copy(U)
            V4 = np.copy(V)
            U_biases4 = np.copy(U_biases)
            V_biases4 = np.copy(V_biases)
        step += 1
    
    t1 = time.time()  
    total_train_time = t1 - t0
    print('total_train_time = ', total_train_time)
    
    t0 = time.time()
    for i in range(tests[0].size):
        if (np.dot(U3[tests[0][i]], V3[tests[1][i]]) + (U_biases3[tests[0][i]] + V_biases3[tests[1][i]])/2 <= 0):
            if (np.dot(U2[tests[0][i]], V2[tests[1][i]]) + (U_biases2[tests[0][i]] + V_biases2[tests[1][i]])/2 > 0):
                predicted_rate = 3
            elif (np.dot(U1[tests[0][i]], V1[tests[1][i]]) + (U_biases1[tests[0][i]] + V_biases1[tests[1][i]])/2 <= 0):
                predicted_rate = 1
            else:
                predicted_rate = 2
        elif (np.dot(U4[tests[0][i]], V4[tests[1][i]]) + (U_biases4[tests[0][i]] + V_biases4[tests[1][i]])/2 <= 0):
            predicted_rate = 4
        else:
            predicted_rate = 5
        Final_Predict[tests[0][i], tests[1][i]] = predicted_rate
    t1 = time.time()       
    total_test_time = t1 - t0
    print('total_test_time = ', total_test_time)
           
    num_predictions = np.count_nonzero(Test_rates)
    E2Test = (Final_Predict - Test_rates)**2
    rmse = np.sqrt(np.sum(E2Test)/num_predictions)
    EabsTest = np.abs(Final_Predict - Test_rates)
    mae = np.sum(EabsTest)/num_predictions
    print('rmse = ', rmse, '     mae = ', mae)
    False_Predicts = Test_rates - Final_Predict
    num_fp = np.count_nonzero(False_Predicts)
    num_tp = num_predictions - num_fp
    print('false predict = ', num_fp , '     percent = ', num_fp/num_predictions)
    print('true predict = ', num_tp , '     percent = ', num_tp/num_predictions)  
    maes[fold_id] = mae
    rmses[fold_id] = rmse
    
print('mean mae = ', maes.mean())
print('mean rmses = ', rmses.mean())


              
                    


