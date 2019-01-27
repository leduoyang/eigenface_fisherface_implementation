"""
 python3 hw2-2_lda.py $1 $2
• $1: path of whole dataset
• $2: path of the first 1 Fisherface
• E.g., python3 hw2-3_lda.py ./hw2/hw2-2_data ./output_Fisher.png
"""
import sys
import os
import numpy as np
from numpy import linalg as LA
import pandas as pd
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error
import pdb
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

###############################################################################
def getFileinfo(dir_path):
    df = pd.DataFrame(columns={"action","labels","paths"})
    count = 0
    for file in sorted(os.listdir(dir_path)):
        if dir_path[-1] == '/':
            filename = dir_path + file
        else:
            filename = dir_path + '/'+ file
        label = file.split('_')[0]
        num = file.split('_')[1]
        num = num.split('.')[0]
        #pdb.set_trace()
        if int(num) <= 7:
            df.loc[count] = pd.Series({'action':'train','labels': int(label), 'paths':filename})
        else:
            df.loc[count] = pd.Series({'action':'test','labels': int(label) , 'paths':filename})
            
        count = count + 1
        
    return df

def dataPreProcess(files_info):
    train_data , test_data= files_info[ files_info["action"] == "train" ] , files_info[ files_info["action"] == "test" ]
    
    print('read training data...')
    train_x , train_y = readImgs( train_data["paths"].values ) , np.array(train_data["labels"].values)
    
    print('read test data...')
    test_x , test_y = readImgs( test_data["paths"].values ) , np.array(test_data["labels"].values)
    
    return train_x , train_y , test_x , test_y

def readImgs(file_paths):
    h , w, ch= cv2.imread(file_paths[0]).shape
    X = np.zeros([len(file_paths),h,w,1])
    count = 0
    for path in file_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X[count,:,:,0] = img
        count = count + 1
    return X


###############################################################################
if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 2:
        raise Exception('please enter filename')
        
    #. Read Files
    folder_path = argv[1] #path of whole dataset
    Output_path = argv[2] #path of the first 1 Fisherface
    files_info = getFileinfo(folder_path)
    
    train_x , train_y , test_x , test_y = dataPreProcess(files_info)

    n, h, w, ch = train_x.shape
    height, width=h,w
    train_x = train_x.reshape((n,h*w*ch))

    ##########################################################################
    # Perform PCA and keep (N − C) eigenvectors
    N = n
    C = 40
       
    MEAN = np.mean(train_x, axis=0)
    
    train_x_normalized = train_x - MEAN
    train_x_normalized_t = np.transpose(train_x_normalized) # to fit np.cov
    cov = np.cov(train_x_normalized_t)
    
    e_value , e_vector = LA.eig(cov)
    e_vector = np.transpose(e_vector.real) # each row present an eigenvector 
    e_value = e_value.real
    
    # sort the eigenvectors in descent order
    indx_inorder = np.argsort(e_value)
    indx_inorder = indx_inorder[::-1]
    e_value = e_value[indx_inorder]
    e_vector = e_vector[indx_inorder]

    e_value = e_value[0:N-C]
    e_vector = e_vector[0:N-C]
    
    pca_e_vector = e_vector
    
    ##########################################################################
    # Project data onto these (N − C) eigenvector.   
    train_x_after_pca = np.transpose(np.dot(pca_e_vector ,np.transpose(train_x_normalized)))
    n , d= train_x_after_pca.shape

    ##########################################################################
    # Perform LDA and compute the (C − 1) projections in the lower-dimension (N-C) subspace.
    Sw = np.zeros((d,d))
    c_mean = np.zeros((C,d))

    #Sw
    for c in range(C):
        indices = np.where(train_y == c+1)
        data = train_x_after_pca[indices]
        u = np.mean(data, axis=0)
        #Sw = Sw + np.dot(np.transpose(data - u),(data - u ))
        c_mean[c,:]=u
        for i in range(data.shape[0]):
            x =  data[i,:].reshape((d,1))
            Sw = Sw  + np.dot((x - u.reshape((d,1))),np.transpose(x - u.reshape((d,1))))
    Sw_inverse = LA.inv(Sw)
    
    #SB
    U = np.mean(c_mean, axis=0)
    tem = U.shape
    U = U.reshape(tem[0],1)
    SB = np.zeros((d,d))
    for c in range(C):
        u = c_mean[c]
        tem = u.shape
        u = u.reshape(tem[0],1)
        SB = SB + np.dot((u - U),np.transpose(u - U))
        #pdb.set_trace() 

    e_value , e_vector = LA.eig(np.dot(Sw_inverse,SB))
    e_vector = np.transpose(e_vector.real)
    e_value = e_value.real
    
    indx_inorder = np.argsort(e_value)
    indx_inorder = indx_inorder[::-1]
    e_value = e_value[indx_inorder]
    e_vector = e_vector[indx_inorder]
    
    
    e_value = e_value[0:C-1]
    e_vector = e_vector[0:C-1]
    
    w = np.transpose(e_vector)
    
    ##########################################################################
    # Compute Fisherfaces.
    Fisherface = np.dot(np.transpose(pca_e_vector), w)
    Fisherface = np.transpose(Fisherface)
    first_5_ff = Fisherface[0:5]
    tem = first_5_ff[0,:]
    m , M = min(tem) , max(tem)
    tem = 255*((tem-m)/(M-m))
    tem = tem.reshape(height, width)
    cv2.imwrite(Output_path, tem)
    
    for i in range(5):
        img = first_5_ff[i]
    
        img = img.reshape(height, width)
        plt.figure(1)
        plt.imshow(img,cmap='gray')
        plt.show()    
    
    ##########################################################################    
    # Use t-SNE to visualize the distribution of the projected testing data, which has the dimension of 30.
    dim = 30
    project_vector = e_vector
    
    n, h, w, ch = test_x.shape
    test_x = test_x.reshape((n,h*w*ch))  
    test_x_normalized = test_x - MEAN
    test_x_after_pca = np.transpose(np.dot(pca_e_vector ,np.transpose(test_x_normalized)))
    
    features = np.transpose(np.dot(project_vector,np.transpose(test_x_after_pca)))
    
    features = TSNE(n_components=2,random_state=6).fit_transform(features)
    plt.figure(figsize=(12,5))
    plt.scatter(features[:,0], features[:,1], c=test_y)   
    plt.title("features after LDA")
    plt.show()    
    
    ############################LDA###########################################
    ## ground truth
    """
    IMGS = train_x
    LABELS = train_y.astype(np.int)
    clf = LinearDiscriminantAnalysis()
    clf.fit(IMGS, LABELS)
    IMGS_after_LDA = clf.transform(test_x)  
    IMGS_after_LDA = TSNE(n_components=2,random_state=6).fit_transform(IMGS_after_LDA)
        
    plt.figure(figsize=(12,5))
    plt.scatter(IMGS_after_LDA[:,0], IMGS_after_LDA[:,1], c=test_y)   
    plt.title("IMGS_after_LDA")
    plt.show()    
    """
    
    # To apply the k-nearest neighbors (k-NN) classifier to recognize the testing
    # set images, please determine the best k and n values by 3-fold crossvalidation.
    K = [1, 3, 5]
    N = [3, 10, 39]
    m_fold = 3
    X = train_x_after_pca
    Y = train_y.astype(np.int)
    kf = KFold(n_splits=m_fold,shuffle=True)
    ACC = []
    for k in K :
        for n in N :
            dim = n
            project_vector = e_vector[0:dim]
            features = np.transpose(np.dot(project_vector,np.transpose(X)))
            
            for train_index, test_index in kf.split(features):
                X_train, X_test = features[train_index], features[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                
                neigh = KNeighborsClassifier(n_neighbors=k,weights='distance')
                neigh.fit(X_train, y_train)
                pred = neigh.predict(X_test)
                score = accuracy_score(pred,y_test)
                ACC.append(score)
                print("ACC : " + str(score))
                
    ACC = np.array(ACC)
    ACC = ACC.reshape((1,-1))
    ACC = ACC.reshape((len(K)*len(N),m_fold))
    ACC = np.mean(ACC,axis=1)

    indx = np.argmax(ACC)
    k = K[2]
    n = N[2]
    dim = n
    
    project_vector = e_vector[0:dim]
    features = np.transpose(np.dot(project_vector,np.transpose(X)))
    neigh = KNeighborsClassifier(n_neighbors=k,weights='distance')
    neigh.fit(features, Y)
    
    features = np.transpose(np.dot(project_vector,np.transpose(test_x_after_pca)))
    pred = neigh.predict(features)
    score = accuracy_score(pred,test_y.astype(np.int))
    print("testing ACC : " + str(score))
    
    
    
    
    
    
    
    
    
    
    
    
    
    






