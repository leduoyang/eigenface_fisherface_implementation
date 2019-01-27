"""
python3 hw2-2_pca.py $1 $2 $3
• $1: path of whole dataset
• $2: path of the input testing image
• $3: path of the output testing image reconstruct by all eigenfaces
• E.g., python3 hw2-2_pca.py ./hw2/hw2-2_data
./hw2/test_image.png ./output_pca.png
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
            df.loc[count] = pd.Series({'action':'train','labels':label, 'paths':filename})
        else:
            df.loc[count] = pd.Series({'action':'test','labels':label, 'paths':filename})
            
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
    
def getEvector(x,mean): #perform PCA and return eigenvector
    x_normalized = x - mean
    x_normalized_t = np.transpose(x_normalized) # to fit np.cov
    cov = np.cov(x_normalized_t)    
    
    e_value , e_vector = LA.eig(cov)
    e_vector =np.transpose(e_vector.real) # each row present an eigenvector 
    e_value = e_value.real
    
    # sort the eigenvectors in descent order
    indx_inorder = np.argsort(e_value)
    indx_inorder = indx_inorder[::-1]
    e_value = e_value[indx_inorder]
    e_vector = e_vector[indx_inorder]
    #pdb.set_trace()
    return e_value,e_vector    

def project2PCA(data,project_vector):    
    p = np.transpose(np.dot(project_vector,np.transpose(data)))
    return p

def reconstructionIMG(p,project_vector,mean): # p : img in pca space
    tem , dim = project_vector.shape
    img = np.zeros((1,dim))
    img = p[0,0] * project_vector[0,:]
    n,tem = project_vector.shape
    for i in range(1,n):
        img = img + p[0,i] * project_vector[i,:]
    img = img+mean
    return img   

###############################################################################    
if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 2:
        raise Exception('please enter filename')
        
    #. Read Files
    folder_path = argv[1] #path of whole dataset
    testImg_path = argv[2] #path of the input testing image
    output_path = argv[3] #path of the output testing image reconstruct by all eigenfaces
    files_info = getFileinfo(folder_path)
    
    train_x , train_y , test_x , test_y = dataPreProcess(files_info)
    
    test_img = cv2.imread(testImg_path)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)    
        
    #. Perform PCA on the training data. Plot the mean face and the first five eigenfaces and show them in the report    
    n, h, w, ch = train_x.shape
    train_x = train_x.reshape((n,h*w*ch))
    mean = np.mean(train_x, axis=0)
    mean = mean.reshape((1,h*w*ch))
    e_value,e_vector = getEvector(train_x,mean)
    #pdb.set_trace()
    # show first five eigenfaces
    for i in range(5):
        v = e_vector[i,:]
        plt.figure(1)
        plt.imshow(v.reshape(h, w),cmap='gray')
        plt.title('eigenvector/eigenface NO.' + str(i+1))
        plt.show()
        pdb.set_trace()
    
    # Project image Person8_image6 onto the above PCA eigenspace. 
    # Reconstruct this image using the first n = { 5, 50, 150, all } eigenfaces. 
    # For each n, compute the mean square error (MSE) between the reconstructed face image and the original image
    # Plot these reconstructed images with the corresponding MSE values in the report.
    h, w = test_img.shape
    test_img = test_img.reshape((1,h*w))
    test_img_normalized = test_img - mean
    N = [ 5, 50, 150, h*w*ch ]
    for dim in N:
        project_vector = e_vector[0:dim]
        p = project2PCA(test_img_normalized,project_vector)
        img = reconstructionIMG(p,project_vector,mean)
        mse = mean_squared_error(test_img.reshape(h*w), img.reshape(h*w))  
        img = img.reshape((h, w))
        plt.figure(1)
        plt.imshow(img,cmap='gray')
        plt.title('dimension reduction to ' + str(n))
        plt.show()
        print('MSE : ' + str(mse))

    project_vector = e_vector[0:h*w*ch]
    p = project2PCA(test_img_normalized,project_vector)
    img = reconstructionIMG(p,project_vector,mean)
    img = img.reshape(h, w)
    cv2.imwrite(output_path, img)
       
    # Reduce the dimension of the image in testing set to dim = 100. Use tSNE to visualize the distribution of test images.
    dim = 100
    n, h, w, ch = test_x.shape
    test_x = test_x.reshape((n,h*w*ch))    
    test_x = test_x - mean
    
    project_vector = e_vector[0:100]
    features =project2PCA(test_x,project_vector)
    features = TSNE(n_components=2,random_state=1).fit_transform(features)
    plt.figure(figsize=(12,5))
    plt.scatter(features[:,0], features[:,1], c=test_y)   
    plt.title("features after PCA")
    plt.show()    
    
    ## ground truth
    """
    pca = PCA(n_components=100,svd_solver='full')
    pca.fit(train_x)
    IMGS_after_PCA = pca.transform(test_x)
    IMGS_after_PCA = TSNE(n_components=2,random_state=1).fit_transform(IMGS_after_PCA)
    plt.figure(figsize=(12,5))
    plt.scatter(IMGS_after_PCA[:,0], IMGS_after_PCA[:,1], c=test_y)   
    plt.title("IMGS_after_PCA")
    plt.show()     
    """
    
    # To apply the k-nearest neighbors (k-NN) classifier to recognize the testing
    # set images, please determine the best k and n values by 3-fold crossvalidation.
    K = [1, 3, 5]
    N = [3, 10, 39]
    m_fold = 3
    X = train_x - mean
    Y = train_y
    kf = KFold(n_splits=m_fold,shuffle=True)
    ACC = []
    for k in K :
        for n in N :
            dim = n
            project_vector = e_vector[0:dim]
            features =project2PCA(X,project_vector)
            
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
    k = K[indx // 3]
    n = N[indx % 3]
    dim = n
    
    project_vector = e_vector[0:dim]
    features =project2PCA(X,project_vector)
    neigh = KNeighborsClassifier(n_neighbors=k,weights='distance')
    neigh.fit(features, Y)
    
    features =project2PCA(test_x,project_vector)
    pred = neigh.predict(features)
    score = accuracy_score(pred,test_y)
    print("testing ACC : " + str(score))


































