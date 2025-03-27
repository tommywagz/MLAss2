from sklearn import datasets as ds
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# part a
def genTreeClassifier(seed):
    X, y = ds.make_classification(n_samples=1000, n_features=20,
                                    n_informative=5, n_redundant=15,
                                    shuffle=False, random_state=seed)

    scaler = StandardScaler()
    newX = scaler.fit_transform(X)

    shuffledIdxs = np.random.default_rng(seed=0).permutation(newX.shape[0])
    shuffledX = newX[shuffledIdxs]
    shuffledy = y[shuffledIdxs]

    dataClassifier = DecisionTreeClassifier(criterion="entropy", random_state=4)
    dataClassifier.fit(shuffledX, shuffledy)

    featImports = dataClassifier.feature_importances_
    
    return featImports, shuffledX
    

# part c
def repeatDT():
    totalFound = []
    for i in range(1, 1000+1):
        featureImportances, X = genTreeClassifier(i)
        top5Indices = np.argsort(featureImportances)[-5:]
        informativeIndices = np.arange(5)
        
        # find the number of important indices found within the top 5
        found = 0
        for idx in top5Indices:
            if idx in informativeIndices:
                found += 1
                
        totalFound.append(found)
        
    avg = np.mean(totalFound)
    print(f"On average the model identified {avg} important features in the top five for every run")

    # histogram of numbers 0 to 5 on x axis 
    # number of times x number of important features was found
    plt.figure(figsize=(10, 6))
    plt.hist(totalFound, bins=np.arange(0, 6) - 0.5, rwidth=0.8)
    plt.xticks(range(6))
    plt.xlabel("Number of Important Features in Top 5")
    plt.ylabel("Frequency")
    plt.title("Distribution of Important Features Found")
    plt.show()


# part d
def genLogReg(scale):
    totalFound = []
    
    for i in range(1, 1000+1):
        X, y = ds.make_classification(n_samples=1000, n_features=20, n_informative=5,
                                      n_redundant=15, shuffle=False, random_state=i)

        if scale:
            scaler = StandardScaler()
            newX = scaler.fit_transform(X)
        else:
            newX = X
            
        logRegClassi = LogisticRegression(penalty=None, random_state=0)
        logRegClassi.fit(newX, y)
        
        featureImportances = np.abs(logRegClassi.coef_[0])
        
        top5Indices = np.argsort(featureImportances)[-5:]
        informativeIndices = np.arange(5)
        
        found = 0
        for idx in top5Indices:
            if idx in informativeIndices:
                found += 1
                
        totalFound.append(found)
        
    avg = np.mean(totalFound)
    print(f"On average the model identified {avg} important features in the top five for every run")
        
    plt.figure(figsize=(10, 6))
    plt.hist(totalFound, bins=np.arange(0, 6) - 0.5, rwidth=0.8)
    plt.xticks(range(6))
    plt.xlabel("Number of Informative Features in Top 5")
    plt.ylabel("Frequency")
    plt.title("Logistic Regression Informative Feature Count")   
    plt.show() 


# part f
def findOverlap():
    totalFoundDT = []
    totalFoundLR = []
    totalOverlaps = []
    for i in range(1, 1000+1):
        X, y = ds.make_classification(n_samples=1000, n_features=20, n_informative=5,
                                      n_redundant=15, shuffle=False, random_state=i)
        
        scaler = StandardScaler()
        newX = scaler.fit_transform(X)
        
        shuffledIdxs = np.random.default_rng(seed=0).permutation(newX.shape[0])
        shuffledX = newX[shuffledIdxs]
        shuffledy = y[shuffledIdxs]
        
        logRegClassi = LogisticRegression(penalty=None, random_state=0)
        logRegClassi.fit(shuffledX, shuffledy)
        
        dataClassifier = DecisionTreeClassifier(criterion="entropy", random_state=4)
        dataClassifier.fit(shuffledX, shuffledy)
        
        LRfeatures = np.abs(logRegClassi.coef_[0])
        DTfeatures = dataClassifier.feature_importances_
        
        informativeIndices = np.arange(5)
        top5LR = np.argsort(LRfeatures)[-5:]
        top5DT = np.argsort(DTfeatures)[-5:]
        
        foundLR = 0
        foundDT = 0
        overlap = 0
        
        for i in top5LR:
            if i in informativeIndices:
                foundLR += 1
        totalFoundLR.append(foundLR)
                
        for j in top5DT:
            if j in informativeIndices:
                foundDT += 1
            if j in totalFoundLR:
                overlap += 1
        totalFoundDT.append(foundDT)
                
        totalOverlaps.append(overlap)
        
    avg = np.mean(totalOverlaps)
    print(f"On average the models identified {avg} of the same important features in the top five for every run")

    plt.figure(figsize=(10, 6))
    plt.hist(totalOverlaps, bins=np.arange(0, 6) - 0.5, rwidth=0.8)
    plt.xticks(range(6))
    plt.xlabel("Number of Informative Features in both Top 5s")
    plt.ylabel("Frequency")
    plt.title("Logistic Regression and Decision Tree Informative Feature Count")   
    plt.show() 
        

if __name__ == "__main__":
    # featImports, X = genTreeClassifier(0)
    
    # top5Indices = np.argsort(featImports)[-5:]
    # informativeIndices = np.arange(5)

    # found = 0
    # for idx in top5Indices:
    #     if idx in informativeIndices:
    #         found += 1
            
    # print(f"Number of informative features in top 5: {found}")
            
    # sortedIndices = np.argsort(featImports)[::-1]
    # sortedImportances = featImports[sortedIndices]

    # plt.figure(figsize=(10, 6))
    # plt.bar(range(X.shape[1]), sortedImportances)
    # plt.xticks(range(X.shape[1]), sortedIndices)
    # plt.xlabel("Features by Importance")
    # plt.ylabel("Feature Importance Score")
    # plt.title("Feature Importance from Decision Tree")
    # plt.show()
    
    # repeatDT()
    genLogReg(True)
    genLogReg(False)
    findOverlap()