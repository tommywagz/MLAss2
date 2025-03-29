from sklearn import datasets as ds
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from sklearn.inspection import permutation_importance as pi
from sklearn.model_selection import train_test_split

############################ Question 1 ###############################

# part a
def genTreeClassifier(seed, scale):
    X, y = ds.make_classification(n_samples=1000, n_features=20,
                                    n_informative=5, n_redundant=15,
                                    shuffle=False, random_state=seed)
    if scale:
        scaler = StandardScaler()
        newX = scaler.fit_transform(X)
    else:
        newX = X

    shuffledIdxs = np.random.default_rng(seed=0).permutation(newX.shape[0])
    shuffledX = newX[shuffledIdxs]
    shuffledy = y[shuffledIdxs]

    dataClassifier = DecisionTreeClassifier(criterion="entropy", random_state=4)
    dataClassifier.fit(shuffledX, shuffledy)

    featImports = dataClassifier.feature_importances_
    
    return featImports, shuffledX

# part c/e
def repeatDT(scale):
    totalFound = []
    for i in range(1, 1000+1):
        featureImportances, X = genTreeClassifier(i, scale)
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
    plt.figure(figsize=(10, 7))
    plt.hist(totalFound, bins=np.arange(0, 7) - 0.5, rwidth=0.8)
    plt.xticks(range(7))
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
        
    plt.figure(figsize=(10, 7))
    plt.hist(totalFound, bins=np.arange(0, 7) - 0.5, rwidth=0.8)
    plt.xticks(range(7))
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

    plt.figure(figsize=(10, 7))
    plt.hist(totalOverlaps, bins=np.arange(0, 7) - 0.5, rwidth=0.8)
    plt.xticks(range(7))
    plt.xlabel("Number of Informative Features in both Top 5s")
    plt.ylabel("Frequency")
    plt.title("Logistic Regression and Decision Tree Informative Feature Count")   
    plt.show() 
        
        
        
############################### Question 2 ####################################3   
        
# part b
def backwardSelectionLR(seed):
    X, y = ds.make_classification(n_samples=1000, n_features=20,
                                n_informative=5, n_redundant=15,
                                shuffle=False, random_state=seed)

    scaler = StandardScaler()
    newX = scaler.fit_transform(X)

    shuffledIdxs = np.random.default_rng(seed=0).permutation(newX.shape[0])
    shuffledX = newX[shuffledIdxs]
    shuffledy = y[shuffledIdxs]
    remainingFeatures = list(range(X.shape[1]))
    
    # find the least helpful feature and remove it from feature matrix
    while len(remainingFeatures) > 5:    
        logRegClassi = LogisticRegression(penalty=None, random_state=4)
        logRegClassi.fit(shuffledX, shuffledy)
        featureImportances = np.argsort(np.abs(logRegClassi.coef_[0]))
        
        worstFeatIdx = remainingFeatures[featureImportances[0]]
        worstFeatIdxRemaining = remainingFeatures.index(worstFeatIdx)
        
        # remove worst feature from remaining and X
        remainingFeatures.pop(worstFeatIdxRemaining)
        shuffledX = np.delete(shuffledX, worstFeatIdxRemaining, axis=1)
        
    score = 0
    for f in remainingFeatures:
        if f in range(5):
            score += 1

    return remainingFeatures, score

        
# part c
def repeatBSLR():
    scores = []
    remainingFeatures = []
    
    for i in range(1, 1000+1):
        currRemaining, currScore = backwardSelectionLR(i)
        scores.append(currScore)
        remainingFeatures.append(currRemaining)

    avg = np.mean(scores) 
    print(f"There were an average of {avg} important features found with backward elimination")

    plt.figure(figsize=(10, 7))
    plt.hist(scores, bins=np.arange(0, 7) - 0.5, rwidth=0.8)
    plt.xticks(range(0, 7))
    plt.xlabel("Number of Important Featuress Present Post-Elimination")
    plt.ylabel("Frequency")
    plt.title("Number of Important Features Recovered per 1000 Runs")
    plt.show()


# part e
def subsetSelection():
    X, y = ds.make_classification(n_samples=1000, n_features=7,
                                n_informative=3, n_redundant=4,
                                shuffle=False, random_state=0)

    scaler = StandardScaler()
    newX = scaler.fit_transform(X)

    shuffledIdxs = np.random.default_rng(seed=0).permutation(newX.shape[0])
    shuffledX = newX[shuffledIdxs]
    shuffledy = y[shuffledIdxs]
    
    bestScore = -np.inf 
    bestSubset = None
    
    recoveries = []
    scores=[]
    for i in range(1, 8):
        for subset in combinations(range(shuffledX.shape[1]), i):
            subsetX = shuffledX[:, list(subset)]
            logRegClassi = LogisticRegression(penalty=None, random_state=4)
            logRegClassi.fit(shuffledX, shuffledy)
            featureImportances = np.abs(logRegClassi.coef_[0])
            subsetScore = np.mean(featureImportances)
            scores.append(subsetScore)
            
            if subsetScore > bestScore:
                bestScore = subsetScore
                bestSubset = subsetX
                
            currCount = 0
            for f in subset:
                if f in range(3):
                    currCount += 1
            recoveries.append(currCount)
    
    avg = np.mean(recoveries)
    print(f"The average number of recoveries is {avg}")
    print(f"The best score for subset Selection was {bestScore}")
    
    plt.figure(figsize=(10, 5))
    plt.hist(x = recoveries, bins=np.arange(0, 5) - 0.5, rwidth=0.8)
    plt.xticks(range(0, 5))
    plt.xlabel("Number of Recovered Informative Features")
    plt.ylabel("Frequency")
    plt.title("Recovered Important Features by Each Subset")
    plt.show()

    return list(bestSubset), bestScore


# part f
def permFeatImport():
    scores = []
    for i in range(1 + 1000+1):
        X, y = ds.make_classification(n_samples=1000, n_features=20,
                                    n_informative=5, n_redundant=15,
                                    shuffle=False, random_state=i)

        scaler = StandardScaler()
        newX = scaler.fit_transform(X)

        shuffledIdxs = np.random.default_rng(seed=0).permutation(newX.shape[0])
        shuffledX = newX[shuffledIdxs]
        shuffledy = y[shuffledIdxs]
        
        logRegClassi = LogisticRegression(penalty=None, random_state=4, solver='lbfgs')
        logRegClassi.fit(shuffledX, shuffledy)
        
        result = pi(logRegClassi, shuffledX, shuffledy, n_repeats=10, random_state=4)
        importances = result.importances_mean
        
        top5Idx = np.argsort(importances)[-5:]
        
        informativeIndices = list(range(6))

        score = 0
        for f in top5Idx:
            if f in informativeIndices:
                score += 1

        scores.append(score)
        
    avg = np.mean(scores)
    print(f"There was an average of {avg} important features with Permutation Importance")
    
    plt.figure(figsize=(10, 7))
    plt.hist(x=scores, bins=np.arange(0, 7) - 0.5, rwidth=0.8)
    plt.xticks(range(0, 7))
    plt.xlabel("Number of Important Features Found")
    plt.ylabel("Frequency")
    plt.title("Number of Important Features Recovered per 1000 Runs with Permutation Importance")
    plt.show()

#######################################################################

if __name__ == "__main__":
    # featImports, X = genTreeClassifier(0, True)
    
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
    
    # repeatDT(True)
    # repeatDT(False)
    # genLogReg(True)
    # genLogReg(False)
    # findOverlap()
    
    # remaining, score = backwardSelectionLR(0)
    # print(f"There were {score} remaing features that were important:")
    # for f in remaining:
    #     print(f)
        
    # repeatBSLR()
    # subsetSelection()
    permFeatImport()