from sklearn import tree
#from sklearn.spatial import distance
from sklearn.metrics import accuracy_score 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import math
def euc(a,b):
    a=np.array(a)
    b=np.array(b)

    sq=(a-b)**2

    return math.sqrt(np.sum(sq))

class UserKNN():
    def fit(self,trainingData,TrainingTarget):
        self.TrainingData=trainingData
        self.TrainingTarget=TrainingTarget

    def predict(self,TestData):
        predictions=[]
        for row in TestData:
            label=self.closest(row)
            predictions.append(label)
        return predictions
    def closest(self,row):
        bestdistance=euc(1,len(self.TrainingData[0]))
        bestindex=0
        for i in range(1,len(self.TrainingData)):
            dist=euc(row,self.TrainingData[i])
            if dist < bestdistance:
                bestdistance=dist
                bestindex=i
        return self.TrainingData[bestindex]

def UserKNeighbour():
    border="-"*50
    iris=load_iris()

    data=iris.data
    target=iris.target

    print(border)
    print("\t\tActual Datasets")
    print(border)

    for i in range(len(iris.target)):
        print("ID : %d ,label %s , Feature : %s"%(9,iris.data[i],iris.target[i]))
    print("size of actual dataset %d"%(i+1))
    data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.5)

    print(border)
    print("\t\tTraining Dataset")
    print(border)

    for i in range(len(data_train)):
        print("ID : %d , Label %s , Feature : %s "%(i,data_train[i],target_train[i]))
    print("size of training dataset : %d"%(i+1))
    
    print(border)
    print("\t\t Test Dataset")
    print(border)

    for i in range(len(data_test)):
        print("ID : %d , Label %s , Feature : %s "%(i,data_test[i],target_test[i]))
    print("size of training dataset : %d"%(i+1))
    
    classifier=UserKNN()
    
    classifier.fit(data_train,target_train)
    
    predictions=classifier.predict(data_test)

    Accuracy=accuracy_score(target_test,predictions)

    return Accuracy
def main():
    Accuracy=UserKNeighbour()
    print("Accuracy Score of Classification algorith with K Neighbor classifier is ",Accuracy*100,"%")
if __name__=="__main__":
    main()
