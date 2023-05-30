import csv
import numpy as np
import math

def loadData(path):
    f = open(path, "r")
    data = csv.reader(f) #csv format
    data = np.array(list(data))# covert to matrix
    data = np.delete(data, 0, 0)# delete header
    data = np.delete(data, 0, 1) # delete index
    np.random.shuffle(data) # shuffle data
    f.close()
    trainSet = data[:100] #training data from 1->100
    testSet = data[100:]# the others is testing data
    return trainSet, testSet

def calcDistancs(pointA, pointB, numOfFeature=4):
    tmp = 0
    for i in range(numOfFeature):
        tmp += (float(pointA[i]) - float(pointB[i])) ** 2
    return math.sqrt(tmp)

def kNearestNeighbor(trainSet, point, k):
    distances = []
    for item in trainSet:
        distances.append({
            "label": item[-1],
            "value": calcDistancs(item, point)
        })
    distances.sort(key=lambda x: x["value"])
    labels = [item["label"] for item in distances]
    return labels[:k]

def findMostOccur(arr):
    labels = set(arr) # set label
    ans = ""
    maxOccur = 0
    for label in labels:
        num = arr.count(label)
        if num > maxOccur:
            maxOccur = num
            ans = label
    return ans

if __name__ == "__main__":
    trainSet, testSet = loadData("Iris.csv")
    for item in testSet:
        knn = kNearestNeighbor(trainSet, item, 5)
        answer = findMostOccur(knn)
        print("label: {} -> predicted: {}".format(item[-1], answer))