import numpy as np
import random

class KFold:
    """
    X: data
    y: label
    k: number of fold
    group 1: number of samples are the same in each class
          2: number of samples might be different in each class
    indexList: number of samples in each class
               default None in group 1 case
               this parameter is required in group 2 case
    """
    def __init__(self, X, y, k, group=1,indexList=None):
        self.X = X
        self.y = y
        self.k = k
        self.group = group
        self.indexList = indexList
        self._order = self.generateOrder()

    """
    num: number of samples in each class(int)
    list: number of samples for one class in each fold
    """
    def generateRandomInt(self,num):
        if num < self.k:
            raise ValueError('KFold only holds %i folds' % num)
        val = int(num / self.k)
        list = [val]*self.k
        # check remainder
        if num-val*self.k != 0:
            for i in range(0,num-val*self.k):
                list[i] += 1
        return list

    """
    num: number of samples in each class(list: 1×class number)
         this function is used in group 2 case
    list: number of samples for one class in each fold
          (list: class number×fold number)
    """
    def generateRandomList(self,num):
        list = []
        for idx in range(0,len(num)):
            list.append(self.generateRandomInt(num[idx]))
        return list

    """
    return the index of samples in each fold according to the random list generated above
    """
    def generateOrder(self):
        # list record the value is selected in each fold
        list = np.zeros(self.X.shape[0])
        res = []
        # number of classes
        classNum = len(np.unique(self.y))
        if self.group == 1:
            num = int(self.X.shape[0]/classNum)
            ranList = self.generateRandomInt(num)
            for i in range(0,len(ranList)-1):
                index = []
                for idx in range(0,classNum):
                    # not optimize
                    flag = True
                    resultList = []
                    while flag:
                        resultList = random.sample(range(idx*num,(idx+1)*num),ranList[i])
                        flag = False
                        for j in range(len(resultList)):
                            if list[resultList[j]] != 0:
                                flag = True;
                                break
                    # random list get√
                    index = index + resultList
                print(index)
                res.append(index)
                # del those selected num
                for j in range(0,len(index)):
                    list[index[j]] = 1
            # the last fold
            index = []
            for j in range(0,len(list)):
                if list[j] == 0:
                    index += [j]
            res.append(index)
        else:
            ranList = self.generateRandomList(self.indexList)
            for i in range(0,self.k-1):
                index = []
                count = 0
                for idx in range(0,len(ranList)):
                    # not optimize
                    flag = True
                    resultList = []
                    while flag:
                        resultList = random.sample(range(count,count+self.indexList[idx]),ranList[idx][i])
                        flag = False
                        for j in range(len(resultList)):
                            if list[resultList[j]] != 0:
                                flag = True;
                                break
                    # random list get√
                    index += resultList
                    count += self.indexList[idx]
                res.append(index)
                # del those selected num
                for j in range(0,len(index)):
                    list[index[j]] = 1
            # the last fold
            index = []
            for j in range(0,len(list)):
                if list[j] == 0:
                    index += [j]
            res.append(index)
        return res

    """
    instance: fold number, start from 0 (int)
    return the training and testing data according to the fold index
    """
    def getItem(self, instance):
        if instance >= self.k:
            raise ValueError('KFold only holds %i folds' % self.k)

        mask = np.zeros(self.X.shape[0], dtype=bool)
        for idx in range(0,len(self._order[instance])):
            mask[self._order[instance][idx]] = True

        x_train = self.X[~mask]
        y_train = np.array(self.y)[~mask]
        x_test = self.X[mask]
        y_test = np.array(self.y)[mask]

        return x_train, y_train, x_test, y_test
