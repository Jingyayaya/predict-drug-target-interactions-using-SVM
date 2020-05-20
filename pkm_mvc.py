
import numpy as np
from sklearn.model_selection import KFold
from sklearn import svm
import evaluate as ev

class PairwiseKernelMethod:
    kernel_target = np.array([[]])
    kernel_compound = np.array([[]])
    interaction = np.array([[]])
    target_lst = []
    compound_lst = []
    __classifier = svm.SVC(kernel='precomputed', probability=True)

    def __init__(self, kernel_tar, kernel_comp, interact, target, compound):
        self.kernel_target = kernel_tar
        self.kernel_compound = kernel_comp
        self.interaction = interact
        self.target_lst = target
        self.compound_lst = compound

    def run(self, train_data, test_data):
        # self.__classifier.C = self.innerLoopCV(train_data)
        self.__classifier.C = 0.1
        extracted_train_data = self.extractNegativeSample(train_data)
        # According to extracted_train_data，create test set mapped with it.
        Ktest = self.makePredictVector(extracted_train_data, test_data)
        # creat train set
        Ktrain = self.makePairwiseKernel(extracted_train_data)
        # train
        probas_ = self.probas(extracted_train_data, Ktest, Ktrain)
        return probas_

    #return probability vector
    def probas(self, train_data, Ktest, Ktrain):
        teach_label = self.interaction.ravel()[train_data]
        probas_ = self.__classifier.fit(Ktrain, teach_label).predict_proba(Ktest)
        return probas_

    #return list of coordinates

    def extractNegativeSample(self, train):
        '''
        blend the 0 coordinate sets with equal number of 1
        :param train: the coordinate of training set
        :return:
        '''
        col = self.interaction.shape[0]
        row = self.interaction.shape[1]
        alv = np.zeros(col * row)
        for i in train:
            alv[i] = self.interaction[int(i/row)][i%row]
        one = np.asarray(np.where(alv==1)[0])
        zero = np.asarray(np.where(alv==0)[0])
        #randomly filtering the zero coodinate with equal number with 1，blend sample
        zero = np.random.choice(zero, one.size, replace=False)
        one = np.sort(np.append(one, zero))
        return one

    #convert array point to matrix coodinate
    def convertToCood(self, data, row):
        '''
        transform data into row-column matrix
        :param data:
        :param row:
        :return:
        '''
        ret = []
        for i in data:
            ret.append((int(i/row), i%row))
        return np.asarray(ret)

    #use for making train data
    def makePairwiseKernel(self, data):
        '''

        :param data:
        :return:
        '''
        row = self.interaction.shape[1]
        size = len(data)
        coodinate = self.convertToCood(data, row)
        Kpair = np.zeros((size, size))
        x_count = 0
        for p in coodinate:
            y_count = 0
            for q in coodinate:
                if(x_count < y_count):
                    break
                #kpair=kernel_target*kernel_compound
                Kpair[x_count][y_count] = self.kernel_target[p[0]][q[0]] * self.kernel_compound[p[1]][q[1]]
                Kpair[y_count][x_count] = self.kernel_target[p[0]][q[0]] * self.kernel_compound[p[1]][q[1]]
                y_count += 1
            x_count += 1
        return Kpair

    #use for making test data
    def makePredictVector(self, train_data, test_data):
        '''
        create set for testing
        :param train_data: train data
        :param test_data:teset data
        :return:
        '''
        row = self.interaction.shape[1]
        x = self.convertToCood(test_data, row)
        y = self.convertToCood(train_data, row)
        Ktest = np.zeros((len(x), len(y)))
        x_count = 0
        for p in x:
            y_count = 0
            for q in y:
                # create ktest=kernel_target*kernel_compound
                Ktest[x_count][y_count] = self.kernel_target[p[0]][q[0]] * self.kernel_compound[p[1]][q[1]]
                y_count += 1
            x_count += 1
        return Ktest

    #train_data -> [1,2,5,7,14,45,75,90...]
    #use to decide svm's paramater C
    def innerLoopCV(self, train_data):
        # time-consuming, discard
        C_lst = [0.1, 1, 10, 100, 1000]
        max_roc = 0
        max_C = 0
        #acutually, this dataset has to change in accordance with type of cross validation
        inner_cv = KFold(3, shuffle=True).split(train_data)
        for C in C_lst:
            print("Start")
            self.__classifier.C = C
            inner_sum = 0
            for inner_train, inner_test in inner_cv:
                inner_train_data = self.extractNegativeSample(train_data[inner_train])
                Kinnertrain = self.makePairwiseKernel(inner_train_data)
                Kinnertest = self.makePredictVector(inner_train_data, train_data[inner_test])
                probas_ = self.probas(inner_train_data, Kinnertest, Kinnertrain)
                auroc = ev.AUROC(self.interaction.ravel()[train_data[inner_test]], probas_[:, 1])
                print(C,auroc)
                inner_sum += auroc
            if(max_roc < inner_sum):
                max_C = C
                max_roc = inner_sum
        return max_C
