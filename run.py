

import convertfile as cf
import cross_validation as cv
import evaluate as ev
from pkm_mvc import PairwiseKernelMethod
from ToMatrix import toMatrix
#toMatrix()
Kp = cf.readMat("TS.txt")
Kc = cf.readMat("DS.txt")
correctLabel = cf.readMat("TDI.txt")
LabelP, LabelC = cf.readAxisLabel("TDI.txt")
# initialize the data
pkm = PairwiseKernelMethod(Kp, Kc, correctLabel, LabelP, LabelC)
# cv_set = cv.Kfold_interaction(correctLabel, 10)
# K-fold validation
cv_set = cv.Kfold_interaction(correctLabel, 2)
ave = 0
for train, test in cv_set:
    print("begin")
    auroc = ev.AUROC(correctLabel.ravel()[test], pkm.run(train,test)[:,1])
    ave += auroc
    print(auroc)
print("average:" + str(ave/2))
