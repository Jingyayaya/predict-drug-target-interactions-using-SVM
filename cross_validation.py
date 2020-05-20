
import numpy as np
from sklearn.model_selection import StratifiedKFold,KFold
import random
from sklearn.model_selection import train_test_split
#make protein-wise cross validation dataset
def Kfold_protein(interaction, n_folds):
    ret = []
    t_len, c_len = interaction.shape
    dataset = np.asarray(range(t_len*c_len))
    for i in range(n_folds):
        samp = random.sample(range(t_len), t_len/n_folds)
        samp.sort()
        test = []
        for j in samp:
            test.extend(np.where((dataset>=j*c_len)&(dataset<(j+1)*c_len))[0].tolist())
        train = list(set(range(len(dataset))) - set(test))
        train.sort()
        ret.append((np.asarray(train),np.asarray(test)))
    return ret

#make compound-wise cross validation dataset
def Kfold_compound(interaction, n_folds):
    ret = []
    t_len, c_len = interaction.shape
    dataset = np.asarray(range(t_len*c_len))
    for i in range(n_folds):
        samp = random.sample(range(c_len), c_len/n_folds)
        samp.sort()
        test = []
        for j in samp:
            test.extend(np.where(dataset%c_len == j)[0].tolist())
        test.sort()
        train = list(set(range(len(dataset))) - set(test))
        train.sort()
        ret.append((np.asarray(train),np.asarray(test)))
    return ret

#make interaction-wise cross validation dataset
def Kfold_interaction(interaction, n_folds):
    dataset = interaction.ravel()
    return StratifiedKFold(n_splits = n_folds, shuffle = True).split(dataset,dataset)
    # return train_test_split(list(range(dataset.shape)),test_size=0.3, random_state=0,stratify=dataset, shuffle=True)
