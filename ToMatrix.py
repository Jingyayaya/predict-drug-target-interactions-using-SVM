
import pandas as pd


import seaborn as sns
sns.set_style("darkgrid",{"font.sans-serif":['KaiTi', 'Arial']})#chinease set

import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline


# Large-scale prediction of drug-target interactions from heterogeneous biological data for drug discovery and repositioning
# https://academic.oup.com/bib/article/15/5/734/2422306
# http://www.doc88.com/p-3307854424381.html
# Drug Target Interpretable similarity matrix
# http://www-erato.ist.hokudai.ac.jp/docs/seminar/yamanishi_erato120810.pdf
# file:///E:/chorme_dowmload/molecules-23-02208%20(1).pdf
#the data in tdi is paired-wise，create 664*445=TDI.shape
# Drug Target interactions  similarity matrix svm
# Collaborative Matrix Factorization with Multiple Similarities for Predicting Dru.
# Schematic figure of PKM. Similarity between ( d , t ) and
# ( d' , t' ) can be computed by the inner product of the drug
# similarity between d and d' and the target similarity between t and t'

# The pairwise kernel method (PKM) [7] developed by Jacob et al.

def toMatrix():
    TS = pd.read_table("target-target similarity.txt", header=None, sep='\t', encoding="gbk")
    DS = pd.read_table("drug-drug similarity.txt", header=None, sep='\t', encoding="gbk")
    TDI = pd.read_table("Enzymes drug-target interactions.txt", header=None, sep='\t', encoding="gbk")

    TDI.columns = ["target", "drug", "interactions"]
    DS.columns = ["drug", "drug2", "similarity"]
    TS.columns = ["target", "target2", "similarity"]

    s=TDI.pivot('target','drug','interactions')
    s.index.name = None
    s.columns.name = None
    s.to_csv('TDI.txt', sep='\t', index=True)
    s=DS.pivot('drug','drug2','similarity')
    s.index.name = None
    s.columns.name = None
    s.to_csv('DS.txt', sep='\t', index=True)
    s=TS.pivot('target','target2','similarity')
    s.index.name = None
    s.columns.name = None
    s.to_csv('TS.txt', sep='\t', index=True)

