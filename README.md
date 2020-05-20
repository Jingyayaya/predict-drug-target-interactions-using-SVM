# predict-drug-target-interactions-using-SVM
predict drug-target interactions using SVM
1. train data
The train data includes instances of enzymes drug-target interactions with drug and target(protein) similarities which show relationships 
among drugs and those among targets respectively.
2. extracting features
Extracted features to describe a drug-protein pair using similarities among drugs and those among proteins.
3. Model
Trained an SVM classifier for predicting whether a drug interacted with a target. 
4. Validation
Used the 2-fold cross validation. The performance was shown by Receiver Operating Characteristic (ROC). The AUC value was 0.88.
