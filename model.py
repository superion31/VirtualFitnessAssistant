# Importing Libraries
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold 
from sklearn import metrics
from sklearn.svm import SVC

# Importing Dataset
df = pd.read_csv('data.csv')

#Label Encoding
le = LabelEncoder()
df['LABEL']= le.fit_transform(df['LABEL'])

#Seperationg dependent and independent variables
X = df.drop(columns = ['LABEL']).copy()
y = df['LABEL']

def train_test_same(tr_size = 0.7, kernel_type = 'linear'):
    #Train-Remaining split
    X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=tr_size)
    
    #Split the remaining data to validation and test sets
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

    #Apply 10 fold split in the train set.
    kf = KFold( n_splits=10, shuffle=False, random_state=None)

    #Initialize parameters
    best_acc = 0.0
    best_svm_clf = None

    #k-fold proccess
    for train_index, test_index in kf.split(X_train):
      X_learn, X_check = X_train.iloc[train_index], X_train.iloc[test_index]
      y_learn, y_check = y_train.iloc[train_index], y_train.iloc[test_index]
      svm_clf = SVC(kernel = kernel_type)
      svm_clf = svm_clf.fit(X_learn, y_learn)
      y_pred = svm_clf.predict(X_check)
      acc_check = metrics.accuracy_score(y_check, y_pred)
      if acc_check > best_acc:
        best_acc = acc_check
        best_svm_clf = svm_clf

    #use the best_clf to predict validation set. Calculate expected accuracy.
    y_pred = best_svm_clf.predict(X_valid)
    expected_acc = metrics.accuracy_score(y_valid, y_pred)

    #calculate the real accuracy
    y_pred = best_svm_clf.predict(X_test)
    real_acc = metrics.accuracy_score(y_test, y_pred)
    
    return expected_acc, real_acc, best_svm_clf

def train_sep(prob_index = True):
    #fit in all data
    svm_clf = SVC(probability=prob_index).fit(X,y)
    
    return svm_clf
