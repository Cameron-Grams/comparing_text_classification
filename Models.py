import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import cross_validate, KFold



def logistic_regrssions_classifier_assessment(X, y):
    """
    Logistic Regression comparison for the different datasets

    args: X dataset of features
          y target set of values for classification

    returns:
        - a dummy classifier score using "most frequent" value assignment
        - the mean of the Logistic Regression Claffifier prediction scores taken from a
          5 fold cross validation on the dataset
        - graphic as a callable function plotting the ROC/AUC curves
    """

    clf_log = LogisticRegression(random_state=0)

    def performance_graphics(X=X, y=y, clf=clf_log):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        clf.fit(X_train, y_train)

        proba_ = clf.decision_function(X_test)

        precision, recall, thresholds = precision_recall_curve(y_test, proba_)
        fp, tp, thresholds_roc = roc_curve(y_test, proba_)

        auc_score = np.round(roc_auc_score(y_test, proba_), 4)

        close_default = np.argmin(np.abs(thresholds))
        close_zero = np.argmin(np.abs(thresholds_roc))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

        ax1.plot(precision, recall, label="Precision Recall Curve")
        ax1.plot(precision[close_default],
                 recall[close_default], 'o',
                 c='r', markersize=10,
                 label='threshold 0',
                 fillstyle="none", mew=2)
        ax1.set_title("Logistic Regression performance")
        ax1.set_xlabel("Precision")
        ax1.set_ylabel("Recall")
        ax1.legend(loc='best')

        ax2.plot(fp, tp, label="ROC curve")
        ax2.plot(fp[close_zero],
                 tp[close_zero], 'o',
                 c='r', markersize=10,
                 label='threshold 0',
                 fillstyle="none", mew=2)
        ax2.set_title(f"ROC performance: AUC Score {auc_score}")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive (Recall)")
        ax2.legend(loc='best')

        plt.show();

    kfold = KFold(n_splits=5)
    cross_val = cross_validate(clf_log, X, y, cv=kfold, return_estimator=True)
    mean_score = cross_val['test_score'].mean()

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X, y)
    d_score = dummy_clf.score(X, y)

    graphic = performance_graphics

    return d_score, mean_score, graphic
# %%

#------------------------------------------------------------





#-------------------------------------------------------------
def SVM_classifier_assessment(X, y):
    """
    SVC comparison for the different datasets

    args: X dataset of features
          y target set of values for classification

    returns: 
        - a dummy classifier score using "most frequent" value assignment
        - the mean of the SVC prediction scores taken from a 
          5 fold cross validation on the dataset
    """
    
    clf_svc = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
        
    def performance_graphics(X=X, y=y, clf=clf_svc):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        clf.fit(X_train, y_train)
     
        proba_ = clf.predict_proba(X_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_test, proba_)
        fp, tp, thresholds_roc = roc_curve(y_test, proba_)
        
        auc_score = np.round(roc_auc_score(y_test, proba_), 4)

        close_default = np.argmin(np.abs(thresholds - 0.5))
        close_zero = np.argmin(np.abs(thresholds_roc))

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 4))

        ax1.plot(precision, recall, label="Precision Recall Curve")
        ax1.plot(precision[close_default], 
                 recall[close_default], 'o', 
                 c='r', markersize=10, 
                 label='threshold 0.5', 
                 fillstyle="none", mew=2)
        ax1.set_title("RF performance")
        ax1.set_xlabel("Precision")
        ax1.set_ylabel("Recall")
        ax1.legend(loc='best')

        ax2.plot(fp, tp, label="ROC curve")
        ax2.plot(fp[close_zero], 
                 tp[close_zero], 'o', 
                 c='r', markersize=10, 
                 label='threshold 0', 
                 fillstyle="none", mew=2)
        ax2.set_title(f"ROC performance: AUC Score {auc_score}")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive (Recall)")
        ax2.legend(loc='best')

        plt.show();

    kfold = KFold(n_splits=5)
    cross_val = cross_validate(clf_svc, X, y, cv=kfold, return_estimator=True)
    mean_score = cross_val['test_score'].mean()
        
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X,y)
    d_score = dummy_clf.score(X,y)
    
    graphic = performance_graphics

    return d_score, mean_score, graphic

#-------------------------------------------------------------------------------





#-------------------------------------------------------------------------------

def NaiveBayes_classifier_assessment(X, y, priors=None):
    """
    GaussianNB comparison for the different datasets

    args: X dataset of features
          y target set of values for classification

    returns: 
        - a dummy classifier score using "most frequent" value assignment
        - the mean of the Gaussian NaiveBayes Claffifier prediction scores taken from a 
          5 fold cross validation on the dataset
    """
    
    clf_nb = GaussianNB(priors=priors)
    
    print(f"Using prior probability: {priors}")
    
    def performance_graphics(X=X, y=y, clf=clf_nb):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        clf.fit(X_train, y_train)
     
        proba_ = clf.predict_proba(X_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_test, proba_)
        fp, tp, thresholds_roc = roc_curve(y_test, proba_)
        
        auc_score = np.round(roc_auc_score(y_test, proba_), 4)

        close_default = np.argmin(np.abs(thresholds - 0.5))

        close_zero = np.argmin(np.abs(thresholds_roc))

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 4))

        ax1.plot(precision, recall, label="Precision Recall Curve")
        ax1.plot(precision[close_default], 
                 recall[close_default], 'o', 
                 c='r', markersize=10, 
                 label='threshold 0.5', 
                 fillstyle="none", mew=2)
        ax1.set_title("NaiveBayes performance")
        ax1.set_xlabel("Precision")
        ax1.set_ylabel("Recall")
        ax1.legend(loc='best')

        ax2.plot(fp, tp, label="ROC curve")
        ax2.plot(fp[close_zero], 
                 tp[close_zero], 'o', 
                 c='r', markersize=10, 
                 label='threshold 0.5', 
                 fillstyle="none", mew=2)
        ax2.set_title(f"ROC performance: AUC Score {auc_score}")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive (Recall)")
        ax2.legend(loc='best')

        plt.show();

    kfold = KFold(n_splits=5)
    cross_val = cross_validate(clf_nb, X, y, cv=kfold, return_estimator=True)
    mean_score = cross_val['test_score'].mean()
        
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X,y)
    d_score = dummy_clf.score(X,y)
    
    graphic = performance_graphics

    return d_score, mean_score, graphic

#----------------------------------------------------------------------------------





#----------------------------------------------------------------------------------

def RandomForest_classifier_assessment(X, y):
    """
    RandomForestClassifier comparison for the different datasets

    args: X dataset of features
          y target set of values for classification

    returns: 
        - a dummy classifier score using "most frequent" value assignment
        - the mean of the RandomForest Claffifier prediction scores taken from a 
          5 fold cross validation on the dataset
        - a dataframe that shows the 10 most important features used by the classifer          
    """
    
    clf_rf = RandomForestClassifier(max_depth=4, random_state=0)
    
    def performance_graphics(X=X, y=y, clf_rf=clf_rf):
        # type of certainty tied to classifier passed in
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        clf_rf.fit(X_train, y_train)
     
        rf_proba = clf_rf.predict_proba(X_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_test, rf_proba)
        fp, tp, thresholds_roc = roc_curve(y_test, rf_proba)
        
        auc_score = np.round(roc_auc_score(y_test, rf_proba), 4)

        close_default = np.argmin(np.abs(thresholds - 0.5))




        close_zero = np.argmin(np.abs(thresholds_roc))

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 4))

        ax1.plot(precision, recall, label="Precision Recall Curve")
        ax1.plot(precision[close_default], 
                 recall[close_default], 'o', 
                 c='r', markersize=10, 
                 label='threshold 0.5', 
                 fillstyle="none", mew=2)
        ax1.set_title("RandomForest performance")
        ax1.set_xlabel("Precision")
        ax1.set_ylabel("Recall")
        ax1.legend(loc='best')

        ax2.plot(fp, tp, label="ROC curve")
        ax2.plot(fp[close_zero], 
                 tp[close_zero], 'o', 
                 c='r', markersize=10, 
                 label='threshold 0.5', 
                 fillstyle="none", mew=2)
        ax2.set_title(f"ROC performance: AUC Score {auc_score}")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive (Recall)")
        ax2.legend(loc='best')

        plt.show();



    clf_rf = RandomForestClassifier(max_depth=4, random_state=0)

    kfold = KFold(n_splits=5)
    cross_val = cross_validate(clf_rf, X, y, cv=kfold, return_estimator=True)
    mean_score = cross_val['test_score'].mean()

    estimator = cross_val['estimator']
    ranked_features = {}

    for i, clf in enumerate(estimator):
        clf_no = i + 1
        feat_imp_val = clf.feature_importances_
        cols = X.columns
        feature_importance = list(zip(cols, feat_imp_val))
        feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
        feature_importance = [ f[0] for f in feature_importance]
        ranked_features[f"Estimator: {clf_no}"] = feature_importance[:10]
        
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X,y)
    d_score = dummy_clf.score(X,y)
    
    df = pd.DataFrame(ranked_features)

    graphic = performance_graphics

    return d_score, mean_score, df, graphic


