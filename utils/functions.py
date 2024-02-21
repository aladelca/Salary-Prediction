import re
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import precision_recall_curve

def clean_data(string):
    cleaned_string = re.sub(r'[^a-zA-Z0-9 ]', '', string)
    cleaned_string = cleaned_string.lower()
    return cleaned_string

def get_metrics(y_pred, y_proba, y_test, plot=False):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    sensitivity = metrics.recall_score(y_test, y_pred, pos_label=1)
    specificity = metrics.recall_score(y_test, y_pred, pos_label=0)
    cm = metrics.confusion_matrix(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_proba[:,1])
    if plot:
        disp = metrics.ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.show()
    else:
        pass
    final_metrics = {
        'accuracy': accuracy, 
        'sensitivity': sensitivity, 
        'specificity': specificity, 
        'auc': auc}
    return final_metrics, cm


def vectorizing(min_range, max_range,x_train, x_test ):
    vect = CountVectorizer(stop_words='english',  ngram_range=(min_range,max_range))
    x_train_vect = vect.fit_transform(x_train['final_description'])
    x_test_vect = vect.transform(x_test['final_description'])
    return x_train_vect, x_test_vect, vect

def training(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model

def predict(model, x_test):
    preds = model.predict(x_test)
    proba = model.predict_proba(x_test)
    return preds, proba

def main_process(min_range, max_range, x_train, x_test, y_train, y_test, model):
    x_train_vect, x_test_vect,_ = vectorizing(min_range, max_range, x_train, x_test)   
    model = training(model, x_train_vect, y_train)
    preds, proba = predict(model, x_test_vect)
    final_metrics, cm = get_metrics(preds, proba, y_test)
    return final_metrics['auc']

def plot_roc_auc(y_test, probas):
    preds = probas[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def plot_precision_recall(y_test, probas):
    y_proba = probas[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    auc_precision_recall = metrics.auc(recall, precision)
    plt.figure()
    plt.plot(thresholds, precision[:-1], "b--", label="Precision")
    plt.plot(thresholds, recall[:-1], "g-", label="Recall")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precisión-Recall')
    plt.legend(loc='best')
    plt.show()
    return precision, recall, thresholds