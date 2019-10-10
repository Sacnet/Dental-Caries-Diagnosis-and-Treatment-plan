import pandas as pd; 
import numpy as np 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, \
MaxAbsScaler, RobustScaler,StandardScaler, MaxAbsScaler
from wolpert import make_stack_layer, StackingPipeline; import seaborn as sn
from sklearn.metrics import confusion_matrix; 
import warnings
#from mlxtend.evaluate import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from deap import algorithms, tools, creator
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE, RFECV,SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier; import csv
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from vecstack import stacking
from mlxtend.classifier import StackingClassifier
from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import scikitplot as skpt
warnings.filterwarnings('ignore')

path = 'datasets/data.csv'
path2 = 'datasets/newdatau.csv'
models = [MultinomialNB(), RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5),MLPClassifier(hidden_layer_sizes=(500, 500))]

def process_data(data):
    data = pd.read_csv(data, quoting=csv.QUOTE_NONE)
    data = data.reset_index()

    data['classification'] = data['classification'].map({'Moderate Risk':'moderate', 'Low Risk':'low', 'High Risk':'high'})
    data1 = data[['age', 'classification', 'present_complain', 'location','duration','pain_sympton', 'swelling_symptom',\
    'nona','stimulus_pain','character_pain','sensitivity','nocarousteeth', 'notooth_affect', \
     'facial_assymmetry', 'investigation', 'diet', 'aggravating_factor', 'progression', 'pain']]     

    return data1, data.iloc[:, -1]


label_cls = LabelEncoder()
label_agg = LabelEncoder()
label_acs = LabelEncoder()
label_pain = LabelEncoder()


lencoder_y = LabelEncoder()


def process_data2(data):
    data = pd.read_csv(data, quoting=csv.QUOTE_NONE)
    data = data.reset_index()    
    #data['number_voc'] = data['number_voc'].fillna(data['number_voc'].mean(axis=0))    
    data2 = data[['age', 'classification', 'present_complain', 'location','duration','pain_sympton', 'swelling_symptom',\
    'nona','stimulus_pain','character_pain','sensitivity','nocarousteeth', 'notooth_affect', \
     'facial_assymmetry', 'investigation', 'diet', 'aggravating_factor', 'progression', 'pain', 'diagnosis']]     
    
    data2['classification'] = data2['classification'].map({'Moderate Risk':'moderate', 'Low Risk':'low', 'High Risk':'high'})
    data2['duration'] = data2['duration'].astype(float)
    return data2.iloc[:, :-1], data2.iloc[:, -1]

def imputation_process():
    X, y = process_data(path)
    X = X.fillna(X.mean(axis=0))
    
    return X, y 
X, y = imputation_process()

X = X.values 
y = y.values


X[:,1] = label_cls.fit_transform(X[:,1])
X[:,16] = label_agg.fit_transform(X[:,16])
X[:,17] = label_acs.fit_transform(X[:,17])
X[:,18] = label_pain.fit_transform(X[:,18])

y = lencoder_y.fit_transform(y)


lencoder2 = LabelEncoder()

def imputation_process2():
    X2, y2 = process_data2(path2)        
    X2 = X2.fillna(X2.mean(axis=0))

    y2 = y2.values
    X2 = X2.values
    #X[:, 1] = lencoder2.fit_transform(X[:,1])
    for i in range(1,2):
        X2[:, i] = lencoder2.fit_transform(X2[:, i]) 
    for j in range(16, 19):
        X2[:, j] = lencoder2.fit_transform(X2[:, j])
    
    y = lencoder2.fit_transform(y2)
        
    return X2, y

X2, y2= imputation_process2()

def scaling(xtr, xts, scaler=None):
    scaler_x = scaler.fit_transform(xtr)
    scaler_t = scaler.transform(xts)
    return scaler_x, scaler_t

def fit_predict_func(xtr, xts, ytr, scaler=None):    
    if scaler == None:
        for model in models:
            if model == models[0]:
                nb = model.fit(xtr, ytr)
                prenb = nb.predict(xts)                
            elif model == models[1]:
                svm = model.fit(xtr, ytr)
                presvm = svm.predict(xts)                
            elif model == models[2]:
                mlp = model.fit(xtr, ytr)
                premlp = mlp.predict(xts)                
        return prenb, presvm, premlp
    else:        
        #scaler_x = scaler.fit_transform(xtr)
        #scaler_t = scaler.transform(xts)
        scaler_x, scaler_t = scaling(xtr, xts, scaler)
        for model in models:
            if model == models[0]:
                nb = model.fit(scaler_x, ytr)
                prenb = nb.predict(scaler_t)                
            elif model == models[1]:
                svm = model.fit(scaler_x, ytr)
                presvm = svm.predict(scaler_t)                
            elif model == models[2]:
                mlp = model.fit(scaler_x, ytr)
                premlp = mlp.predict(scaler_t)                
        return prenb, presvm, premlp

def data_normalization():
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=7)
    return Xtrain, Xtest, ytrain, ytest
dx, dy = imputation_process2()

def data_normalization2():
    dxtrain, dxtest, dytrain, dytest = train_test_split(dx, dy, test_size=0.2, random_state=7)
    return dxtrain, dxtest, dytrain, dytest

Xtrain, Xtest, ytrain, ytest = data_normalization()
Xtrain2, Xtest2, ytrain2, ytest2 = data_normalization2()


clf_svm_nb = [RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5), GaussianNB()]
clf_svm_mlp = [RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5), MLPClassifier(hidden_layer_sizes=(500, 500))]
clf_nb_mlp = [MLPClassifier(hidden_layer_sizes=(500,500)), GaussianNB()]

def stacking_prediction(m1, m2, meta):
    # model_train, model_test = stacking(clf, Xtrain, ytrain, Xtest)
    # m = model.fit(model_train, ytrain)
    tr, ts = scaling(Xtrain, Xtest, MaxAbsScaler())
    m = StackingClassifier(classifiers=[m1,m2],meta_classifier=meta)
    m.fit(tr, ytrain)
    predict_m = m.predict(ts)
    return predict_m
def stacking_prediction2(m1, m2, meta):
    # model_train, model_test = stacking(clf, Xtrain2,ytrain2, Xtest2)
    # model.fit(model_train, ytrain2)
    tr, ts = scaling(Xtrain2,Xtest2,MaxAbsScaler())
    m = StackingClassifier(classifiers=[m1, m2],meta_classifier=meta) 
    m.fit(tr, ytrain2)
    predict_mm = m.predict(ts)
    return predict_mm

# def final_stacking_NOFS():
#     mlps = stacking_prediction2()
#     nbs = stacking_prediction2(clf_svm_mlp, GaussianNB())
#     svms = stacking_prediction2(clf_nb_mlp, SVC(kernel='rbf'))
#     return mlps, nbs, svms

def final_stacking():
    mlp_st = stacking_prediction(GaussianNB(), RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5),MLPClassifier(hidden_layer_sizes=(500,500)))
    nb_st = stacking_prediction(MLPClassifier(hidden_layer_sizes=(500,500)),RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5), GaussianNB())
    svm_st = stacking_prediction(MLPClassifier(hidden_layer_sizes=(500,500)),GaussianNB(), RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5))
    return mlp_st, nb_st, svm_st

def final_stacking2():
    mlp_st = stacking_prediction2(GaussianNB(), RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5),MLPClassifier(hidden_layer_sizes=(500,500)))
    nb_st = stacking_prediction2(MLPClassifier(hidden_layer_sizes=(500,500)),RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5), GaussianNB())
    svm_st = stacking_prediction2(MLPClassifier(hidden_layer_sizes=(500,500)),GaussianNB(), RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5))
    return mlp_st, nb_st, svm_st

# def plot_cm(name,pre):
#     skpt.metrics.plot_confusion_matrix(ytest, pre)   
#     plt.title(f"Accuracy:{round(accuracy_score(ytest, pre), 4)}")       
#     plt.savefig(f'static/images/{name}.png')      
     

# def plot_cm_graph(): 
#     ml, nb, sm = final_stacking()     
#     plot_cm('mlp',ml)
#     plot_cm('nb',nb)
#     plot_cm('svm', sm)

import seaborn as sn

#plot_cm_graph()
def perform_metrics(prediction):    
    acc = accuracy_score(ytest, prediction)
    prc = precision_score(ytest, prediction, average='macro')
    rc = recall_score(ytest, prediction, average='macro')
    fm = f1_score(ytest, prediction, average='macro')
    #roc = roc_auc_score(yts, prediction, average='macro')
    cm = confusion_matrix(ytest, prediction)
    return acc, prc, rc,fm, cm

def perform_metrics2(prediction):    
    acc = accuracy_score(ytest2, prediction)
    prc = precision_score(ytest2, prediction, average='macro')
    rc = recall_score(ytest2, prediction, average='macro')
    fm = f1_score(ytest2, prediction, average='macro')
    #roc = roc_auc_score(yts, prediction, average='macro')
    cm = confusion_matrix(ytest2, prediction)
    return acc, prc, rc,fm, cm

def perform_ml_matrix():
    nb, svm, mlp = fit_predict_func(Xtrain, Xtest, ytrain, MaxAbsScaler())
    return nb, svm, mlp 

def perform_ml_matrix2():
    nb, svm, mlp = fit_predict_func(Xtrain2, Xtest2, ytrain2, MaxAbsScaler())
    return nb, svm, mlp 

def _StackModel_F(xt, ytr, xts=None, sc=None):
    xtr, _ = scaling(xt, xts, sc)
    svm_nb_layer = make_stack_layer(RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5), GaussianNB())
    svm_mlp_layer = make_stack_layer(RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5),MLPClassifier(hidden_layer_sizes=(500,500)))
    nb_mlp_layer = make_stack_layer(GaussianNB(), MLPClassifier(hidden_layer_sizes=(500,500)))

    mlp_clf = StackingPipeline([('l0',svm_nb_layer),('l1', MLPClassifier(hidden_layer_sizes=(500,500)))])
    mlp_clf.fit(xtr, ytr)
    
    nb_cf = StackingPipeline([('l0',svm_mlp_layer),('l1', GaussianNB())])
    nb_cf.fit(xtr, ytr)
    
    svm_cf = StackingPipeline([('l0', nb_mlp_layer),('l1', RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5))])
    svm_cf.fit(xtr, ytr)
    return mlp_clf, nb_cf,svm_cf

# stacking for all the three algorithm
def stackingPerformanceEditor():
    nb_clf = GaussianNB()
    svm_clf = RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5)
    mlp_clff = MLPClassifier(hidden_layer_sizes=(500,500))
    label = ["NB","RF","MLP"]

    acc = StackingClassifier(classifiers=[nb_clf,svm_clf,mlp_clff], meta_classifier=svm_clf)
    acc.fit(Xtrain2, ytrain2)
    pred = accuracy_score(ytest, acc.predict(Xtest2))
    return pred

def stacking3Model(model1, model2, metamodel, xtr, ytr, xts, yts):
    model = StackingClassifier(classifiers=[model1, model2], meta_classifier=metamodel)
    train, testt = scaling(xtr, xts,MaxAbsScaler())
    model.fit(train, ytr)
    acc = accuracy_score(yts, model.predict(testt))
    predict = model.predict(testt)
    return acc, predict

tt = [[5,2,2,1,1,9,19000,4.2,1,0,1]]
result,_ = stacking3Model(GaussianNB(), RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5),MLPClassifier(hidden_layer_sizes=(500,500)),Xtrain,ytrain,Xtest,ytest)
#print(f"MLP result: {result}")
result,_ = stacking3Model(GaussianNB(), MLPClassifier(hidden_layer_sizes=(500,500)),RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5),Xtrain,ytrain,Xtest,ytest)
#print(f"SVM result: {result}")
result,_ = stacking3Model(MLPClassifier(hidden_layer_sizes=(500,500)),RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5), GaussianNB(),Xtrain,ytrain,Xtest,ytest)
#print(f"NB result: {result}")


def plotStacking3Model_NBCM(m1,m2,m3,name,savename):
    tr, ts = scaling(Xtrain, Xtest, MaxAbsScaler())
    acc, prr = stacking3Model(m1, m2, m3, tr, ytrain, ts, ytest)

    cm = confusion_matrix(ytest, prr)
    dfcm = pd.DataFrame(cm)
    plt.figure(figsize=(5.5, 4))
    sn.heatmap(dfcm, annot=True)
    plt.title(f"{name} stacking classifier\n{round(acc,4)}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(f'static/images/{savename}.png')


import matplotlib.gridspec as gridspec
import seaborn as sn 
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier

# stacking evaluating the accuracy of the dataset
def stackingDetection():
    nb_clf = GaussianNB()
    svm_clf = RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5)
    mlp_clff = MLPClassifier(hidden_layer_sizes=(500,500))
    label = ["NB", "RF","MLP"]

    metaclassifier = RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5)
    clf_list = [nb_clf, svm_clf, mlp_clff]

    clf_cv_mean = []
    clf_cv_std = []

    for clf, label in zip(clf_list, label):
        scores = cross_val_score(clf, Xtrain, ytrain, cv=3, scoring='accuracy')
        print(f"Accuracy: {round(scores.mean(), 4)} Std: {round(scores.std(), 4)} Label: {label}")
        # clf_cv_mean.append(scores.mean())
        # clf_cv_std.append(scores.std())
    bagging1 = BaggingClassifier(base_estimator=mlp_clff, n_estimators=10, max_samples=0.8, max_features=0.8)
    plt.figure()
    plot_learning_curves(Xtrain, ytrain, Xtest, ytest, bagging1, print_model=False, style='ggplot')
    plt.show()
stackingDetection()

def singleEnsembleModel(m1, m2, m3, metamodel, xtr, ytr):
    model = StackingClassifier(classifiers=[m1, m2, m3], meta_classifier=metamodel)
    model.fit(xtr, ytr)
    return model

def singleFeatureSelectionModel(test):   
    Xtt, trt = scaling(Xtrain, test, MaxAbsScaler()) 
    mlp = singleEnsembleModel(MLPClassifier(hidden_layer_sizes=(500,500)),RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5),GaussianNB(),GaussianNB(),Xtt, ytrain) 
    resul = mlp.predict(trt)
    res = lencoder_y.inverse_transform([resul])

    return res[0]

def predict_singleItem(age, classification,present_complain,location, duration, pain_sympton, swelling_symptom, nona,\
    stimulus_pain,character_pain, sensitivity, nocarousteeth,notooth_affect,facial_assymmetry,investigation,diet, aggravating_factor, progression, pain):
    age= age 
    classification=label_cls.transform([classification])
    present_complain =present_complain 
    location = location
    duration = duration
    pain_sympton = pain_sympton 
    swelling_symptom = swelling_symptom 
    nona = nona
    stimulus_pain = stimulus_pain
    character_pain = character_pain
    sensitivity = sensitivity
    nocarousteeth = nocarousteeth
    notooth_affect = notooth_affect
    facial_assymmetry = facial_assymmetry
    investigation = investigation
    diet=diet
    aggravate = label_agg.transform([aggravating_factor])
    progression = label_acs.transform([progression])
    pain = label_pain.transform([pain])
    inputTest = [[int(age), int(classification[0]),int(present_complain),int(location),int(duration),int(pain_sympton),int(swelling_symptom),int(nona),\
    int(stimulus_pain),int(character_pain),int(sensitivity), int(nocarousteeth), int(notooth_affect),\
    int(facial_assymmetry), int(investigation), int(diet), int(aggravate[0]), int(progression[0]), int(pain[0])]]
    nb, svm, mlp = fit_predict_func(Xtrain,inputTest,ytrain, MaxAbsScaler())
    stk = singleFeatureSelectionModel(inputTest)

    nbpredict = lencoder_y.inverse_transform([nb]) 
    svmpredict = lencoder_y.inverse_transform([svm])
    mlppredict = lencoder_y.inverse_transform([mlp])
    return nbpredict[0], svmpredict[0], mlppredict[0], stk  

nbpredict, svmpredict, mlppredict,test1 = predict_singleItem(5,'moderate',4,3,3,0,3,4,0,1,0,2,1,1,5, 1,'Yes','small','sharp')
# resul = singleFeatureSelectionModel(5,'middle_class',2,1,1,9,2002,5.2,'No','No','Yes')
print(test1)
print(f"NB: {nbpredict}")
print(f"RF: {svmpredict}")
print(f"MLP: {mlppredict}")
print("================================")


def singleEnsemble(test):    
    mlp, nb, svm = _StackModel_F(Xtrain, ytrain,Xtest, MaxAbsScaler())
    mlpred = lencoder_y.inverse_transform([mlp.predict(test)])
    nbpred = lencoder_y.inverse_transform([nb.predict(test)])
    svpred = lencoder_y.inverse_transform([svm.predict(test)])
    return mlpred[0], nbpred[0], svpred[0]
    

# mlpp, nbb, svmm = singleEnsemble(test1)
# print(f"MLP Ens: {mlpp}, type: {type(mlpp)}")
# print(f"Nb Ens: {nbb}, type: {type(nbb)}")
# print(f"Svm Ens: {svmm}, type: {type(svmm)}")
# nps, svmp, mps = predict_singleItem(5,'lower_class',2,1,1,9,19000,4.2,'Yes','No','Yes')

'''
nbss, svms, mlps = fit_predict_func(Xtrain,[[4.0, 2, 2.0, 0, 1, 9, 19000, 4.2, 0, 0, 0]],ytrain, MaxAbsScaler())
print(f"Naive: {nbss}")
print(f"SVM: {svms}")
print(f"MLP: {mlps}")

nbp, svp, mlpp = predict_singleItem(4,'middle_class',3,1,2,5,9700,7.0,'Yes','Yes','No')
print(f"Naive: {nbp}")
print(f"SVM: {svp}")
print(f"MLP: {mlpp}")
print(label_social.transform(['middle_class']))

mlp_train, mlp_test = stacking(clf_svm_nb, Xtrain_s, ytr, Xtest_s)
mlp_model = MLPClassifier(hidden_layer_sizes=(500, 500))
mlp_model.fit(mlp_train, ytr)
pred_mlp = mlp_model.predict(mlp_test)
print(f'Accuracy: {accuracy_score(yts, pred_mlp)}')

nb_train, nb_test = stacking(clf_svm_mlp, Xtrain_s, ytr, Xtest_s)
nb_model = GaussianNB()
nb_model.fit(nb_train, ytr)
pred_nb = nb_model.predict(nb_test)
print(f'Accuracy: {accuracy_score(yts, pred_mlp)}')
'''
'''

#clf = [RandomForestClassifSVC(kernel='rbf')ier(n_estimators=100, n_jobs=-1, criterion='gini'), GaussianNB()]
layer0 = make_stack_layer(SVC(kernel='rbf'), GaussianNB())

clf = StackingPipeline([('l0', layer0),('l1', MLPClassifier(hidden_layer_sizes=(500,500)))])
clf.fit(Xtrain, ytrain)
pred = clf.predict(Xtest)
print(accuracy_score(ytest, pred))
df = pd.read_csv(path)
df = df.to_dict(orient='records')
#for d in list(df):
 #   print(d["sex"])
 # 
'''

def perform_ensemble():
    mlp_cf,nb_cf,svm_cf = _StackModel_F(Xtrain, ytrain)
    mlppred = mlp_cf.predict(Xtest)    
    nbpred = nb_cf.predict(Xtest)    
    svmpred = svm_cf.predict(Xtest)
    return mlp_cf, nb_cf, svm_cf, mlppred, nbpred, svmpred

def perform_ensemble2():
    svm_nb_layer = make_stack_layer(RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5), GaussianNB())
    svm_mlp_layer = make_stack_layer(RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5),MLPClassifier(hidden_layer_sizes=(500,500)))
    nb_mlp_layer = make_stack_layer(GaussianNB(), MLPClassifier(hidden_layer_sizes=(500,500)))

    mlp_clf = StackingPipeline([('l0',svm_nb_layer),('l1', MLPClassifier(hidden_layer_sizes=(500,500)))])
    mlp_clf.fit(Xtrain2, ytrain2)
    mlppred = mlp_clf.predict(Xtest2)

    nb_cf = StackingPipeline([('l0',svm_mlp_layer),('l1', GaussianNB())])
    nb_cf.fit(Xtrain2, ytrain2)
    nbpred = nb_cf.predict(Xtest2)

    svm_cf = StackingPipeline([('l0', nb_mlp_layer),('l1', RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5))])
    svm_cf.fit(Xtrain2, ytrain2)
    svmpred = svm_cf.predict(Xtest2)
    return mlp_clf, nb_cf, svm_cf, mlppred, nbpred, svmpred


'''
y1, s1 = fit_predict(Xtrain, Xtest, ytrain, ytest)
y2, s2 = fit_predict(Xtrain, Xtest, ytrain, ytest, StandardScaler())
print(f"Without feature selection and no scaler: {s1}")
print(f"Without feature selection and with scaler: {s2}")

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
select = SelectFromModel(rf)
select.fit(Xtrain, ytrain)
mask = select.get_support()

Xtrain_s = select.transform(Xtrain)
Xtest_s = select.transform(Xtest)
y, s3 = fit_predict(Xtrain_s, Xtest_s, ytrain, ytest, StandardScaler())
y, s4 = fit_predict(Xtrain_s, Xtest_s, ytrain, ytest)
print(f"With feature selection and with scaler: {s3}")
print(f"With feature selection and no scaler: {s4}")

select2 = RFE(estimator=rf)
select2.fit(Xtrain, ytrain)
feature = select2.n_features_
Xtrain_s2 = select.transform(Xtrain)
Xtest_s2 = select.transform(Xtest)
y5, s5 = fit_predict(Xtrain_s2, Xtest_s2, ytrain, ytest, StandardScaler())
y6, s6 = fit_predict(Xtrain_s2, Xtest_s2, ytrain, ytest)
print(f"With feature selection and with scaler: {s5}")
print(f"Without feature selection and no scaler: {s6}")
print(f"With feature selection and no scaler: {feature}")
print(select2.get_support())
'''