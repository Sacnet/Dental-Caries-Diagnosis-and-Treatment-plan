from flask import Flask, jsonify, render_template, url_for, request
from app import process_data
import app as m; import pandas as pd 
from app import singleEnsemble, singleFeatureSelectionModel
from sklearn.metrics import precision_score, recall_score,auc, f1_score, roc_auc_score, roc_curve
from matplotlib import pyplot as plt 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix as cm, classification_report
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from app import *
app = Flask(__name__)


data = list(pd.read_csv(m.path))
from sklearn.metrics import accuracy_score
#Xtr, Xts, ytr, yts = data_normalization()

# ploting the roc curves

plotStacking3Model_NBCM(MLPClassifier(hidden_layer_sizes=(500,500)),RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5),GaussianNB(),"Naive Bayes","nb")
plotStacking3Model_NBCM(MLPClassifier(hidden_layer_sizes=(500,500)),GaussianNB(),RandomForestClassifier(n_estimators=100, max_depth=400, random_state=5),"RF","rf")
plotStacking3Model_NBCM(GaussianNB(),SVC(kernel='rbf'),MLPClassifier(hidden_layer_sizes=(500,500)),"MLP","mlp")

def generate_metrics(model, xts, yts):
        acc = accuracy_score(yts, model.predict(xts))
        pre = precision_score(yts, model.predict(xts), average='micro')
        rec = recall_score(yts, model.predict(xts), average='micro')
        f1 = f1_score(yts, model.predict(xts), average='micro')
        return acc, pre, rec, f1


@app.route('/')
@app.before_first_request
def index():

    return render_template('index.html')

@app.route('/loaddata')
def loaddata():
    df1 = pd.read_csv(m.path)
    df = df1.to_dict(orient='records')
    return render_template('loadata.html',data=data, df=list(df),
    featurenames=len(df1.count()))

@app.route('/loadfeatures')
def loadfeatures():
    feature,_ = process_data(m.path)
    d = feature.columns
    feature = feature.to_dict(orient='records')    
    return render_template('loadextra.html',data = list(d), feature=feature)

@app.route('/loadfeatureset')
def loadfeatureset():
    X,_ = m.imputation_process()
    feature, _ = process_data(m.path)
    d = feature.columns
    #df = pd.DataFrame(X)
    featureset = X.to_dict(orient='records')    
    return render_template('loadfeatureset.html', dataset=d, featureset=featureset)

@app.route('/alogrithm_base_fs')
def alogrithm_base_fs():
    
    return jsonify({})

@app.route('/testcases')
def testcases():
        return render_template('testcases.html')


@app.route('/systemdeotesting')
def systemdeotesting():
        age = request.args.get('age')
        classification = request.args.get('classification')
        present_complain = request.args.get('present_complain')
        location = request.args.get('location')
        duration = request.args.get('duration')
        pain_sympton = request.args.get('pain_sympton')
        swelling_symptom = request.args.get('swelling_symptom')
        nona = request.args.get('nona')
        stimulus_pain = request.args.get('stimulus_pain')
        pain_character = request.args.get('pain_character')
        sensitivity = request.args.get('sensitivity')
        
        nocarousteeth = request.args.get('nocarousteeth')
        notooth_affect = request.args.get('notooth_affect')
        facial_assymmetry = request.args.get('facial_assymmetry')
        investigation = request.args.get('investigation')
        diet = request.args.get('diet')
        aggravating_factor = request.args.get('aggravating_factor')
        progression = request.args.get('progression')
        pain = request.args.get('pain')

        nbpre, svmpre, mlppre,stk = m.predict_singleItem(age,classification,present_complain,location,duration, pain_sympton,\
                swelling_symptom, nona, stimulus_pain,pain_character,sensitivity,nocarousteeth,notooth_affect,facial_assymmetry,\
                   investigation, diet, aggravating_factor, progression, pain)   
        if stk == 'acute':
                tre='EXTRACTION, SPACE MAINTANANCE'
        elif stk == 'reversible':
                tre = 'VITAL PULPOTOMY'               
        elif stk == 'enamel':
                tre ='GIC'
        elif stk == 'dentine':
                tre = 'GIC'
        else:
                tre = 'EXTRACTION'
        if age < nocarousteeth:
                risk='Disease Rate is High'
        elif age == nocarousteeth:
                risk='Disease Rate is still Moderate'
        else:
                risk='Disease rate is still Low'
        # 5,'upper_class',2,1,1,9,19000,4.2,'No','No','Yes'
        return jsonify({"nb_res":nbpre,"svm_res":svmpre,"mlp_res":mlppre, "stackres":stk, "trepla":tre, "risk":risk})

@app.route('/systemdemo')
def systemdemo():
    return render_template('systemdemo.html')        
#metrics for algorithms with FS
nbpred, svmpred, mlpred = m.perform_ml_matrix()
acc_nb, prec_nb, rec_nb, fms_nb, _ = m.perform_metrics(nbpred)
acc_svm, prec_svm, rec_svm, fms_svm, _ = m.perform_metrics(svmpred)
acc_mlp, prec_mlp, rec_mlp, fms_mlp, _ = m.perform_metrics(mlpred)

#metrics for algorithms with NFS
nbpred_nf, svmpred_nf, mlpred_nf = m.perform_ml_matrix2()
acc_nb_nf, prec_nb_nf, rec_nb_nf, fms_nb_nf,_ = m.perform_metrics2(nbpred_nf)
acc_svm_nf, prec_svm_nf, rec_svm_nf, fms_svm_nf, _ = m.perform_metrics2(svmpred_nf)
acc_mlp_nf, prec_mlp_nf, rec_mlp_nf, fms_mlp_nf, _ = m.perform_metrics2(mlpred_nf)

#metrics for stacking with FS
ml_s, nb_s, sm_s= m.final_stacking()
acc_nb_s, prec_nb_s, rec_nb_s, fms_nb_s, _ = m.perform_metrics(nb_s)
acc_svm_s, prec_svm_s, rec_svm_s, fms_svm_s, _ = m.perform_metrics(sm_s)
acc_mlp_s, prec_mlp_s, rec_mlp_s, fms_mlp_s, _ = m.perform_metrics(ml_s)

ml_s_nf, nb_s_nf, sm_s_nf = m.final_stacking2()
acc_nb_s_nf, prec_nb_s_nf, rec_nb_s_nf, fms_nb_s_nf,_ = m.perform_metrics2(nb_s_nf)
acc_svm_s_nf, prec_svm_s_nf, rec_svm_s_nf, fms_svm_s_nf, _ = m.perform_metrics2(sm_s_nf)
acc_mlp_s_nf, prec_mlp_s_nf, rec_mlp_s_nf, fms_mlp_s_nf,_ = m.perform_metrics2(mlpred_nf)


@app.route('/algorithm_process_FS')
def systemevaluation():        
        nbacc, nbprec, nbrec, nbfms = round(acc_nb, 4) * 100, round(prec_nb, 4)*100, \
                round(rec_nb, 4)*100, round(fms_nb, 4)*100
        svmacc, svmprec, svmrec, svmfms = round(acc_svm,4)*100, round(prec_svm,4)*100,\
                round(rec_svm, 4)*100, round(fms_svm,4)*100
        mlpacc, mlpprec, mlprec, mlpfms = round(acc_mlp,4)*100,round(prec_mlp,4)*100,round(rec_mlp,4)*100,\
                round(fms_mlp,4) * 100

        return jsonify({'acc_nb':round(nbacc,4), 'prec_nb':round(nbprec,4), 'rec_nb':round(nbrec,4), 'fms_nb':round(nbfms,4),\
                'acc_svm':round(svmacc,4),'prec_svm':round(svmprec,4),'rec_svm':round(svmrec,4),'fms_svm':round(svmfms, 4),\
                        'acc_mlp':round(mlpacc,4),'prec_mlp':round(mlpprec, 4),'rec_mlp':round(mlprec,4),'fms_mlp':round(mlpfms,4)})

@app.route('/algorithm_process_NFS')
def systemevaluation_NFS():        
        nbacc, nbprec, nbrec, nbfms = round(acc_nb_nf, 4) * 100, round(prec_nb_nf, 4)*100, \
                round(rec_nb_nf, 4)*100, round(fms_nb_nf, 4)*100
        svmacc, svmprec, svmrec, svmfms = round(acc_svm_nf,4)*100, round(prec_svm_nf,4)*100,\
                round(rec_svm_nf, 4)*100, round(fms_svm_nf,4)*100
        mlpacc, mlpprec, mlprec, mlpfms = round(acc_mlp_nf,4)*100,round(prec_mlp_nf,4)*100,round(rec_mlp_nf,4)*100,\
                round(fms_mlp_nf,4) * 100

        return jsonify({'acc_nb':round(nbacc,4), 'prec_nb':round(nbprec,4), 'rec_nb':round(nbrec,4), 'fms_nb':round(nbfms,4),\
                'acc_svm':round(svmacc,4),'prec_svm':round(svmprec,4),'rec_svm':round(svmrec,4),'fms_svm':round(svmfms, 4),\
                        'acc_mlp':round(mlpacc,4),'prec_mlp':round(mlpprec, 4),'rec_mlp':round(mlprec,4),'fms_mlp':round(mlpfms,4)})



@app.route('/stackingmodel_FS')
def stackingmodel_FS():
        nbacc_s, nbprec_s, nbrec_s, nbfms_s = round(acc_nb_s, 4)*100, round(prec_nb_s,4)*100,round(rec_nb_s,4)*100, \
                round(fms_nb_s,4)*100
        svmacc_s, svmprec_s, svmrec_s, svmfms_s = round(acc_svm_s,4)*100,round(prec_svm_s,4)*100,round(rec_svm_s,4)*100,\
                round(fms_svm_s,4)*100
        mlpacc_s,mlpprec_s,mlprec_s, mlpfms_s = round(acc_mlp_s,4)*100,round(prec_mlp_s,4)*100,round(rec_mlp_s,4) * 100,\
                round(fms_mlp_s, 4) * 100
        return jsonify(
                {'nbacc_s':round(nbacc_s,4), 'nbprec_s':round(nbprec_s,4), 'nbrec_s':round(nbrec_s,4), 'nbfms_s':round(nbfms_s,4),\
                'svmacc_s':round(svmacc_s,4),'svmprec_s':round(svmprec_s,4),'svmrec_s':round(svmrec_s,4),'svmfms_s':round(svmfms_s, 4),\
                'mlpacc_s':round(mlpacc_s,4),'mlpprec_s':round(mlpprec_s, 4),'mlprec_s':round(mlprec_s,4),'mlpfms_s':round(mlpfms_s,4)}
        )

@app.route('/stackingmodel_NFS')
def stackingmodel_NFS():
        nbacc_s, nbprec_s, nbrec_s, nbfms_s = round(acc_nb_s_nf, 4)*100, round(prec_nb_s_nf,4)*100,round(rec_nb_s_nf,4)*100, \
                round(fms_nb_s_nf,4)*100
        svmacc_s, svmprec_s, svmrec_s, svmfms_s = round(acc_svm_s_nf,4)*100,round(prec_svm_s_nf,4)*100,round(rec_svm_s_nf,4)*100,\
                round(fms_svm_s_nf,4)*100
        mlpacc_s,mlpprec_s,mlprec_s, mlpfms_s = round(acc_mlp_s_nf,4)*100,round(prec_mlp_s_nf,4)*100,round(rec_mlp_s_nf,4) * 100,\
                round(fms_mlp_s_nf, 4) * 100
        return jsonify(
                {'nbacc_s':round(nbacc_s,4), 'nbprec_s':round(nbprec_s,4), 'nbrec_s':round(nbrec_s,4), 'nbfms_s':round(nbfms_s,4),\
                'svmacc_s':round(svmacc_s,4),'svmprec_s':round(svmprec_s,4),'svmrec_s':round(svmrec_s,4),'svmfms_s':round(svmfms_s, 4),\
                'mlpacc_s':round(mlpacc_s,4),'mlpprec_s':round(mlpprec_s, 4),'mlprec_s':round(mlprec_s,4),'mlpfms_s':round(mlpfms_s,4)}
        ) 
        
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd 
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,\
RobustScaler, StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from matplotlib import pyplot as plt 
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sn 
from flask import url_for

#dataset = pd.read_csv('datasets/7-link-content-obvious.csv')
dataset = 'datasets/Link-features-only-new.csv'
testing = 'datasets/testing_linkbase.csv'

#dataset = pd.read_csv('datasets/7-link-content-obvious.csv')
dataset = 'datasets/Link-features-only-new.csv'
testing = 'datasets/testing_linkbase.csv'

def get_processdata(data):
    dataset = pd.read_csv(data)
    #dataset = dataset.reset_index()
    dataset['Class'] = dataset['Class'].map({'non-spam':'ham','spam':'spam'})
    dataset['Class'] = dataset['Class'].map({'ham':0,'spam':1})
    X = dataset.iloc[:,:-1]
    y = dataset.iloc[:,-1]
#     for i in X.columns:
#         X[i] = X[i].fillna(np.mean(X[i]))
#   dataset = pd.read_csv(data)
#     #dataset = dataset.reset_index()
#     dataset['Class'] = dataset['Class'].map({'non-spam':'ham','spam':'spam'})
#     dataset['Class'] = dataset['Class'].map({'ham':0,'spam':1})
#     X = dataset.iloc[:,:-1]
#     y = dataset.iloc[:,-1]
    for i in X.columns:
        X[i] = X[i].fillna(np.mean(X[i]))
        

@app.route("/systemeval")
def systemeval():        

        return render_template('systemeval.html')

@app.route('/confussion_matrix')
def confussion_matrix():

    return render_template('confussion_matrix.html', cmmlp='static/images/mlp.png',\
            cmnb='static/images/nb.png', cmsvm='static/images/rf.png')
if __name__ == "__main__":
    app.run(debug=True)
