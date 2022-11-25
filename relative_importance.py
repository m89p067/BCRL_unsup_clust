import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE,SMOTEN
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import RFECV
import seaborn as sns
import joblib
import math
from sklearn.utils.estimator_checks import check_estimator
from sklearn.inspection import permutation_importance
print('Load saved models created previously by analizzatore_features')
print('and create the graphs')
n_clust=3
cartella_princ='D:/BCRL_data/Using_'+str(n_clust)+'/'
cartella_salvataggi='D:/BCRL_data/Saving_ALT_'+str(n_clust)+'/'
if not os.path.isdir(cartella_salvataggi):
    os.makedirs(cartella_salvataggi)
    print("created folder : ", cartella_salvataggi)
else:
    print(cartella_salvataggi, "folder already exists.")
def make_the_plot(dato,eti,stringa_filename,cartella_filename,eti2):
    group=np.asarray(eti)
    cdict={'A':'red','B':'forestgreen','C':'royalblue','D':'lime','E':'orchid','F':'teal','G':'silver','O':'black','no bcrl':'orange','bcrl':'darkcyan'}
    fig, ax = plt.subplots()
    
    for indice, g in enumerate(group):
        if eti2[indice]=='no bcrl':
            ax.scatter(dato[indice,0], dato[indice,1], c = cdict[g],  s = 30)
        else:
            ax.scatter(dato[indice,0], dato[indice,1], c = cdict[g],  s = 30,marker='x')
    ax.legend(np.unique(group),loc ="best")
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    
    plt.savefig(cartella_filename+stringa_filename,dpi=300,bbox_inches='tight')
    plt.close()

    
do_dense=True
if do_dense:
    cartelle_out=[cartella_princ+'inters_dens/',cartella_princ+'union_dens/',cartella_princ+'contr_dens/']
else:
    cartelle_out=[cartella_princ+'inters/',cartella_princ+'union/',cartella_princ+'contr/']
num1=str(2290)#3165
num2=str(4458)#8548
tipo_operaz=0
if tipo_operaz==0:
    my_dati=np.load(cartelle_out[tipo_operaz]+'/inters_'+num1+'_'+num2+'.npz')
    stringa='inters_'+num1+'_'+num2
elif tipo_operaz==1:
    my_dati=np.load(cartelle_out[tipo_operaz]+'/union_'+num1+'_'+num2+'.npz')
    stringa='union_'+num1+'_'+num2
elif tipo_operaz==2:
    my_dati=np.load(cartelle_out[tipo_operaz]+'/contr_'+num1+'_'+num2+'.npz')
    stringa='contr_'+num1+'_'+num2
if do_dense==True:
    stringa2='_dense'
else:
    stringa2='_no_dense'
the_mapper=my_dati['arr_0']
etich1=my_dati['arr_1']
sil_scoring=my_dati['arr_2']
etich2=my_dati['arr_3']
classi=np.load('D:/BCRL_data/bcrl_classi.npy')
dictionary = {0:'no bcrl', 1:'bcrl'}
the_bcrl=list(map(dictionary.get, classi.tolist()))

dataMI=pd.read_csv('dataMILANO.csv', index_col=False)
dataNO=pd.read_csv('dataNOVARA.csv', index_col=False)
data=pd.concat([dataMI, dataNO], axis=0)
print('After merging:',data.shape[0],' patients')

data.drop('AXILLA SURGERY', axis=1, inplace=True)
data['AGE GROUP'],age_bins = pd.cut(data['AGE AT DIAGNOSIS'],bins= 10, labels=False,retbins=True)
data['BMI GROUP'] ,bmi_bins= pd.cut(data['BMI'], bins=10, labels=False,retbins=True)
data[['AGE GROUP', 'BMI GROUP']] = data[['AGE GROUP', 'BMI GROUP']].astype(int)
data.drop('AGE AT DIAGNOSIS', axis=1, inplace=True)
data.drop('BMI', axis=1, inplace=True)
for c in data.columns:
   print( "---- %s ---" % c)
   print( data[c].value_counts())
data.drop('BCRL', axis=1, inplace=True)

X1=data[["NR METASTATIC LN","TOTAL NR DISSECTED LN","RT TYPE","HR DRUG 1","HISTOTYPE","G.1","T",
         "N EFF","MOLECULAR SUBTYPE","AGE GROUP","BMI GROUP"]]
X2=data[["BREAST SURGERY","SIDE","Ki_67","TAXANE BASED CT","HT","TTZ (her2+)","LVI",
         "EXTRACAPSULAR EXTENSION (ECE)","ER","HER2","NCD","PR"]]
X2["BREAST SURGERY"] = X2["BREAST SURGERY"].subtract(1)
X2["SIDE"] = X2["SIDE"].subtract(1)
X2["Ki_67"] = X2["Ki_67"].subtract(1)
print('Ordinal dataframe is ',X1.shape[0],' x ',X1.shape[1])
print('Binary dataframe is ',X2.shape[0],' x ',X2.shape[1])
full_names=["NR METASTATIC LN","TOTAL NR DISSECTED LN","RT TYPE","HR DRUG 1","HISTOTYPE","G.1","T",
         "N EFF","MOLECULAR SUBTYPE","AGE GROUP","BMI GROUP"]+["BREAST SURGERY","SIDE","Ki_67","TAXANE BASED CT","HT","TTZ (her2+)","LVI",
         "EXTRACAPSULAR EXTENSION (ECE)","ER","HER2","NCD","PR"]
ord_names=["NR METASTATIC LN","TOTAL NR DISSECTED LN","RT TYPE","HR DRUG 1","HISTOTYPE","G.1","T",
         "N EFF","MOLECULAR SUBTYPE","AGE GROUP","BMI GROUP"]
bin_names=["BREAST SURGERY","SIDE","Ki_67","TAXANE BASED CT","HT","TTZ (her2+)","LVI",
         "EXTRACAPSULAR EXTENSION (ECE)","ER","HER2","NCD","PR"]
print('----------------All data loaded')

def evaluate_model(my_data, my_labels, model):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
    # evaluate model
    scores = cross_val_score(model, my_data, my_labels, scoring='balanced_accuracy', cv=cv, n_jobs=-1)
    return scores
def get_indexes(test_list,elem):
    res_list = []
    for i in range(0, len(test_list)) :
        if test_list[i] == elem :
            res_list.append(i)
    return res_list
def remove_from_list(test_list,idx_list):
    res = []
    for idx, ele in enumerate(test_list):     
        # checking if element not present in index list
        if idx not in idx_list:
            res.append(ele)
    return res

 

def calcolatore(dati,tipi_stringhe,la_stringa,tipo_lavoro,feature_names,do_bal=False):
    if tipo_lavoro==0:
        mystringa='_Three_Clusters'
    elif tipo_lavoro==1:
        mystringa='_B_vs_All'
    else:
        mystringa='_BCRL'
    mystringa='_'+tipi_stringhe+mystringa    
    print(mystringa[1::])
    if tipo_lavoro==0:
        print('Keeping the three clusters')
        target=dati[1]
    elif tipo_lavoro==1:
        mydictionary={'A':'O','B':'B','C':'O'}
        target=list(map(mydictionary.get, dati[1]))
    else:
        target=dati[2]
    X=dati[0]    
    new_bcrl=dati[2]
    new_mapper=dati[3]

    print(tipi_stringhe+' dataset shape:',X.shape)
    model = DummyClassifier(strategy='stratified')
    dicts2 = {}
    # evaluate the model
    scores = evaluate_model(X, target, model)
    # summarize performance
    print('DUMMY Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    dicts2['DUMMY']=str(round(np.mean(scores),3))+"+"+str(round(np.std(scores),3))
    # define models to test

    
    if do_bal:
      var_classificare2='over'
      print('OVERSAMPLING by SMOTE')
    else:
      var_classificare2='orig'
      print('Keeping original counts, without SMOTE')
    # evaluate each model
    filename_clf=cartella_salvataggi+'clf_'+la_stringa+'_'+mystringa+'_'+var_classificare2+'.pkl'
    clf=joblib.load(filename_clf) # HERE LOADS THE ML MODELS PREVIOUSLY CREATED


    if do_bal:
        if 'ORD' in tipi_stringhe:
            sm=SMOTEN(k_neighbors=5,sampling_strategy='all',random_state=0)
            X_res, y_res = sm.fit_resample(X, target)
            clf.fit(X_res, y_res)
        elif 'BIN' in tipi_stringhe:
            rus = RandomUnderSampler(random_state=0, replacement=False)
            X_res, y_res = rus.fit_resample(X, target)
            clf.fit(X_res, y_res)
        np.savez(cartella_salvataggi+'rebal_'+la_stringa+'_'+mystringa+'_RelImp', X_res, y_res)
    else:
        X_res=X.copy()
        y_res=target.copy()
        clf.fit(X_res, y_res)
    
    
    min_features_to_select = 1  # Minimum number of features to consider
    
    if hasattr(clf, 'feature_importances_'):
        rfecv = RFECV(
            estimator=clf,
            step=1,
            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10), #StratifiedKFold(5),
            scoring="balanced_accuracy",
            min_features_to_select=min_features_to_select,)
    else:
        print('Wanna see my trick?')
        
        def my_coeff_(clf):
            return pow(math.e,clf.coef_) # For Logistic Regression
        def my_feature_importances_(clf):
            return np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)
        if hasattr(clf, 'coef_'):
            rfecv = RFECV(
            estimator=clf,
            step=1,
            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10), #StratifiedKFold(5),
            scoring="balanced_accuracy",
            min_features_to_select=min_features_to_select,importance_getter=my_coeff_)            
        else:
            rfecv = RFECV(
            estimator=clf,
            step=1,
            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10), #StratifiedKFold(5),
            scoring="balanced_accuracy",
            min_features_to_select=min_features_to_select,importance_getter=my_feature_importances_)            
    rfecv.fit(X_res, y_res)
    maschera=rfecv.support_
    feature_importances =rfecv.cv_results_['mean_test_score']
    #pdb.set_trace()
    f_type='RFECV'
    print("Optimal number of features : %d" % rfecv.n_features_)
    scores = evaluate_model(X, target,clf)
    print('Average classifier performance:',np.mean(scores),'+/-',np.std(scores))
    do_graphs=True
    if do_graphs==True:
        #GRAFICI
        tmp_names=[x1 for x1, y1 in zip(feature_names, maschera) if y1 == True]
        data_df=pd.DataFrame({'Feature':tmp_names,'Importance':feature_importances[maschera]})
        gli_score=data_df[['Importance']].to_numpy()
        sns.barplot(x=data_df['Importance'],y=data_df['Feature'])
        plt.axvline(x = np.mean(scores), color = 'black', linewidth=2,linestyle='--')
        plt.xlim(np.min(gli_score)-(np.min(gli_score)*0.05), np.mean(scores)+(np.mean(scores)*0.025))
        plt.xlabel('Balanced accuracy')
        plt.savefig(cartella_salvataggi+la_stringa+'_feat_'+mystringa+'_'+f_type+'_'+var_classificare2+'RelImp.png', bbox_inches='tight',dpi=300)
        plt.close()

        sns.barplot(x=data_df['Importance'],y=data_df['Feature'])
        plt.axvline(x = np.mean(scores), color = 'black', linewidth=2,linestyle='--')
        plt.xlim(np.min(gli_score)-(np.min(gli_score)*0.05), np.mean(scores)+(np.mean(scores)*0.025))
        plt.xlabel('Balanced accuracy')
        plt.savefig(cartella_salvataggi+la_stringa+'_featur_'+mystringa+'_'+f_type+'_'+var_classificare2+'RelImp.png', bbox_inches='tight',dpi=300)
        plt.close()

        porcent = list(100.*data_df['Importance']/data_df['Importance'].sum()  )
        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

        recipe = list(data_df['Feature'])
        data = porcent

        wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                  bbox=bbox_props, zorder=0, va="center")

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                        horizontalalignment=horizontalalignment, **kw)

        #ax.set_title("Matplotlib bakery: A donut")
        plt.savefig(cartella_salvataggi+la_stringa+'_donut_'+mystringa+'_'+f_type+'_'+var_classificare2+'RelImp.png', bbox_inches='tight',dpi=300)
        plt.close()


        # define Seaborn color palette to use
        palette_color = sns.color_palette('bright')

        # plotting data on chart
        out =plt.pie(porcent,  colors=palette_color, startangle=90, radius=1.2)
        patches=out[0]
        texts=out[1]
        #n_text=out[2]
        labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(recipe, porcent)]

        sort_legend = True
        if sort_legend:
            patches, labels, dummy =  zip(*sorted(zip(patches, labels, porcent),key=lambda recipe: recipe[2], reverse=True))

        plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.),fontsize=8)

        plt.savefig(cartella_salvataggi+la_stringa+'_torta_'+mystringa+'_'+f_type+'_'+var_classificare2+'RelImp.png', bbox_inches='tight',dpi=300)
        # displaying chart
        plt.close()
    make_the_plot(new_mapper,target,la_stringa+stringa2+'_'+mystringa+'_'+var_classificare2+'RelImp.png',cartella_salvataggi,new_bcrl)
    if tipo_lavoro==0 or tipo_lavoro==1:
        out_tab=pd.crosstab(np.asarray(new_bcrl),np.asarray(target),rownames=['BCRL'],colnames=['Clusters'],margins=True)
        out_tab2=pd.crosstab(np.asarray(new_bcrl),np.asarray(target),rownames=['BCRL'],colnames=['Clusters'],margins=True, normalize='all')
        print(out_tab)
        print(out_tab2)
        out_tab.to_csv(cartella_salvataggi+'crosstab_'+la_stringa+'_'+mystringa+'_'+var_classificare2+'RelImp.csv')
        out_tab2.to_csv(cartella_salvataggi+'crosstab_freq_'+la_stringa+'_'+mystringa+'_'+var_classificare2+'RelImp.csv')
    else:
        print('Cross-tabs not saved...')
    print('----------------------------------')

X_full = pd.concat([X1, X2], axis=1,ignore_index=True).to_numpy()
tipi_stringhe=['Full','ORD','BIN']
#tipo_lavoro: "0" è Three original clusters mentre "1" è B vs ALL (aka O)

# 2 Clusters
calcolatore([X_full,etich1,the_bcrl,the_mapper],tipi_stringhe[0],stringa+stringa2,1,full_names) # B vs ALL
#calcolatore([X1.to_numpy(),etich1,the_bcrl,the_mapper],tipi_stringhe[1],stringa+stringa2,1,ord_names) # B vs ALL ORD
#calcolatore([X2.to_numpy(),etich1,the_bcrl,the_mapper],tipi_stringhe[2],stringa+stringa2,1,bin_names) # B vs ALL BIN

# Three Clusters
calcolatore([X_full,etich1,the_bcrl,the_mapper],tipi_stringhe[0],stringa+stringa2,0,full_names) # B vs ALL
#calcolatore([X1.to_numpy(),etich1,the_bcrl,the_mapper],tipi_stringhe[1],stringa+stringa2,0,ord_names) # Three Clusters
#calcolatore([X2.to_numpy(),etich1,the_bcrl,the_mapper],tipi_stringhe[2],stringa+stringa2,0,bin_names) # Three Clusters


#BCRL as labels to predict
#calcolatore([X1.to_numpy(),etich1,the_bcrl,the_mapper],tipi_stringhe[1],stringa+stringa2,2,ord_names,do_bal=True) # predizione su BCRL
#calcolatore([X2.to_numpy(),etich1,the_bcrl,the_mapper],tipi_stringhe[2],stringa+stringa2,2,bin_names,do_bal=True) # predizione su BCRL
