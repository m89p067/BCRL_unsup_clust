import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import joblib
import json
from scipy.stats import mannwhitneyu,chi2_contingency,fisher_exact
from sklearn.utils.estimator_checks import check_estimator
from sklearn.inspection import permutation_importance
from statsmodels.graphics.mosaicplot import mosaic
from math import exp,log,sqrt
import researchpy as rp
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


n_clust=3
cartella_princ='D:/SABCS 2022/CODE/Using_'+str(n_clust)+'/'


def make_the_plot(dato,eti,stringa_filename,cartella_filename,eti2):
    group=np.asarray(eti)
    cdict={'A':'red','B':'forestgreen','C':'royalblue','D':'lime','E':'orchid','F':'teal','G':'silver','O':'black'}
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
num1=str(2290)
num2=str(4458)
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
mapper=my_dati['arr_0']
etich1=my_dati['arr_1']
sil_scoring=my_dati['arr_2']
etich2=my_dati['arr_3']
classi=np.load('D:/SABCS 2022/CODE/bcrl_classi.npy')
dictionary = {0:'no', 1:'yes'}
bcrl=list(map(dictionary.get, classi.tolist()))

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
X2=data[["BREAST SURGERY","SIDE","Ki_67","CT","TAXANE BASED CT","HT","TTZ (her2+)","LVI",
         "EXTRACAPSULAR EXTENSION (ECE)","ER","HER2","NCD","PR"]]
X2["BREAST SURGERY"] = X2["BREAST SURGERY"].subtract(1)
X2["SIDE"] = X2["SIDE"].subtract(1)
X2["Ki_67"] = X2["Ki_67"].subtract(1)
print('Ordinal dataframe is ',X1.shape[0],' x ',X1.shape[1])
print('Binary dataframe is ',X2.shape[0],' x ',X2.shape[1])
full_names=["NR METASTATIC LN","TOTAL NR DISSECTED LN","RT TYPE","HR DRUG 1","HISTOTYPE","G.1","T",
         "N EFF","MOLECULAR SUBTYPE","AGE GROUP","BMI GROUP"]+["BREAST SURGERY","SIDE","Ki_67","CT","TAXANE BASED CT","HT","TTZ (her2+)","LVI",
         "EXTRACAPSULAR EXTENSION (ECE)","ER","HER2","NCD","PR"]
ord_names=["NR METASTATIC LN","TOTAL NR DISSECTED LN","RT TYPE","HR DRUG 1","HISTOTYPE","G.1","T",
         "N EFF","MOLECULAR SUBTYPE","AGE GROUP","BMI GROUP"]
bin_names=["BREAST SURGERY","SIDE","Ki_67","CT","TAXANE BASED CT","HT","TTZ (her2+)","LVI",
         "EXTRACAPSULAR EXTENSION (ECE)","ER","HER2","NCD","PR"]
print('----------------All data loaded')
out_tab0=pd.crosstab(np.asarray(bcrl),etich1,rownames=['BCRL'],colnames=['Clusters'],margins=True)
#print(out_tab0)
print('\n')
def prepara_B_vs_All(dato):
    mydictionary={'A':'O','B':'B','C':'O'}
    target2=list(map(mydictionary.get, dato))
    return target2
print('*************************************************************************')
cluster=prepara_B_vs_All(etich1)
out_tabz=pd.crosstab(np.asarray(bcrl),np.asarray(cluster),rownames=['BCRL'],colnames=['Clusters'],margins=True)
#print(out_tabz)
Xord=X1.to_numpy()
Xbin=X2.to_numpy()
bcrl_bin=classi
df_tmp=pd.DataFrame({'BCRL':np.asarray(bcrl),'Cluster':np.asarray(etich1)})
crosstab1, res1 = rp.crosstab(df_tmp['BCRL'],df_tmp['Cluster'] , test= "g-test",
                            cramer_correction=True)
print(crosstab1)
print(res1)
print('----------------------------------------------')
df_tmp=pd.DataFrame({'BCRL':np.asarray(bcrl),'Cluster':np.asarray(cluster)})
crosstab2, res2 = rp.crosstab(df_tmp['BCRL'],df_tmp['Cluster'] , test= "g-test",
                            cramer_correction=True)
print(crosstab2)
print(res2)
