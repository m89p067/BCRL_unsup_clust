import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import joblib
import json
from scipy.stats import mannwhitneyu
from sklearn.utils.estimator_checks import check_estimator
from sklearn.inspection import permutation_importance
from statsmodels.graphics.mosaicplot import mosaic
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
cartella_princ='D:/BCRL_data/Using_'+str(n_clust)+'/'
cartella_salvataggi='D:/BCRL_data/Stats_'+str(n_clust)+'/'
if not os.path.isdir(cartella_salvataggi):
    os.makedirs(cartella_salvataggi)
    print("created folder : ", cartella_salvataggi)
else:
    print(cartella_salvataggi, "folder already exists.")
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

    
do_dense=False
if do_dense:
    cartelle_out=[cartella_princ+'inters_dens/',cartella_princ+'union_dens/',cartella_princ+'contr_dens/']
else:
    cartelle_out=[cartella_princ+'inters/',cartella_princ+'union/',cartella_princ+'contr/']
num1=str(3165)
num2=str(8548)
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
dictionary = {0:'no', 1:'yes'}
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
out_tab0=pd.crosstab(np.asarray(the_bcrl),etich1,rownames=['BCRL'],colnames=['Clusters'],margins=True)
print(out_tab0)
print('\n')
def prepara_AB(dati):
    indici=get_indexes(dati[2].tolist(),'C')
    Xordinali= np.delete(dati[0], indici, axis=0)
    Xbinarie= np.delete(dati[1], indici, axis=0)
    new_bcrl=remove_from_list(dati[3],indici)
    new_mapper = np.delete(dati[4], indici, axis=0)
    target2=np.delete(dati[2],indici)
    target3=np.delete(dati[5],indici)
    if Xordinali.shape[0]==len(new_bcrl)==new_mapper.shape[0]==target2.shape[0]==Xbinarie.shape[0]==len(target3):
        print('Sanity check passed: all C elements were removed correctly')
    else:
        print('We have a problem here...')
    return Xordinali,Xbinarie,new_bcrl,new_mapper,target2,target3
def prepara_C_vs_All(dato):
    mydictionary={'A':'O','B':'O','C':'C'}
    target2=list(map(mydictionary.get, dato))
    return target2

Xord,Xbin,bcrl,mapper,cluster,bcrl_bin=prepara_AB([X1.to_numpy(),X2.to_numpy(),etich1,the_bcrl,the_mapper,classi])
#targetC=prepara_C_vs_All(etich1)

def annot_heatmap(la_stringa,dat1,dat2,do_norm=False):
    if do_norm==False:
        dat1=dat1.astype(int)
        dat2=dat2.astype(int)
    else:
        dat1=np.around(dat1*100,decimals=2)
        dat2=np.around(dat2*100,decimals=2)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    if do_norm==False:
        fig.suptitle(la_stringa+': cluster A vs B')
    else:
        fig.suptitle(la_stringa+': cluster A vs B [%]')
    im1 = ax1.imshow(dat1,cmap="winter")
    im2 = ax2.imshow(dat2,cmap="winter")
    # Show all ticks and label them with the respective list entries
    ax1.set_xticks([0.0,1.0])
    ax1.set_yticklabels([0,1])
    ax1.set_yticks([0.0,1.0])
    ax1.set_xticklabels(['no','yes'])

    ax2.set_xticks([0.0,1.0])
    ax2.set_yticklabels([0,1])
    ax2.set_yticks([0.0,1.0])
    ax2.set_xticklabels(['no','yes'])
    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    #plt.setp(ax1.get_yticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    #plt.setp(ax2.get_yticklabels(), rotation=90, ha="right",rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(2):
        for j in range(2):
            text1 = ax1.text(j, i, dat1[i, j],ha="center", va="center", color="black")
            text2 = ax2.text(j, i, dat2[i, j],ha="center", va="center", color="black")
    ax1.set_title("Cluster A")
    ax2.set_title("Cluster B")
    ax1.set_ylabel(la_stringa)
    ax1.set_xlabel("BCRL")
    ax2.set_ylabel(la_stringa)
    ax2.set_xlabel("BCRL")
    fig.tight_layout()
    if do_norm==False:
        plt.savefig(cartella_salvataggi+la_stringa+'_heatmap_AB'+'.png', bbox_inches='tight',dpi=300)
    else:
        plt.savefig(cartella_salvataggi+la_stringa+'_norm_heatmap_AB'+'.png', bbox_inches='tight',dpi=300)
    plt.close()
def conv_cross_to_numpy(tabella):
    array=np.zeros([2, 2])    
    for i in range(2):
        for j in range(2):
            array[j,i]=tabella.iloc[j,i]
    if array[0,0]==tabella.iloc[0,0] and array[0,1]==tabella.iloc[0,1] and array[1,0]==tabella.iloc[1,0] and array[1,1]==tabella.iloc[1,1]:
        print('Safe conversion...')
    else:
        print('Error in the conversion!')
        array[:]=np.NaN
    return array
def ripara_tab(tabella):
    array=np.zeros([2, 2])
    indici1=tabella.index.to_numpy()
    col1=tabella.columns.to_numpy()
    if len(indici1)==1:
        quale=indici1[0]
        array[quale,:]=tabella.to_numpy()
    if len(col1)==1:
        quale=col1[0]
        array[:,quale]=tabella.to_numpy()
    return array
do_mosaic=False
if do_mosaic:
    #Mosaic plot on BINARY A vs B
    for i in range(Xbin.shape[1]):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(bin_names[i]+': cluster A vs B')
        dati_bin=pd.DataFrame({bin_names[i]:Xbin[:,i],'BCRL':bcrl,'Cluster':cluster})
        dati_bin1 = dati_bin[dati_bin['Cluster'] == 'A']
        dati_bin2 = dati_bin[dati_bin['Cluster'] == 'B']
        mosaic(dati_bin1[[bin_names[i], 'BCRL']],index=[bin_names[i],'BCRL'],title="Cluster A",ax=ax1,axes_label=True)
        mosaic(dati_bin2[[bin_names[i], 'BCRL']],index=[bin_names[i],'BCRL'],title="Cluster B",ax=ax2,axes_label=True)
        plt.savefig(cartella_salvataggi+bin_names[i]+'_AB'+'.png', bbox_inches='tight',dpi=300)
        plt.close()
        for tipo_norm in [False,True]:
            cross1=pd.crosstab(dati_bin1[bin_names[i]], dati_bin1['BCRL'],normalize=tipo_norm)
            cross2=pd.crosstab(dati_bin2[bin_names[i]], dati_bin2['BCRL'],normalize=tipo_norm)
            if cross1.shape[0]>1 and cross1.shape[1]>1:
                tab1=conv_cross_to_numpy(cross1)
            else:
                tab1=ripara_tab(cross1)
            if cross2.shape[0]>1 and cross2.shape[1]>1:
                tab2=conv_cross_to_numpy(cross2)
            else:
                tab2=ripara_tab(cross2)
            annot_heatmap(bin_names[i],tab1,tab2,do_norm=tipo_norm)
out_tab=pd.crosstab(np.asarray(bcrl),cluster,rownames=['BCRL'],colnames=['Clusters'],margins=True)
out_tab2=pd.crosstab(np.asarray(bcrl),cluster,rownames=['BCRL'],colnames=['Clusters'],margins=True, normalize='all')
print(out_tab)
print('\n')
print(out_tab2)
def fare_linee(lista_val,titolo,aster1,aster2):
    colori=['blue','deepskyblue','red','orange']
    fig, ax = plt.subplots()
    for i,valore in enumerate(lista_val):
        ax.plot(np.median(valore),i+1,marker='o',color=colori[i])
        ax.plot((np.amin(valore), np.max(valore)), (i+1, i+1), color=colori[i])  
    x1,x2=plt.xlim()
    plt.xlim(x1-1, x2+1)
    if aster1==1:
        ax.plot(x1-0.5,1,marker='x',color='black')
        ax.plot(x1-0.5,3,marker='x',color='black')
    if aster2==1:
        ax.plot(x1-0.5,2,marker='*',color='black')
        ax.plot(x1-0.5,4,marker='*',color='black')
    ax.set_yticks([1,1.5,2,3,3.5,4])
    ax.set_yticklabels(['No BCRL',r'$\bf{Cluster A}$','BCRL','No BCRL',r'$\bf{Cluster B}$','BCRL'])
    ax.set_title(titolo)
    plt.ylim(0.5, 4.5)
    plt.tick_params(left = False, bottom = False,labelbottom = False)
    plt.savefig(cartella_salvataggi+titolo+'_confronto'+'.png', bbox_inches='tight',dpi=300)
    plt.close()
do_ord_stats=True
if do_ord_stats:
    print('\n\nStatistics')
    for i in range(Xord.shape[1]):
        asterisco1=0
        asterisco2=0
        dati_ord=pd.DataFrame({ord_names[i]:Xord[:,i],'BCRL':bcrl,'Cluster':cluster})
        dati_ord1 = dati_ord[dati_ord['Cluster'] == 'A']
        dati_ord2 = dati_ord[dati_ord['Cluster'] == 'B']
        dati_ord11 = dati_ord1[dati_ord1['BCRL'] == 'no'] #no BCRL in A
        dati_ord12 = dati_ord1[dati_ord1['BCRL'] == 'yes'] #BCRL in A
        dati_ord21 = dati_ord2[dati_ord2['BCRL'] == 'no'] # no BCRL in B
        dati_ord22 = dati_ord2[dati_ord2['BCRL'] == 'yes'] #BCRL in B
        no_BCRL_A=dati_ord11[ord_names[i]].to_numpy()
        BCRL_A=dati_ord12[ord_names[i]].to_numpy()
        no_BCRL_B=dati_ord21[ord_names[i]].to_numpy()
        BCRL_B=dati_ord22[ord_names[i]].to_numpy()
        print('Variable:',ord_names[i])
        U1, p1 = mannwhitneyu(no_BCRL_A, BCRL_A, method="exact")
        U2, p2 = mannwhitneyu(no_BCRL_B,BCRL_B , method="exact")
        if p1<0.05:
            print('Within Cluster A the difference is significant [NO BCRL vs BCRL]')
        if p2<0.05:
            print('Within Cluster B the difference is significant [NO BCRL vs BCRL]')
        U1, p1 = mannwhitneyu(no_BCRL_A, no_BCRL_B, method="exact")
        U2, p2 = mannwhitneyu(BCRL_A, BCRL_B, method="exact")
        if p1<0.05:
            print('Between Cluster A and B the difference is significant [NO BCRL patients]')
            asterisco1=1
        if p2<0.05:
            print('Between Cluster A and B the difference is significant [BCRL patients]')
            asterisco2=1
        print('\n')
        pippo=[no_BCRL_A, BCRL_A,no_BCRL_B,BCRL_B]        
        fare_linee(pippo,ord_names[i],asterisco1,asterisco2)
