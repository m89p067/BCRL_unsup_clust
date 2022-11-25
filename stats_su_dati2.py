import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import joblib
import json
from scipy.stats import mannwhitneyu,chi2_contingency,fisher_exact,mode
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
cartella_salvataggi='D:/SABCS 2022/CODE/Stats_ALT_'+str(n_clust)+'/'

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
print(out_tab0)
print('\n')
def prepara_B_vs_All(dato):
    mydictionary={'A':'O','B':'B','C':'O'}
    target2=list(map(mydictionary.get, dato))
    return target2
print('*************************************************************************')
cluster=prepara_B_vs_All(etich1)
out_tabz=pd.crosstab(np.asarray(bcrl),np.asarray(cluster),rownames=['BCRL'],colnames=['Clusters'],margins=True)
print(out_tabz)
Xord=X1.to_numpy()
Xbin=X2.to_numpy()
bcrl_bin=classi
def annot_heatmap(la_stringa,dat1,dat2,do_norm=False):
    if do_norm==False:
        dat1=dat1.astype(int)
        dat2=dat2.astype(int)
    else:
        dat1=np.around(dat1*100,decimals=2)
        dat2=np.around(dat2*100,decimals=2)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    if do_norm==False:
        fig.suptitle(la_stringa+': cluster B vs O')
    else:
        fig.suptitle(la_stringa+': cluster B vs O [%]')
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
    ax1.set_title("Cluster B")
    ax2.set_title("Cluster O")
    ax1.set_ylabel(la_stringa)
    ax1.set_xlabel("BCRL")
    ax2.set_ylabel(la_stringa)
    ax2.set_xlabel("BCRL")
    fig.tight_layout()
    if do_norm==False:
        plt.savefig(cartella_salvataggi+la_stringa+'_heatmap_BO'+'.png', bbox_inches='tight',dpi=300)
    else:
        plt.savefig(cartella_salvataggi+la_stringa+'_norm_heatmap_BO'+'.png', bbox_inches='tight',dpi=300)
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
def odds_ratio(tabella):
    #https://www.ncbi.nlm.nih.gov/books/NBK431098/
    a=tabella[1,1]
    b=tabella[1,0]
    c=tabella[0,1]
    d=tabella[0,0]
    if a==0 or b==0 or c==0  or d==0:
        # Haldane-Anscombe correction
        # It involves adding 0.5 to all of the cells of a contingency table if any of the cell expectations would cause a division by zero error.
        a=a+0.5
        b=b+0.5
        c=c+0.5
        d=d+0.5
    Odds_ratio = (a*d) / (b*c)
    Upper_CI=exp(log(Odds_ratio)+ 1.96*sqrt( (1/a) + (1/b) + (1/c) + (1/d) ))
    Lower_CI=exp(log(Odds_ratio)- 1.96*sqrt( (1/a) + (1/b) + (1/c) + (1/d) ))    
    return Odds_ratio, Upper_CI, Lower_CI
def rel_risk(tabella):
    a=tabella[1,1]
    b=tabella[1,0]
    c=tabella[0,1]
    d=tabella[0,0]    
    if a==0 or b==0 or c==0  or d==0:
        # Where zeros cause problems with computation of the relative risk or its standard error, 0.5 is added to all cells (a, b, c, d) (Pagano & Gauvreau, 2000; Deeks & Higgins, 2010)
        a=a+0.5
        b=b+0.5
        c=c+0.5
        d=d+0.5
    relative_risk = (a / (a+b)) / (c / (c+d))
    Upper_CI=exp(log(relative_risk)+ 1.96*sqrt( (1/a) + (1/b) + (1/c) + (1/d) ))
    Lower_CI=exp(log(relative_risk)- 1.96*sqrt( (1/a) + (1/b) + (1/c) + (1/d) ))
    return relative_risk,Upper_CI,Lower_CI
def abs_risk_diff(tabella_uno, tabella_due):
    rischio1=tabella_uno[1,1]/(tabella_uno[1,1]+tabella_uno[1,0])
    rischio2=tabella_due[1,1]/(tabella_due[1,1]+tabella_due[1,0])
    return rischio1,rischio2

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
def overlap_percentage(xlist,ylist):
    min1 = min(xlist)
    max1 = max(xlist)
    min2 = min(ylist)
    max2 = max(ylist)

    overlap = max(0, min(max1, max2) - max(min1, min2))
    length = max1-min1 + max2-min2
    lengthx = max1-min1
    lengthy = max2-min2

    return 2*overlap/length , overlap/lengthx  , overlap/lengthy
def overlap(start1, end1, start2, end2):
    """how much does the range (start1, end1) overlap with (start2, end2)"""
    return max(max((end2-start1), 0) - max((end2-end1), 0) - max((start2-start1), 0), 0)

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
    ax.set_yticklabels(['No BCRL',r'$\bf{Cluster B}$','BCRL','No BCRL',r'$\bf{Cluster O}$','BCRL'])
    ax.set_title(titolo)
    plt.ylim(0.5, 4.5)
    plt.tick_params(left = False, bottom = False,labelbottom = False)
    #plt.savefig(cartella_salvataggi+titolo+'_confronto'+'.png', bbox_inches='tight',dpi=300)
    plt.close()
do_mosaic=False
if do_mosaic:
    #Mosaic plot on BINARY B vs O
    for i in range(Xbin.shape[1]):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(bin_names[i]+': cluster B vs O')
        dati_bin=pd.DataFrame({bin_names[i]:Xbin[:,i],'BCRL':bcrl,'Cluster':cluster})
        dati_bin1 = dati_bin[dati_bin['Cluster'] == 'B']
        dati_bin2 = dati_bin[dati_bin['Cluster'] == 'O']
        #mosaic(dati_bin1[[bin_names[i], 'BCRL']],index=[bin_names[i],'BCRL'],title="Cluster B",ax=ax1,axes_label=True)
        #mosaic(dati_bin2[[bin_names[i], 'BCRL']],index=[bin_names[i],'BCRL'],title="Cluster O",ax=ax2,axes_label=True)
        #plt.savefig(cartella_salvataggi+bin_names[i]+'_BO'+'.png', bbox_inches='tight',dpi=300)
        plt.close()
        for tipo_norm in [False,True]:
            if dati_bin1[bin_names[i]].dtype == "object":
                dati_bin1[bin_names[i]] = dati_bin1[bin_names[i]].astype(str).astype(int)    
            if dati_bin2[bin_names[i]].dtype == "object":
                dati_bin2[bin_names[i]] = dati_bin2[bin_names[i]].astype(str).astype(int)
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
            #annot_heatmap(bin_names[i],tab1,tab2,do_norm=tipo_norm)
            print('-----------------------------------')
            if tipo_norm == False:
                #rr1,u_rr1,l_rr1=rel_risk(tab1)
                #rr2,u_rr2,l_rr2=rel_risk(tab2)
                print('Variable:',bin_names[i])
                #print('Cluster B relative risk:',round(rr1,3),' [',round(l_rr1,3),'-',round(u_rr1,3),']')
                #print('Cluster O relative risk:',round(rr2,3),' [',round(l_rr2,3),'-',round(u_rr2,3),']')                
                #odds1,U_odds1,L_odds1=odds_ratio(tab1)
                #odds2,U_odds2,L_odds2=odds_ratio(tab2)
                #average_odds, x_odds,y_odds = overlap_percentage([L_odds1,U_odds1],[L_odds2,U_odds2])
                ardB,ardO=abs_risk_diff(tab1,tab2)
                print('Absolute Risk Cluster B :',round(ardB,3))
                print('Absolute Risk Cluster O :',round(ardO,3))
                print('Absolute Risk Difference cluster B vs cluster O :',round(ardB,3)-round(ardO,3) )
                #print("Variable:",bin_names[i]," ->average: ", average_odds*100,"%    x: ", x_odds*100,"%    y: ",y_odds*100,"%")
                #outover=overlap(L_odds1,U_odds1,L_odds2,U_odds2)
                #print('Cluster B odds ratio:',round(odds1,3),' [',round(L_odds1,3),'-',round(U_odds1,3),']')
                #print('Cluster O odds ratio:',round(odds2,3),' [',round(L_odds2,3),'-',round(U_odds2,3),']')
                #print("Variable:",bin_names[i]," -> overlap percentage between Odds ratio Conf. Interv. is ",outover)
                print('\n')
                if np.amin(tab1)<=5:
                    fs1,p1=fisher_exact(tab1)
                    if p1<0.1:
                        print('Cluster B ->  Presence of BCRL and ',bin_names[i],' varible are not equally distributed at with Fisher’s exact test [',fs1,' odds ratio] and p value ',p1)
                else:
                    chi1,p1,_,_=chi2_contingency(tab1, correction=True)
                    if p1<0.1:
                        print('Cluster B ->  Presence of BCRL and ',bin_names[i],' varible are not equally distributed at Chi-Square Test of Independence [',chi1,'] and p value ',p1)
                if np.amin(tab2)<=5:
                    fs2,p2=fisher_exact(tab2)
                    if p2<0.1:
                        print('Cluster O ->  Presence of BCRL and ',bin_names[i],' varible are not equally distributed at with Fisher’s exact test [',fs2,' odds ratio] and p value ',p2)
                else:
                    chi2,p2,_,_=chi2_contingency(tab2, correction=True)
                    if p2<0.1:
                        print('Cluster B ->  Presence of BCRL and ',bin_names[i],' varible are not equally distributed at Chi-Square Test of Independence [',chi2,'] and p value ',p2)
                g, p, dof, expctd = chi2_contingency(tab1, lambda_="log-likelihood")
                print("Cluster B : G={}; df={}; P={}".format(g, dof, p))
                g, p, dof, expctd = chi2_contingency(tab1, lambda_="log-likelihood")
                print("Cluster O : G={}; df={}; P={}".format(g, dof, p))
##                df_B=pd.DataFrame({'BCRL':dati_bin1['BCRL'],bin_names[i]:dati_bin1[bin_names[i]]})
##                df_O=pd.DataFrame({'BCRL':dati_bin2['BCRL'],bin_names[i]:dati_bin2[bin_names[i]]})
##                print('RESEARCHPY - - - - - Chi-square test of independence [',bin_names[i],']')
##                crosstab_B, res_B = rp.crosstab(df_B['BCRL'],df_B[bin_names[i]] , test= "chi-square")
##                crosstab_O, res_O = rp.crosstab(df_O['BCRL'],df_O[bin_names[i]], test= "chi-square")
##                print('Cluster B')
##                print(crosstab_B)
##                print(res_B)
##                print('Cluster O')
##                print(crosstab_O)
##                print(res_O)                
##                print('---------------------')
##                print('RESEARCHPY - - - - - g-test [',bin_names[i],']')
##                crosstab_B, res_B = rp.crosstab(df_B['BCRL'],df_B[bin_names[i]], test= "g-test")
##                crosstab_O, res_O = rp.crosstab(df_O['BCRL'],df_O[bin_names[i]], test= "g-test")
##                print('Cluster B')
##                print(crosstab_B)
##                print(res_B)
##                print('Cluster O')
##                print(crosstab_O)
##                print(res_O)                
##                print('---------------------')
##                print('RESEARCHPY - - - - - Fisher’s Exact test [',bin_names[i],']')
##                crosstab_B, res_B = rp.crosstab(df_B['BCRL'],df_B[bin_names[i]], test= "fisher")
##                crosstab_O, res_O = rp.crosstab(df_O['BCRL'],df_O[bin_names[i]], test= "fisher")
##                print('Cluster B')
##                print(crosstab_B)
##                print(res_B)
##                print('Cluster O')
##                print(crosstab_O)
##                print(res_O)                
##                print('---------------------')
##                #For conducting a McNemar test, make sure the outcomes in both variables are labelled the same.
##                print('RESEARCHPY - - - - - McNemar test [',bin_names[i],']')
##                crosstab_B, res_B = rp.crosstab(df_B['BCRL'],df_B[bin_names[i]], test= "mcnemar")
##                crosstab_O, res_O = rp.crosstab(df_O['BCRL'],df_O[bin_names[i]], test= "mcnemar")
##                print('Cluster B')
##                print(crosstab_B)
##                print(res_B)
##                print('Cluster O')
##                print(crosstab_O)
##                print(res_O)                
##                print('---------------------')
##out_tab=pd.crosstab(np.asarray(bcrl),np.asarray(cluster),rownames=['BCRL'],colnames=['Clusters'],margins=True)
##out_tab2=pd.crosstab(np.asarray(bcrl),np.asarray(cluster),rownames=['BCRL'],colnames=['Clusters'],margins=True, normalize='all')
##print(out_tab)
##print('\n')
##print(out_tab2)

do_ord_stats=False
if do_ord_stats:
    print('\n\nStatistics')
    for i in range(Xord.shape[1]):
        asterisco1=0
        asterisco2=0
        dati_ord=pd.DataFrame({ord_names[i]:Xord[:,i],'BCRL':bcrl,'Cluster':cluster})
        dati_ord1 = dati_ord[dati_ord['Cluster'] == 'B']
        dati_ord2 = dati_ord[dati_ord['Cluster'] == 'O']
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
            print('Within Cluster B the difference is significant [NO BCRL vs BCRL]')
        if p2<0.05:
            print('Within Cluster O the difference is significant [NO BCRL vs BCRL]')
        U1, p1 = mannwhitneyu(no_BCRL_A, no_BCRL_B, method="exact")
        U2, p2 = mannwhitneyu(BCRL_A, BCRL_B, method="exact")
        if p1<0.05:
            print('Between Cluster B and O the difference is significant [NO BCRL patients]')
            asterisco1=1
        if p2<0.05:
            print('Between Cluster B and O the difference is significant [BCRL patients]')
            asterisco2=1
        print('\n')
        pippo=[no_BCRL_A, BCRL_A,no_BCRL_B,BCRL_B]        
        fare_linee(pippo,ord_names[i],asterisco1,asterisco2)
nominal_var=False
if nominal_var:
    variabili=["RT TYPE","HR DRUG 1","HISTOTYPE","MOLECULAR SUBTYPE"]
    for var_nom in variabili:
        dati_ord = pd.DataFrame({var_nom:X1[var_nom].to_numpy(),'BCRL':bcrl,'Cluster':cluster})
        dati_ord1 = dati_ord[dati_ord['Cluster'] == 'B']
        dati_ord2 = dati_ord[dati_ord['Cluster'] == 'O']
        dati_ord11 = dati_ord1[dati_ord1['BCRL'] == 'no'] #no BCRL in A
        dati_ord12 = dati_ord1[dati_ord1['BCRL'] == 'yes'] #BCRL in A
        dati_ord21 = dati_ord2[dati_ord2['BCRL'] == 'no'] # no BCRL in B
        dati_ord22 = dati_ord2[dati_ord2['BCRL'] == 'yes'] #BCRL in B
        no_BCRL_A=dati_ord11[var_nom].to_numpy()
        BCRL_A=dati_ord12[var_nom].to_numpy()
        no_BCRL_B=dati_ord21[var_nom].to_numpy()
        BCRL_B=dati_ord22[var_nom].to_numpy()
        print('Variable:',var_nom)
        Mode1=mode(no_BCRL_A)[0]
        Mode2=mode(BCRL_A)[0]
        Mode3=mode(no_BCRL_B)[0]
        Mode4=mode(BCRL_B)[0]
        print('Within Cluster B, NO BCRL group mode:',Mode1,' vs. BCRL group mode:',Mode2)
        print('Within Cluster O, NO BCRL group mode:',Mode3,' vs. BCRL group mode:',Mode4)
        print('NO BCRL patients, Mode in cluster B:',Mode1,' vs. Mode in cluster O:',Mode3)
        print('BCRL patients, Mode in cluster B:',Mode2,' vs. Mode in cluster O:',Mode4)
        print('\n')
bin_var_cont_tables=True
if bin_var_cont_tables:
    for bin_var in bin_names:
        print('\n----------------------------------------------------')
        out=pd.crosstab(X2[bin_var],bcrl,rownames=[bin_var],colnames=['BCRL'],margins=False)
        print(out)
