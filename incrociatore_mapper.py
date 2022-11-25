import pandas as pd
import numpy as np
from random import randint
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score,calinski_harabasz_score
from sklearn.metrics import pairwise_distances,davies_bouldin_score
from sklearn.mixture import GaussianMixture,BayesianGaussianMixture
import winsound
import json
import os
classi=np.load('D:/BCRL_data/bcrl_classi.npy')
unique, frequency = np.unique(classi,  return_counts = True)
dictionary = {0:'no', 1:'yes'}
diz_etich={1:'A',2:'B',3:'C',4:'D',5:'E',6:'F',7:'G'}
# print unique values array
print("Unique Values:", 
      unique)
  
# print frequency array
print("Frequency Values:",
      frequency)
bcrl=list(map(dictionary.get, classi.tolist()))
n_clust=3
cartella_princ='D:/BCRL_data/Using_'+str(n_clust)+'/'
do_dense=False
if do_dense:
    cartella1='D:/BCRL_data/mapper_data1/mapper1'
    cartella2='D:/BCRL_data/mapper_data2/mapper2'
    file_param1='stringhe_mapper1.csv'
    file_param2='stringhe_mapper2.csv'
    cartelle_out=[cartella_princ+'inters_dens/',cartella_princ+'union_dens/',cartella_princ+'contr_dens/']
else:
    cartella1='D:/BCRL_data/mapper_data3/mapper3'
    cartella2='D:/BCRL_data/mapper_data4/mapper4'
    file_param1='stringhe_mapper3.csv'
    file_param2='stringhe_mapper4.csv'
    cartelle_out=[cartella_princ+'inters/',cartella_princ+'union/',cartella_princ+'contr/']
for MYDIR in cartelle_out:
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
        os.makedirs(MYDIR+'crosstabs/')
    else:
        print(MYDIR, "folder already exists.")
print('... gathering data')
df_param1 = pd.read_csv(file_param1,delimiter=',',  header=None,  index_col=False)
df_param2 = pd.read_csv(file_param2,delimiter=',',  header=None,  index_col=False)
su1 = df_param1.rename(columns={df_param1.columns[0]: 'Rozzo'})
su2 = df_param2.rename(columns={df_param2.columns[0]: 'Rozzo'})
df_param1[['Inutile','File', 'n_Neigb','LR','Min_dist','Spreads','Metric']] = su1['Rozzo'].str.split('_', 0, expand=True)
df_param2[['Inutile','File', 'n_Neigb','LR','Min_dist','Spreads','Metric']] = su2['Rozzo'].str.split('_', 0, expand=True)
df_param1.drop(df_param1.columns[[0, 1]], axis = 1, inplace = True)
df_param2.drop(df_param2.columns[[0, 1]], axis = 1, inplace = True)

niter=5000
soglia_sil=0.8
quanti_file=400000
rnd1=np.random.randint(low=0, high=df_param1.shape[0], size=(quanti_file,))
rnd2=np.random.randint(low=0, high=df_param2.shape[0], size=(quanti_file,))
#rnd1=[4606,2739]
#rnd2=[1741,52]
del [su1,su2]
def fare_clustering(dati):
    gmm1 = GaussianMixture(n_components=n_clust,max_iter=niter,init_params='k-means++')
    try:
        gmm1.fit(dati)
        etichette1 = gmm1.predict(dati)
    except:
        sil1=0
        ch_index=0
        db_index=1        
        etichette1 =np.nan * np.ones(shape=(10,))
    try:
        sil1=silhouette_score(dati , etichette1, metric = 'euclidean')
        ch_index=calinski_harabasz_score(dati , etichette1)
        db_index=davies_bouldin_score(dati , etichette1)
    except:
        sil1=0
        ch_index=0
        db_index=1
    return etichette1,[sil1,ch_index,db_index]
def make_the_plot(dato,eti,stringa_filename,cartella_filename):
    group=np.asarray(eti)
    cdict={'A':'red','B':'forestgreen','C':'royalblue','D':'lime','E':'orchid','F':'teal','G':'silver'}
    fig, ax = plt.subplots()
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(dato[ix,0], dato[ix,1], c = cdict[g], label = g, s = 30)
    ax.legend(loc ="best")
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.savefig(cartella_filename+stringa_filename,dpi=300,bbox_inches='tight')
    plt.close()
for i_rnd1,i_rnd2 in zip(rnd1,rnd2):
    valori1=df_param1.iloc[[i_rnd1]]
    valori2=df_param2.iloc[[i_rnd2]]
    file1=valori1['File'].item()
    file2=valori2['File'].item()
    nb1=valori1['n_Neigb'].item()
    nb2=valori2['n_Neigb'].item()
    lr1=valori1['LR'].item()
    lr2=valori2['LR'].item()
    md1=valori1['Min_dist'].item()
    md2=valori2['Min_dist'].item()
    sp1=valori1['Spreads'].item()
    sp2=valori2['Spreads'].item()
    metr1=valori1['Metric'].item()
    metr2=valori2['Metric'].item()

    out_dict1={'File':file1,'n_Neigb':nb1,'LR':lr1,'Min_dist':md1,'Spreads':sp1,'Metric':metr1}
    out_dict2={'File':file2,'n_Neigb':nb2,'LR':lr2,'Min_dist':md2,'Spreads':sp2,'Metric':metr2}
    
    umap1=joblib.load(cartella1+'_'+file1)
    umap2=joblib.load(cartella2+'_'+file2)

    #intersection_mapper = umap1 * umap2
    #union_mapper = umap1 + umap2
    #contrast_mapper = umap1 - umap2
    
    intersection_mapper = np.multiply(umap1 , umap2)
    union_mapper = np.add(umap1 , umap2)
    contrast_mapper = np.subtract(umap1 , umap2)
    
    labels1R,score_clust1=fare_clustering(intersection_mapper)
    labels2R,score_clust2=fare_clustering(union_mapper)
    labels3R,score_clust3=fare_clustering(contrast_mapper)

    labels1R +=1
    labels2R +=1
    labels3R +=1

    unique1, frequency1 = np.unique(labels1R,  return_counts = True)
    unique2, frequency2 = np.unique(labels2R,  return_counts = True)
    unique3, frequency3 = np.unique(labels3R,  return_counts = True)


    labels1=np.asarray(list(map(diz_etich.get, labels1R.tolist())))
    labels2=np.asarray(list(map(diz_etich.get, labels2R.tolist())))
    labels3=np.asarray(list(map(diz_etich.get, labels3R.tolist())))
    
    if score_clust1[0]>soglia_sil and len(unique1)==n_clust:
        stringa='inters_'+file1+'_'+file2
        make_the_plot(intersection_mapper,labels1,stringa+'.png',cartelle_out[0])
        np.savez(cartelle_out[0]+stringa, intersection_mapper, labels1,np.asarray(score_clust1),labels1R)
        out_dict3={'Operation':'Intersection','Sil':str(score_clust1[0]),'CHI':str(score_clust1[1]),'DBI':str(score_clust1[2])}
        output = [out_dict3,out_dict1,out_dict2]        
        out_tab=pd.crosstab(np.asarray(bcrl),labels1,rownames=['BCRL'],colnames=['Clusters'],margins=True)
        out_tab2=pd.crosstab(np.asarray(bcrl),labels1,rownames=['BCRL'],colnames=['Clusters'],margins=True, normalize='all')
        tfile = open(cartelle_out[0]+'crosstabs/'+stringa+"_ass.txt", "a")
        tfile.write(out_tab.to_string())
        tfile.close()
        rfile = open(cartelle_out[0]+'crosstabs/'+stringa+"_rel.txt", "a")
        rfile.write(out_tab2.to_string())
        rfile.close()
        with open(cartelle_out[0]+stringa+".json", "w") as final:
            json.dump(output, final,indent=2)
        
    elif score_clust2[0]>soglia_sil and  len(unique2)==n_clust:
        stringa='union_'+file1+'_'+file2
        make_the_plot(union_mapper,labels2,stringa+'.png',cartelle_out[1])
        np.savez(cartelle_out[1]+stringa, union_mapper, labels2,np.asarray(score_clust2),labels2R)
        out_dict3={'Operation':'Union','Sil':str(score_clust2[0]),'CHI':str(score_clust2[1]),'DBI':str(score_clust2[2])}
        output = [out_dict3,out_dict1,out_dict2]
        out_tab=pd.crosstab(np.asarray(bcrl),labels2,rownames=['BCRL'],colnames=['Clusters'],margins=True)
        out_tab2=pd.crosstab(np.asarray(bcrl),labels2,rownames=['BCRL'],colnames=['Clusters'],margins=True, normalize='all')
        tfile = open(cartelle_out[1]+'crosstabs/'+stringa+"_ass.txt", "a")
        tfile.write(out_tab.to_string())
        tfile.close()
        rfile = open(cartelle_out[1]+'crosstabs/'+stringa+"_rel.txt", "a")
        rfile.write(out_tab2.to_string())
        rfile.close()
        with open(cartelle_out[1]+stringa+".json", "w") as final:
            json.dump(output, final,indent=2)
            
    elif score_clust3[0]>soglia_sil and len(unique3)==n_clust:
        stringa='contr_'+file1+'_'+file2
        make_the_plot(contrast_mapper,labels3,stringa+'.png',cartelle_out[2])
        np.savez(cartelle_out[2]+stringa, contrast_mapper, labels3,np.asarray(score_clust3),labels3R)
        out_dict3={'Operation':'Contrast','Sil':str(score_clust3[0]),'CHI':str(score_clust3[1]),'DBI':str(score_clust3[2])}
        output = [out_dict3,out_dict1,out_dict2]
        out_tab=pd.crosstab(np.asarray(bcrl),labels3,rownames=['BCRL'],colnames=['Clusters'],margins=True)
        out_tab2=pd.crosstab(np.asarray(bcrl),labels3,rownames=['BCRL'],colnames=['Clusters'],margins=True, normalize='all')
        tfile = open(cartelle_out[2]+'crosstabs/'+stringa+"_ass.txt", "a")
        tfile.write(out_tab.to_string())
        tfile.close()
        rfile = open(cartelle_out[2]+'crosstabs/'+stringa+"_rel.txt", "a")
        rfile.write(out_tab2.to_string())
        rfile.close()
        with open(cartelle_out[2]+stringa+".json", "w") as final:
            json.dump(output, final,indent=2)
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
