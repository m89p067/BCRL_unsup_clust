import numpy as np
#import seaborn as sns
import pandas as pd
#from sklearn.metrics import silhouette_score
#from sklearn.metrics import pairwise_distances
#from sklearn.mixture import GaussianMixture,BayesianGaussianMixture
import joblib
import winsound
import string
import os
import csv
from random import randint,choice
import umap
dataMI=pd.read_csv('dataMILANO.csv', index_col=False)
dataNO=pd.read_csv('dataNOVARA.csv', index_col=False)
data=pd.concat([dataMI, dataNO], axis=0)
print('After merging:',data.shape[0],' patients')

data.drop('AXILLA SURGERY', axis=1, inplace=True)
data['AGE GROUP'],age_bins = pd.cut(data['AGE AT DIAGNOSIS'],bins= 10, labels=False,retbins=True)
data['BMI GROUP'] ,bmi_bins= pd.cut(data['BMI'], bins=10, labels=False,retbins=True)
np.savez('intervalli_mino', age_bins=age_bins, bmi_bins=bmi_bins)
data[['AGE GROUP', 'BMI GROUP']] = data[['AGE GROUP', 'BMI GROUP']].astype(int)
data.drop('AGE AT DIAGNOSIS', axis=1, inplace=True)
data.drop('BMI', axis=1, inplace=True)
for c in data.columns:
   print( "---- %s ---" % c)
   print( data[c].value_counts())
classe=data['BCRL'].to_numpy()
data.drop('BCRL', axis=1, inplace=True)
if os.path.isfile('D:/BCRL_data/bcrl_classi.npy'):
   print('Class data already saved')
else:
   np.save('bcrl_classi', classe)
X1=data[["NR METASTATIC LN","TOTAL NR DISSECTED LN","RT TYPE","HR DRUG 1","HISTOTYPE","G.1","T",
         "N EFF","MOLECULAR SUBTYPE","AGE GROUP","BMI GROUP"]]
X2=data[["BREAST SURGERY","SIDE","Ki_67","CT","TAXANE BASED CT","HT","TTZ (her2+)","LVI",
         "EXTRACAPSULAR EXTENSION (ECE)","ER","HER2","NCD","PR"]]
X2["BREAST SURGERY"] = X2["BREAST SURGERY"].subtract(1)
X2["SIDE"] = X2["SIDE"].subtract(1)
X2["Ki_67"] = X2["Ki_67"].subtract(1)
print('Ordinal dataframe is ',X1.shape[0],' x ',X1.shape[1])
print('Binary dataframe is ',X2.shape[0],' x ',X2.shape[1])
print('----------------All data loaded')
n_Neigbh=np.linspace(2,round(data.shape[0]/5), num=70,dtype=int)
lr=[0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,0.75]
minima_dist=[0.0,0.05, 0.1,0.15,0.2, 0.25,0.3,0.35,0.4,0.45, 0.5,0.6,0.65,0.7,0.75, 0.8,0.85,0.9,0.925,0.95, 0.99]
spreads=[0.5,1,1.5,2,2.5,3,4,5]
categ_metr=[
    'euclidean',
    'manhattan',
    'chebyshev',
    'minkowski',
    'canberra',
    'braycurtis',
    #'haversine',
    'mahalanobis',
    'wminkowski',
    'seuclidean',
    'cosine',
    'correlation'
    ]
bin_metr=[
    'hamming',
    'jaccard',
    'dice',
    'russellrao',
    'kulsinski',
    'rogerstanimoto',
    'sokalmichener',
    'sokalsneath',
    'yule'
]

execute_part=4
tot_iterat=12000
iniz_iter=11501
if execute_part==1:    
   cartella='D:/BCRL_data/mapper_data1/'
   if not os.path.exists(cartella):

     # if the demo_folder directory is not present 
     # then create it.
     os.makedirs(cartella)
   map_density=True
   stringa=[]
   contatore=1
   for i in range(iniz_iter,tot_iterat):
        
      nb =choice( n_Neigbh)
      lernr =choice( lr)
      md =choice( minima_dist)
      spr =choice( spreads)
      if md<spr:
         metr =choice(categ_metr)
         print('MAPPER1')
         try:
            mapper1 = umap.UMAP(n_neighbors= nb,n_epochs=10000,random_state=42,
                           min_dist=md,learning_rate=lernr,spread=spr,
                           metric=metr,densmap=map_density).fit_transform(X1)
            stringa.append('FILE_'+str(i)+'_'+str(nb)+'_'+str(lernr)+'_'+str(md)+'_'+str(spr)+'_'+metr)
            joblib.dump(mapper1,cartella+ 'mapper1_'+str(i))
            print('- Executed')
            del(mapper1)
         except:
            print('- Skipped')
      contatore=contatore+1                            
                            

      if i % 50==0:
         print('------>Done:',round((contatore/(tot_iterat-iniz_iter))*100,2),' %')
   if os.path.isfile('D:/BCRL_data/stringhe_mapper1.csv'):
      df2=pd.DataFrame(stringa)
      df2.to_csv('D:/BCRL_data/stringhe_mapper1.csv',index=False,header=False,mode="a")
      del(df2)
   else:
      df1 = pd.DataFrame(stringa)
      df1.to_csv(r'stringhe_mapper1.csv',header=False, index=False)
      del(df1)
elif execute_part==2:
   cartella='D:/BCRL_data/mapper_data2/'
   if not os.path.exists(cartella):

     # if the demo_folder directory is not present 
     # then create it.
     os.makedirs(cartella)
   map_density=True
   stringa=[]
   contatore=1
   for i in range(iniz_iter,tot_iterat):
        
      nb =choice( n_Neigbh)
      lernr =choice( lr)
      md =choice( minima_dist)
      spr =choice( spreads)
      if md<spr:
         metr =choice(categ_metr)
         print('MAPPER2')
         try:
            mapper2 = umap.UMAP(n_neighbors= nb,n_epochs=10000,random_state=42,
                           min_dist=md,learning_rate=lernr,spread=spr,
                           metric=metr,densmap=map_density).fit_transform(X2)
            stringa.append('FILE_'+str(i)+'_'+str(nb)+'_'+str(lernr)+'_'+str(md)+'_'+str(spr)+'_'+metr)
            joblib.dump(mapper2,cartella+ 'mapper2_'+str(i))
            print('- Executed')
            del(mapper2)
         except:
            print('- Skipped')
                            
      contatore=contatore+1                      

      if i % 50==0:
         print('------>Done:',round((contatore/(tot_iterat-iniz_iter))*100,2),' %')
   if os.path.isfile('D:/BCRL_data/stringhe_mapper2.csv'):
      df2=pd.DataFrame(stringa)
      df2.to_csv('D:/BCRL_data/stringhe_mapper2.csv',index=False,header=False,mode="a")
      del(df2)
   else:
      df1 = pd.DataFrame(stringa)
      df1.to_csv(r'stringhe_mapper2.csv',header=False, index=False)
      del(df1)
elif execute_part==3:    
   cartella='D:/BCRL_data/mapper_data3/'
   if not os.path.exists(cartella):

     # if the demo_folder directory is not present 
     # then create it.
     os.makedirs(cartella)
   map_density=False
   stringa=[]
   contatore=1
   for i in range(iniz_iter,tot_iterat):
        
      nb =choice( n_Neigbh)
      lernr =choice( lr)
      md =choice( minima_dist)
      spr =choice( spreads)
      if md<spr:
         metr =choice(categ_metr)
         print('MAPPER3')
         try:
            mapper3 = umap.UMAP(n_neighbors= nb,n_epochs=10000,random_state=42,
                           min_dist=md,learning_rate=lernr,spread=spr,
                           metric=metr,densmap=map_density).fit_transform(X1)
            stringa.append('FILE_'+str(i)+'_'+str(nb)+'_'+str(lernr)+'_'+str(md)+'_'+str(spr)+'_'+metr)
            joblib.dump(mapper3,cartella+ 'mapper3_'+str(i))
            print('- Executed')
            del(mapper3)
         except:
            print('- Skipped')
                            
      contatore=contatore+1                      

      if i % 50==0:
         print('------>Done:',round((contatore/(tot_iterat-iniz_iter))*100,2),' %')
   if os.path.isfile('D:/BCRL_data/stringhe_mapper3.csv'):
      df2=pd.DataFrame(stringa)
      df2.to_csv('D:/BCRL_data/stringhe_mapper3.csv',index=False,header=False,mode="a")
      del(df2)
   else:
      df1 = pd.DataFrame(stringa)
      df1.to_csv(r'stringhe_mapper3.csv',header=False, index=False)
      del(df1)
elif execute_part==4: 
   cartella='D:/BCRL_data/mapper_data4/'
   if not os.path.exists(cartella):

     # if the demo_folder directory is not present 
     # then create it.
     os.makedirs(cartella)
   map_density=False
   stringa=[]
   contatore=1
   for i in range(iniz_iter,tot_iterat):
        
      nb =choice( n_Neigbh)
      lernr =choice( lr)
      md =choice( minima_dist)
      spr =choice( spreads)
      if md<spr:
         metr =choice(categ_metr)
         print('MAPPER4')
         try:
            mapper4 = umap.UMAP(n_neighbors= nb,n_epochs=10000,random_state=42,
                           min_dist=md,learning_rate=lernr,spread=spr,
                           metric=metr,densmap=map_density).fit_transform(X2)
            stringa.append('FILE_'+str(i)+'_'+str(nb)+'_'+str(lernr)+'_'+str(md)+'_'+str(spr)+'_'+metr)
            joblib.dump(mapper4,cartella+ 'mapper4_'+str(i))
            print('- Executed')
            del(mapper4)
         except:
            print('- Skipped')
                            
      contatore=contatore+1                      

      if i % 50==0:
         print('------>Done:',round((contatore/(tot_iterat-iniz_iter))*100,2),' %')
   if os.path.isfile('D:/BCRL_data/stringhe_mapper4.csv'):
      df2=pd.DataFrame(stringa)
      df2.to_csv('D:/BCRL_data/stringhe_mapper4.csv',index=False,header=False,mode="a")
      del(df2)
   else:
      df1 = pd.DataFrame(stringa)
      df1.to_csv(r'stringhe_mapper4.csv',header=False, index=False)
      del(df1)
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
