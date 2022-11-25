import pandas as pd
import numpy as np
import os
from math import isclose
import matplotlib.pyplot as plt
n_clust=3
cartella_princ='D:/BCRL_data/Using_'+str(n_clust)+'/'

do_dense=False
if do_dense:
    cartelle_out=[cartella_princ+'inters_dens/',cartella_princ+'union_dens/',cartella_princ+'contr_dens/']
else:
    cartelle_out=[cartella_princ+'inters/',cartella_princ+'union/',cartella_princ+'contr/']

if do_dense==True and n_clust==3:
    numero_ass=7.0 # freq assolute di bcrl
    diff_val=4.0 # differenza fra freq relative di bcrl, voglio clusters ben spaziati
    num_min_pz=45 # che ci siano alemno un tot numero di pz totali in un cluster
elif do_dense==False and n_clust==3:
    numero_ass=7.0 # freq assolute di bcrl
    diff_val=4.0 # differenza fra freq relative di bcrl, voglio clusters ben spaziati
    num_min_pz=45 # che ci siano alemno un tot numero di pz totali in un cluster

elif do_dense==True and n_clust==2:
    numero_ass=10 # freq assolute di bcrl
    diff_val=17.5 # differenza fra freq relative di bcrl, voglio clusters ben spaziati
    num_min_pz=1 # che ci siano alemno un tot numero di pz totali in un cluster
elif do_dense==False and n_clust==2:
    numero_ass=10 # freq assolute di bcrl
    diff_val=17.5 # differenza fra freq relative di bcrl, voglio clusters ben spaziati
    num_min_pz=1

elif do_dense==True and n_clust==4:
    diff_val=4.0
    num_min_pz=30 # che ci siano alemno un tot numero di pz totali in un cluster
elif do_dense==False and n_clust==4:    
    diff_val=4.0 # differenza fra freq relative di bcrl, voglio clusters ben spaziati
    num_min_pz=30

def calc_diff(ini_list):
    # Calculating difference list
    diff_list = []
    for x, y in zip(ini_list[0::], ini_list[1::]):
        diff_list.append(y-x)
    return diff_list

def remove_spaces(test_list):
    while("" in test_list) :
        test_list.remove("")
    return test_list
def make_the_plot(dato,eti,stringa_filename,cartella_filename,eti2):
    group=np.asarray(eti)
    cdict={'A':'red','B':'forestgreen','C':'royalblue','D':'lime','E':'orchid','F':'teal','G':'silver'}
    fig, ax = plt.subplots()
    
    for indice, g in enumerate(group):
        if eti2[indice]=='no bcrl':
            ax.scatter(dato[indice,0], dato[indice,1], c = cdict[g],  s = 30)
        else:
            ax.scatter(dato[indice,0], dato[indice,1], c = cdict[g],  s = 30,marker='x')
    #ax.legend(loc ="best")
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.savefig(cartella_filename+stringa_filename,dpi=300,bbox_inches='tight')
    plt.close()
    
cerca=True
if cerca==True:
    for cartella in cartelle_out:
        cartella_dati=cartella+'crosstabs/'
        for x in os.listdir(cartella_dati):
            if x.endswith("rel.txt"):
                freq_bcrl=[]
                freq_bcrl2=[]
                ass_bcrl=[]
                freq_sani=[]
                freq_sani2=[]
                stringa=x.split("_")
                tipo=stringa[0]
                num1=stringa[1]
                num2=stringa[2]
                #df = pd.read_table(cartella+'crosstabs/'+x ,sep=r'\s{2,}')
                #df_ass = pd.read_table(cartella+'crosstabs/'+tipo+'_'+num1+'_'+num2+'_ass.txt' ,sep=r'\s{2,}')
                df = pd.read_table(cartella+'crosstabs/'+x ,sep='\t')
                df_ass = pd.read_table(cartella+'crosstabs/'+tipo+'_'+num1+'_'+num2+'_ass.txt' ,sep='\t')
                for indice in range(n_clust):
                    #tmp=df.iloc[[2],[indice+1]] # frequency bcrl patients in that cluster
                    #freq_bcrl.append(tmp.to_numpy()[0][0]*100)
                    tmp=df.iloc[[2]].to_numpy()[0].tolist()[0].split(' ')
                    while("" in tmp) :
                        tmp.remove("")
                    freq_bcrl.append(float(tmp[indice+1])*100)
                    freq_bcrl2.append(float(tmp[indice+1])*100)
                    #tmp2=df_ass.iloc[[3],[indice+1]] # number of pz in that cluster
                    #ass_bcrl.append(int(tmp2.to_numpy()[0][0]))
                    tmp2=df_ass.iloc[[3]].to_numpy()[0].tolist()[0].split(' ')
                    while("" in tmp2) :
                        tmp2.remove("")
                    ass_bcrl.append(int(tmp2[indice+1]))
                    sani=df.iloc[[1]].to_numpy()[0].tolist()[0].split(' ')
                    while("" in sani) :
                        sani.remove("")
                    freq_sani.append(float(sani[indice+1])*100)
                    freq_sani2.append(float(sani[indice+1])*100)
                freq_bcrl.sort()
                ass_bcrl.sort()
                freq_sani.sort()
                diff_bcrl=calc_diff(freq_bcrl)
                diff_sani=calc_diff(freq_sani)
                if  (n_clust==3) and (freq_bcrl[0]<numero_ass) and (numero_ass < freq_bcrl[1] and freq_bcrl[1] < (numero_ass*2)) and (freq_bcrl[2]>(numero_ass*2)) :
                    if ass_bcrl[0]>num_min_pz:
                        print('COMBINATION:',num1,' vs ',num2,' type [',tipo,']')
                        print('SANI A:',round(freq_sani2[0],3),' SANI B:',round(freq_sani2[1],3),' SANI C:',round(freq_sani2[2],3))
                        print('BCRL A:',round(freq_bcrl2[0],3),' BCRL B:',round(freq_bcrl2[1],3),' BCRL C:',round(freq_bcrl2[2],3))
                        print('\n')
                elif (n_clust==3) and diff_bcrl[0] >diff_val and diff_bcrl[1] >diff_val :
                    if ass_bcrl[0]>num_min_pz:
                        print('COMBINATION:',num1,' vs ',num2,' type [',tipo,'] - By BCRL difference criteria')
                        print('SANI A:',round(freq_sani2[0],3),' SANI B:',round(freq_sani2[1],3),' SANI C:',round(freq_sani2[2],3))
                        print('BCRL A:',round(freq_bcrl2[0],3),' BCRL B:',round(freq_bcrl2[1],3),' BCRL C:',round(freq_bcrl2[2],3))
                        print('\n')                    

    ##            elif  (n_clust==2) and (freq_bcrl[0]<numero_ass) and (freq_sani[1]>60.0):
    ##                if ass_bcrl[0]>num_min_pz:
    ##                    print('COMBINATION:',num1,' vs ',num2,' type [',tipo,']')
    ##                    print('SANI A:',round(freq_sani2[0],3),' SANI B:',round(freq_sani2[1],3))
    ##                    print('BCRL A:',round(freq_bcrl2[0],3),' BCRL B:',round(freq_bcrl2[1],3))
    ##                    print('\n')                    
    ##            elif (n_clust==2) and (diff_bcrl[0] >diff_val) and  (diff_sani[0] >60.0):
    ##                if ass_bcrl[0]>num_min_pz:
    ##                    print('COMBINATION:',num1,' vs ',num2,' type [',tipo,'] - By difference criteria')
    ##                    print('SANI A:',round(freq_sani2[0],3),' SANI B:',round(freq_sani2[1],3))
    ##                    print('BCRL A:',round(freq_bcrl2[0],3),' BCRL B:',round(freq_bcrl2[1],3))
    ##                    print('\n')                    
                elif (n_clust==2) and (freq_sani2[0] >freq_sani2[1]) and  (freq_bcrl2[0] <freq_bcrl2[1]) and (diff_bcrl[0] >diff_val) and (diff_sani[0] >60.0):
                    if ass_bcrl[0]>num_min_pz:
                        print('COMBINATION:',num1,' vs ',num2,' type [',tipo,'] - By major')
                        print('SANI A:',round(freq_sani2[0],3),' SANI B:',round(freq_sani2[1],3))
                        print('BCRL A:',round(freq_bcrl2[0],3),' BCRL B:',round(freq_bcrl2[1],3))
                        print('\n')
                elif (n_clust==2) and (freq_sani2[0] <freq_sani2[1]) and  (freq_bcrl2[0] >freq_bcrl2[1]) and (diff_bcrl[0] >diff_val) and (diff_sani[0] >60.0):
                    if ass_bcrl[0]>num_min_pz:
                        print('COMBINATION:',num1,' vs ',num2,' type [',tipo,'] - By major')
                        print('SANI A:',round(freq_sani2[0],3),' SANI B:',round(freq_sani2[1],3))
                        print('BCRL A:',round(freq_bcrl2[0],3),' BCRL B:',round(freq_bcrl2[1],3))
                        print('\n')                    

                elif (n_clust==4) and not isclose( diff_bcrl[0],diff_bcrl[1],abs_tol= diff_val) and not isclose( diff_bcrl[1],diff_bcrl[2],abs_tol= diff_val) :                    
                    if ass_bcrl[0]>num_min_pz:
                        print('COMBINATION:',num1,' vs ',num2,' type [',tipo,'] - By difference BCRL criteria')                
                        print('SANI A:',round(freq_sani2[0],3),' SANI B:',round(freq_sani2[1],3),' SANI C:',round(freq_sani2[2],3),' SANI D:',round(freq_sani2[3],3))
                        print('BCRL A:',round(freq_bcrl2[0],3),' BCRL B:',round(freq_bcrl2[1],3),' BCRL C:',round(freq_bcrl2[2],3),' BCRL C:',round(freq_bcrl2[3],3))
                        print('\n')
                elif (n_clust==4) and not isclose( diff_sani[0],diff_sani[1],abs_tol= 10.0) and not isclose( diff_sani[1],diff_sani[2],abs_tol= 10.0) :
                    if ass_bcrl[0]>num_min_pz:
                        print('COMBINATION:',num1,' vs ',num2,' type [',tipo,'] - By difference HEALTHY criteria')                
                        print('SANI A:',round(freq_sani2[0],3),' SANI B:',round(freq_sani2[1],3),' SANI C:',round(freq_sani2[2],3),' SANI D:',round(freq_sani2[3],3))
                        print('BCRL A:',round(freq_bcrl2[0],3),' BCRL B:',round(freq_bcrl2[1],3),' BCRL C:',round(freq_bcrl2[2],3),' BCRL C:',round(freq_bcrl2[3],3))
                        print('\n')
else:
    num1=input('Immettere primo numero:')
    num2=input('Immettere secondo numero:')
    tipo_operaz=int(input('Inters [0] or Union [1] or Contrast [2] ?   '))
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
    scoring=my_dati['arr_2']
    etich2=my_dati['arr_3']
    classi=np.load('D:/BCRL_data/bcrl_classi.npy')
    dictionary = {0:'no bcrl', 1:'bcrl'}
    bcrl=list(map(dictionary.get, classi.tolist()))
    make_the_plot(mapper,etich1,stringa+stringa2+'.png',cartella_princ,bcrl)
    out_tab=pd.crosstab(np.asarray(bcrl),etich1,rownames=['BCRL'],colnames=['Clusters'],margins=True)
    out_tab2=pd.crosstab(np.asarray(bcrl),etich1,rownames=['BCRL'],colnames=['Clusters'],margins=True, normalize='all')
    print(out_tab)
    print(out_tab2)
