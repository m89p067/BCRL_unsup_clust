import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

D={
'BREAST SURGERY':	0.155,
'SIDE':0.040,
'Ki67':0.161 ,
'TAXANE BASED CT':0.318 ,
'HT':0.17 ,
'TTZ':0.519,
'LVI':0.239,
'ECE':0.18,
'ER':	0.183 ,
'HER2':0.483,
'NCD':0.214,
'PR':0.163 
    }
sorted_footballers_by_goals = sorted(D.items(), key=lambda x:x[1], reverse=True)
converted_dict = dict(sorted_footballers_by_goals)
##plt.bar(range(len(converted_dict)), list(converted_dict.values()), align='center')
##plt.xticks(range(len(converted_dict)), list(converted_dict.keys()))
##plt.ylabel('Differential Risk')
##plt.show()

fig, ax = plt.subplots()

# Example data
people = list(converted_dict.keys())
y_pos = np.arange(len(people))
performance = list(converted_dict.values())
error = np.random.rand(len(people))

ax.barh(y_pos, performance,  align='center')
ax.set_yticks(y_pos, labels=people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Differential Risk')
#ax.set_title('How fast do you want to go today?')
plt.savefig('D:/SABCS 2022/CODE/'+'Bar_plot_bin_var.png', bbox_inches='tight',dpi=300)
plt.show()
