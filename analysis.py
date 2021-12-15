

import numpy as np 
import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt


bdd = pd.read_excel(os.getenv("filer_dir") + "2021_hackathon_varenne_eau/OUTPUT/bdd_hackathon.xlsx")
bdd_86_79 = bdd[bdd["DEPT_PARCELLE"].isin([79,86])].copy()
bdd_86_79 = bdd_86_79[bdd_86_79[""]]
bdd_86_79["TX_PERTE_RENDEMENT"] = round(bdd_86_79["TX_PERTE_RENDEMENT"], 1)
bdd_86_79["month"] = bdd_86_79["DATE_SURVENANCE"].apply(lambda x: x.month)
bdd_86_79["jour"] = bdd_86_79["DATE_SURVENANCE"].apply(lambda x: x.day)
bdd_86_79_ble = bdd_86_79[bdd_86_79["GROUPE_CULTURES"].isin(["BLE TENDRE", "BLE DUR"])].copy()
bdd_86_79_ble_sech = bdd_86_79_ble[bdd_86_79_ble["LIB_ALEA"]=="SECHERESSE"].copy()



# Draw boxplot for one categorical column in Pandas dataframe
def draw_boxplot_categorical_col(df):
  dico = {categ:list(df[df["LIB_ALEA"]==categ]["TX_PERTE_RENDEMENT"]) for categ in list(set(df["LIB_ALEA"]))}
  return dico
        
draw_boxplot_categorical_col(bdd_86_79_ble)



df = pd.DataFrame(np.random.randn(10, 2),
                  columns=['Col1', 'Col2'])
df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A',
                     'B', 'B', 'B', 'B', 'B'])

                    
boxplot = bdd_86_79_ble.boxplot(column="TX_PERTE_RENDEMENT", by='LIB_ALEA')


 plt.ylim(0,10)

 plt.savefig('SimpleBoxPlot.png')
 plt.show()

# Taux de perte en fonction de l’aléa
f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(15)
ax = sns.boxplot(x="LIB_ALEA", y="TX_PERTE_RENDEMENT", data=bdd_86_79_ble)
plt.xticks(rotation=45)
plt.savefig('TX_PERTE_RENDEMENT_fct_LIB_ALEA.png')

# Taux de perte en fonction du mois
f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(15)
ax = sns.boxplot(x="month", y="TX_PERTE_RENDEMENT", data=bdd_86_79_ble_sech)
plt.xticks(rotation=45)
plt.savefig('TX_PERTE_RENDEMENT_fct_MONTH.png')

# Taux de perte le 1er jour du mois
data = bdd_86_79_ble_sech[bdd_86_79_ble_sech["jour"]==1].copy()
f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(15)
ax = sns.boxplot(x="month", y="TX_PERTE_RENDEMENT", data=data)
plt.xticks(rotation=45)
plt.savefig('TX_PERTE_RENDEMENT_fct_MONTH_jour1.png')
