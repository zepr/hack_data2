import numpy as np 
import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score


# Import database
sinistres = pd.read_excel(os.getenv("filer_dir") + "2021_hackathon_varenne_eau/OUTPUT/bdd_hackathon.xlsx")
climat = pd.read_csv(os.getenv("filer_dir") + "2021_hackathon_varenne_eau/base_finale_meteo_V2.csv", sep=";", low_memory=False)

# Feature engineering
sinistres_spec = sinistres[(sinistres["LIB_ALEA"]=="SECHERESSE")&(sinistres["DEPT_PARCELLE"].isin([79,86]))&(sinistres["GROUPE_CULTURES"].isin(["BLE TENDRE", "BLE DUR"]))].copy()
climat["date"] = climat["DATE"].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

# Merge
bdd = climat.merge(sinistres_spec[["COMMUNE", "LIB_MODE_PRODUCTION", "CULTURE_IRRIGUEE", "SURFACE", "TX_PERTE_RENDEMENT", "DATE_SURVENANCE", "PRA_Code", "PRA_Lib"]].copy(), left_on=["insee", "date"], right_on=["COMMUNE", "DATE_SURVENANCE"], how="left")

# Analyse
bdd = bdd.drop(["Unnamed: 0"], axis=1)
bdd["flag_sech"] = bdd["DATE_SURVENANCE"].apply(lambda x: 1 if x==x else 0)
bdd["MOIS"] = bdd["date"].apply(lambda x: x.month)
bdd["JOUR"] = bdd["date"].apply(lambda x: x.day)

# Export bdd filtrée sur les sinistres
### Beaucoup de données sinistres sans température ?
print("{} sinistres SECHERESSE sans données Température Max".format(len(bdd[(bdd["flag_sech"]==1)&(bdd["TX"]!=bdd["TX"])])))
bdd[(bdd["flag_sech"]==1)&(bdd["TX"]==bdd["TX"])].copy().to_csv(os.getenv("filer_dir") + "2021_hackathon_varenne_eau/sinistres_secheresse_meteo.csv", sep=";", index=False)

# Température du 5 aout 2018 ds commune 86118
bdd[(bdd["DATE"].isin([20180801,20180802,20180803,20180804,20180805,20180806,20180807,20180808]))&(bdd["insee"]==86118)]["TX"]
bdd[(bdd["DATE"].isin([20180801,20180802,20180803,20180804,20180805,20180806,20180807,20180808]))&(bdd["insee"]==86165)]["TX"]


# Température en fct du temps avec distingo aléa ou non
f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(15)
sns.lineplot(data=bdd, x="date", y="TX")
sns.scatterplot(data=bdd, x="DATE_SURVENANCE", y="TX")
plt.xticks(rotation=45)
plt.savefig('temp_fct_temps.png')







#############################################
### MODELE 1 : TOUTES VARIABLES CLIMATIQUES #
#############################################


# Préparation données pour modèle
data = bdd[["RR",	"TN",	"TX",	"TM",	"TAMPLI",	"FFM",	"FXI",	"FXY",	"UM",	"GLOT", "ETPMON", "JOUR", "MOIS", "TX_PERTE_RENDEMENT",	"flag_sech", "insee"]].copy()

# Ne garder que les lignes avec toutes les infos météo
data_sub = data.dropna(subset=["RR",	"TN",	"TX",	"TM",	"TAMPLI",	"FFM",	"FXI",	"FXY",	"UM",	"GLOT", "ETPMON"])
for col in ["RR",	"TN",	"TX",	"TM",	"TAMPLI",	"FFM",	"FXI",	"FXY",	"UM",	"GLOT", "ETPMON"]:
  data_sub["{}".format(col)]= data_sub["{}".format(col)].apply(lambda x: float(str(x).replace(",",".")))

data_sub.describe()

print("En ne gardant que les lignes où tous les indicateurs météo sont présents, il reste {} sinistres".format(len(data_sub[data_sub["flag_sech"]==1])))

# Ne garder que les lignes avec une perte de rdt inf à 20% ou sup à 50%
data_sub2 = data_sub[(data_sub["TX_PERTE_RENDEMENT"]<0.2)|(data_sub["TX_PERTE_RENDEMENT"]>0.5)|(data_sub["TX_PERTE_RENDEMENT"]!=data_sub["TX_PERTE_RENDEMENT"])].copy()
print("En ne gardant que les lignes où la perte de rendement est inf à 20% ou sup à 50%, il reste {} sinistres".format(len(data_sub2[data_sub2["flag_sech"]==1])))

data_sub2[(data_sub2["flag_sech"]==1)&(data_sub2["JOUR"]!=1)]

# Sampling
## Liste des codes communes présents dans le dataset sinistre
communes_sin = list(set(data_sub[(data_sub["flag_sech"]==1)&(data_sub["TX_PERTE_RENDEMENT"]>0.3)]["insee"]))
## On veut environ 5300 lignes sans sinistres, représentatives du dataset, avec que des communes présentes dans les sinistres

data_ss_sin_repr = data_sub[(data_sub["insee"].isin(communes_sin))&(data_sub["flag_sech"]==0)].copy().sample(n=5300, axis=0)

final_df = pd.concat([data_ss_sin_repr, data_sub2[(data_sub2["flag_sech"]==1)&(data_sub2["JOUR"]!=1)].copy()])
#final_df["IRRIGUE"] = final_df["CULTURE_IRRIGUEE"].apply(lambda x: 1 if x=="OUI" else 0)
#final_df.drop(["TX_PERTE_RENDEMENT", "insee", "CULTURE_IRRIGUEE"], axis=1, inplace=True)
final_df.drop(["TX_PERTE_RENDEMENT", "insee", "JOUR"], axis=1, inplace=True)
final_df.columns

# Train test split
X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]
X["MOIS"] = X["MOIS"].astype("str")
X = pd.get_dummies(data=X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modèle
regr = linear_model.LogisticRegression(max_iter=800)
regr.fit(X_train, y_train)

# NE CONVERGE PAS



#############################################
### MODELE 2 : TEMPERATURE ET PLUIE ET MOIS #
#############################################

# Préparation données pour modèle
data2 = bdd[["RR",	"TX", "JOUR", "MOIS", "TX_PERTE_RENDEMENT",	"flag_sech", "insee"]].copy()

# Ne garder que les lignes avec toutes les infos météo
data_sub = data2.dropna(subset=["RR",	"TX"])
for col in ["RR",	"TX"]:
  data_sub["{}".format(col)]= data_sub["{}".format(col)].apply(lambda x: float(str(x).replace(",",".")))

data_sub.describe()

print("En ne gardant que les lignes où tous les indicateurs météo sont présents, il reste {} sinistres".format(len(data_sub[data_sub["flag_sech"]==1])))

# Ne garder que les lignes avec une perte de rdt inf à 20% ou sup à 50%
data_sub2 = data_sub[(data_sub["TX_PERTE_RENDEMENT"]<0.2)|(data_sub["TX_PERTE_RENDEMENT"]>0.5)|(data_sub["TX_PERTE_RENDEMENT"]!=data_sub["TX_PERTE_RENDEMENT"])].copy()
print("En ne gardant que les lignes où la perte de rendement est inf à 20% ou sup à 50%, il reste {} sinistres".format(len(data_sub2[data_sub2["flag_sech"]==1])))

data_sub2[(data_sub2["flag_sech"]==1)&(data_sub2["JOUR"]!=1)]

# Sampling
## Liste des codes communes présents dans le dataset sinistre
communes_sin = list(set(data_sub[(data_sub["flag_sech"]==1)&(data_sub["TX_PERTE_RENDEMENT"]>0.3)]["insee"]))
## On veut environ 6000 lignes sans sinistres, représentatives du dataset, avec que des communes présentes dans les sinistres

data_ss_sin_repr = data_sub[(data_sub["insee"].isin(communes_sin))&(data_sub["flag_sech"]==0)].copy().sample(n=6000, axis=0)

final_df = pd.concat([data_ss_sin_repr, data_sub2[(data_sub2["flag_sech"]==1)&(data_sub2["JOUR"]!=1)].copy()])
#final_df["IRRIGUE"] = final_df["CULTURE_IRRIGUEE"].apply(lambda x: 1 if x=="OUI" else 0)
#final_df.drop(["TX_PERTE_RENDEMENT", "insee", "CULTURE_IRRIGUEE"], axis=1, inplace=True)
final_df.drop(["TX_PERTE_RENDEMENT", "insee", "JOUR"], axis=1, inplace=True)
final_df.columns

# Train test split
X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]

X["MOIS"]= X["MOIS"].astype("str")
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Modèle
regr = linear_model.LogisticRegression(max_iter=300)
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)
y_pred_proba = regr.predict_proba(X_test)
THRESHOLD = 0.3
preds = np.where(regr.predict_proba(X_test)[: , 1] > THRESHOLD, 1, 0)
preds.sum()
pred_proba = pd.DataFrame(preds)
confusion_matrix(y_test, pred_proba)


regr.score(X, y)

test = X_test.merge(y_test, left_index=True, right_index=True)
test[test["flag_sech"]==1]

test_proba = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True), pred_proba], axis=1)
test_proba[test_proba["flag_sech"]==1]["proba"].min()
test_proba[test_proba["flag_sech"]==1]["proba"].max()
test_proba[test_proba["flag_sech"]==1]["proba"].mean()

test_proba[test_proba["flag_sech"]==0]["proba"].min()
test_proba[test_proba["flag_sech"]==0]["proba"].max()
test_proba[test_proba["flag_sech"]==0]["proba"].mean()


pd.DataFrame(data = [accuracy_score(y_test, pred_proba), recall_score(y_test, pred_proba),
      precision_score(y_test, pred_proba), roc_auc_score(y_test, pred_proba)
   ],
   index = ["accuracy", "recall", "precision", "roc_auc_score"])



#############################################
### MODELE 3 : TEMPERATURE, PLUIE, ETP et MOIS      #
#############################################

# Préparation données pour modèle
data3 = bdd[["RR",	"TX", "ETPMON", "JOUR", "MOIS", "TX_PERTE_RENDEMENT",	"flag_sech", "insee"]].copy()

# Ne garder que les lignes avec toutes les infos météo
data_sub = data3.dropna(subset=["RR",	"TX", "ETPMON"])
for col in ["RR",	"TX", "ETPMON"]:
  data_sub["{}".format(col)]= data_sub["{}".format(col)].apply(lambda x: float(str(x).replace(",",".")))

data_sub.describe()

print("En ne gardant que les lignes où tous les indicateurs météo sont présents, il reste {} sinistres".format(len(data_sub[data_sub["flag_sech"]==1])))

# Ne garder que les lignes avec une perte de rdt inf à 20% ou sup à 50%
data_sub2 = data_sub[(data_sub["TX_PERTE_RENDEMENT"]<0.2)|(data_sub["TX_PERTE_RENDEMENT"]>0.5)|(data_sub["TX_PERTE_RENDEMENT"]!=data_sub["TX_PERTE_RENDEMENT"])].copy()
print("En ne gardant que les lignes où la perte de rendement est inf à 20% ou sup à 50%, il reste {} sinistres".format(len(data_sub2[data_sub2["flag_sech"]==1])))

data_sub2[(data_sub2["flag_sech"]==1)&(data_sub2["JOUR"]!=1)]

# Sampling
## Liste des codes communes présents dans le dataset sinistre
communes_sin = list(set(data_sub[(data_sub["flag_sech"]==1)&(data_sub["TX_PERTE_RENDEMENT"]>0.3)]["insee"]))
## On veut environ 2600 lignes sans sinistres, représentatives du dataset, avec que des communes présentes dans les sinistres

data_ss_sin_repr = data_sub[(data_sub["insee"].isin(communes_sin))&(data_sub["flag_sech"]==0)].copy().sample(n=2600, axis=0)

final_df = pd.concat([data_ss_sin_repr, data_sub2[(data_sub2["flag_sech"]==1)&(data_sub2["JOUR"]!=1)].copy()])
#final_df["IRRIGUE"] = final_df["CULTURE_IRRIGUEE"].apply(lambda x: 1 if x=="OUI" else 0)
#final_df.drop(["TX_PERTE_RENDEMENT", "insee", "CULTURE_IRRIGUEE"], axis=1, inplace=True)
final_df.drop(["TX_PERTE_RENDEMENT", "insee", "JOUR"], axis=1, inplace=True)
final_df.columns

# Train test split
X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]

X["MOIS"]= X["MOIS"].astype("str")
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Modèle
regr = linear_model.LogisticRegression(max_iter=300)
regr.fit(X_train, y_train)

y_pred_proba = regr.predict_proba(X_test)
THRESHOLD = 0.2
preds = np.where(regr.predict_proba(X_test)[: , 1] > THRESHOLD, 1, 0)
preds.sum()
pred_proba = pd.DataFrame(preds)
confusion_matrix(y_test, pred_proba)

pd.DataFrame(data = [accuracy_score(y_test, pred_proba), recall_score(y_test, pred_proba),
      precision_score(y_test, pred_proba), roc_auc_score(y_test, pred_proba)
   ],
   index = ["accuracy", "recall", "precision", "roc_auc_score"])

# get importance
importance = regr.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.savefig("feature_importance.png")


#############################################
### MODELE 4 : MOIS      #
#############################################

# Préparation données pour modèle
data4 = bdd[["JOUR", "MOIS", "TX_PERTE_RENDEMENT",	"flag_sech", "insee"]].copy()

# Ne garder que les lignes avec une perte de rdt inf à 20% ou sup à 50%
data_sub2 = data4[(data4["TX_PERTE_RENDEMENT"]<0.2)|(data4["TX_PERTE_RENDEMENT"]>0.5)|(data4["TX_PERTE_RENDEMENT"]!=data4["TX_PERTE_RENDEMENT"])].copy()
print("En ne gardant que les lignes où la perte de rendement est inf à 20% ou sup à 50%, il reste {} sinistres".format(len(data_sub2[data_sub2["flag_sech"]==1])))

data_sub2[(data_sub2["flag_sech"]==1)&(data_sub2["JOUR"]!=1)]

# Sampling
## Liste des codes communes présents dans le dataset sinistre
communes_sin = list(set(data4[(data4["flag_sech"]==1)&(data4["TX_PERTE_RENDEMENT"]>0.3)]["insee"]))
## On veut environ 3100 lignes sans sinistres, représentatives du dataset, avec que des communes présentes dans les sinistres

data_ss_sin_repr = data4[(data4["insee"].isin(communes_sin))&(data4["flag_sech"]==0)].copy().sample(n=3100, axis=0)

final_df = pd.concat([data_ss_sin_repr, data_sub2[(data_sub2["flag_sech"]==1)&(data_sub2["JOUR"]!=1)].copy()])
#final_df["IRRIGUE"] = final_df["CULTURE_IRRIGUEE"].apply(lambda x: 1 if x=="OUI" else 0)
#final_df.drop(["TX_PERTE_RENDEMENT", "insee", "CULTURE_IRRIGUEE"], axis=1, inplace=True)
final_df.drop(["TX_PERTE_RENDEMENT", "insee", "JOUR"], axis=1, inplace=True)
final_df.columns

# Train test split
X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]

X["MOIS"]= X["MOIS"].astype("str")
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Modèle
regr = linear_model.LogisticRegression(max_iter=300)
regr.fit(X_train, y_train)

y_pred_proba = regr.predict_proba(X_test)
THRESHOLD = 0.2
preds = np.where(regr.predict_proba(X_test)[: , 1] > THRESHOLD, 1, 0)
preds.sum()
pred_proba = pd.DataFrame(preds)
confusion_matrix(y_test, pred_proba)

pd.DataFrame(data = [accuracy_score(y_test, pred_proba), recall_score(y_test, pred_proba),
      precision_score(y_test, pred_proba), roc_auc_score(y_test, pred_proba)
   ],
   index = ["accuracy", "recall", "precision", "roc_auc_score"])

# get importance
importance = regr.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.close()
plt.bar([x for x in range(len(importance))], importance)
plt.savefig("feature_importance.png")


#############################################
### MODELE 5 : TEMPERATURE moyenne, PLUIE, ETP et MOIS      #
#############################################

# Préparation données pour modèle
data5 = bdd[["RR",	"TM", "ETPMON", "JOUR", "MOIS", "TX_PERTE_RENDEMENT",	"flag_sech", "insee"]].copy()

# Ne garder que les lignes avec toutes les infos météo
data_sub = data5.dropna(subset=["RR",	"TM", "ETPMON"])
for col in ["RR",	"TM", "ETPMON"]:
  data_sub["{}".format(col)]= data_sub["{}".format(col)].apply(lambda x: float(str(x).replace(",",".")))

data_sub.describe()

print("En ne gardant que les lignes où tous les indicateurs météo sont présents, il reste {} sinistres".format(len(data_sub[data_sub["flag_sech"]==1])))

# Ne garder que les lignes avec une perte de rdt inf à 20% ou sup à 50%
data_sub2 = data_sub[(data_sub["TX_PERTE_RENDEMENT"]<0.2)|(data_sub["TX_PERTE_RENDEMENT"]>0.5)|(data_sub["TX_PERTE_RENDEMENT"]!=data_sub["TX_PERTE_RENDEMENT"])].copy()
print("En ne gardant que les lignes où la perte de rendement est inf à 20% ou sup à 50%, il reste {} sinistres".format(len(data_sub2[data_sub2["flag_sech"]==1])))

data_sub2[(data_sub2["flag_sech"]==1)&(data_sub2["JOUR"]!=1)]

# Sampling
## Liste des codes communes présents dans le dataset sinistre
communes_sin = list(set(data_sub[(data_sub["flag_sech"]==1)&(data_sub["TX_PERTE_RENDEMENT"]>0.3)]["insee"]))
## On veut environ 2650 lignes sans sinistres, représentatives du dataset, avec que des communes présentes dans les sinistres

data_ss_sin_repr = data_sub[(data_sub["insee"].isin(communes_sin))&(data_sub["flag_sech"]==0)].copy().sample(n=2650, axis=0)

final_df = pd.concat([data_ss_sin_repr, data_sub2[(data_sub2["flag_sech"]==1)&(data_sub2["JOUR"]!=1)].copy()])
#final_df["IRRIGUE"] = final_df["CULTURE_IRRIGUEE"].apply(lambda x: 1 if x=="OUI" else 0)
#final_df.drop(["TX_PERTE_RENDEMENT", "insee", "CULTURE_IRRIGUEE"], axis=1, inplace=True)
final_df.drop(["TX_PERTE_RENDEMENT", "insee", "JOUR"], axis=1, inplace=True)
final_df.columns

# Train test split
X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]

X["MOIS"]= X["MOIS"].astype("str")
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Modèle
regr = linear_model.LogisticRegression(max_iter=300)
regr.fit(X_train, y_train)

y_test.sum()
y_pred_proba = regr.predict_proba(X_test)
THRESHOLD = 0.5
preds = np.where(regr.predict_proba(X_test)[: , 1] > THRESHOLD, 1, 0)
preds.sum()
pred_proba = pd.DataFrame(preds)
confusion_matrix(y_test, pred_proba)

pd.DataFrame(data = [accuracy_score(y_test, pred_proba), recall_score(y_test, pred_proba),
      precision_score(y_test, pred_proba), roc_auc_score(y_test, pred_proba)
   ],
   index = ["accuracy", "recall", "precision", "roc_auc_score"])

# get importance
importance = regr.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.savefig("feature_importance_model_5.png")






################################################
# Prévision survenance sécheresse dans le futur
################################################

future = pd.read_csv(os.getenv("filer_dir")+"2021_hackathon_varenne_eau/Future_2022-2080_Dept79_idstations.csv", sep=";")
future.head()
future.columns
future.drop('Unnamed: 0', axis=1, inplace=True)

future["ANNEE"] = future["Date"].apply(lambda x: pd.to_datetime(x).year)
future_2023 = future[future["ANNEE"]==2023].copy()
future_2023["MOIS"] = future_2023["Date"].apply(lambda x: pd.to_datetime(x).month)
future_2023 = future_2023.rename(columns={"ETPP":"ETPMON"})
for col in ['Latitude', 'Longitude', 'TM', 'RR', 'rldsAdjust', 'FFM', 'ETPMON', 'RG']:
  future_2023[col] = future_2023[col].apply(lambda x: float(str(x).replace(",", ".")))

X = future_2023[['RR', 'TM', 'ETPMON', 'MOIS']].copy()
X["MOIS"] = X["MOIS"].astype("str")
X = pd.get_dummies(data=X, drop_first=True)
#to_change = X.pop("RR")
#X.insert(0, column="RR", value=to_change)

# pred
pred_future = regr.predict_proba(X)
pred_future = pd.DataFrame(data=pred_future, columns=["pred_0", "pred"])
pred_future = pred_future["pred"]


# merge
future_2023.reset_index(inplace=True, drop=True)
df = future_2023.merge(pred_future, right_index=True, left_index=True)

# Viz
f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(15)
sns.scatterplot(data=df, x="Date", y="pred")
plt.savefig('proba_secheresse_fct_temps.png')

f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(15)
sns.lineplot(data=df, x="Date", y="pred")
plt.savefig('proba_secheresse_fct_temps_line.png')


list(set(df[df["pred"]>0.5]["station"]))
list(set(df[df["pred"]>0.55]["station"]))
df["pred"].sort_values().head(10)

df_10 = df[df["station"]==10].copy()

f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(15)
sns.scatterplot(data=df_10, x="Date", y="pred")
plt.savefig('proba_secheresse_fct_temps_station10.png')

f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(15)
sns.lineplot(data=df_10, x="Date", y="pred")
plt.savefig('proba_secheresse_fct_temps_station10.png')

df_10[df_10["pred"]>0.55]["Date"]
df_10[df_10["pred"]>0.45]["Date"]


# Station 94
df_94 = df[df["station"]==94].copy()

f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(15)
sns.lineplot(data=df_94, x="Date", y="pred")
plt.savefig('proba_secheresse_fct_temps_station94.png')

df_94[df_94["pred"]>0.45]["Date"]



#########################################################
# Prediction de 2022 à 2030 sur les stations 1 à 10 et 94
#########################################################

df_to_2040 = future[(future["ANNEE"]>=2022)&(future["ANNEE"]<=2040)&((future["station"]<=10)|(future["station"]==94))].copy()
df_to_2040["MOIS"] = df_to_2040["Date"].apply(lambda x: str(pd.to_datetime(x).month))
df_to_2040 = df_to_2040.rename(columns={"ETPP":"ETPMON"})
for col in ['TM', 'RR', 'rldsAdjust', 'FFM', 'ETPMON', 'RG']:
  df_to_2040[col] = df_to_2040[col].apply(lambda x: float(str(x).replace(",", ".")))

X = df_to_2040[['RR', 'TM', 'ETPMON', 'MOIS']].copy()
X = pd.get_dummies(data=X, drop_first=True)

# pred
pred_to_2040 = regr.predict_proba(X)
pred_to_2040 = pd.DataFrame(data=pred_to_2040, columns=["pred_0", "pred"])
pred_to_2040 = pred_to_2040["pred"]


# merge
df_to_2040.reset_index(inplace=True, drop=True)
df = df_to_2040.merge(pred_to_2040, right_index=True, left_index=True)
dico_station_commune = {1:"79106", 2:"79198",3:"79153", 4:"79211", 5:"79033", 6:"79350",
          7:"79348",8:"79057", 9:"79083", 10:"79175", 94:"79132"}
df["CODE_COMMUNE"] = df["station"].apply(lambda x: dico_station_commune[x])
df.drop(['Latitude', 'Longitude', 'TM', 'RR', 'rldsAdjust', 'FFM',
            'ETPMON', 'RG', 'MOIS', 'ANNEE', "station"], axis=1, inplace=True)
df.columns

df.to_csv(os.getenv("proj_dir")+"OUTPUT/hackathon_2021/prevision_secheresse_to_2040.csv", sep=";", index=False)

# Viz
f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(15)
sns.lineplot(data=df, x="Date", y="pred")
plt.savefig('proba_secheresse_fct_temps_to_2040.png')

df[df["pred"]>0.6]["Date"].sort_values()
