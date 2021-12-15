
import numpy as np 
import pandas as pd 
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

bdd = pd.read_excel(os.getenv("filer_dir") + "2021_hackathon_varenne_eau/OUTPUT/bdd_hackathon.xlsx")
bdd_86_79 = bdd[bdd["DEPT_PARCELLE"].isin([79,86])].copy()
bdd_86_79_ble = bdd_86_79[bdd_86_79["GROUPE_CULTURES"].isin(["BLE TENDRE", "BLE DUR"])].copy()
bdd_86_79_ble_sech = bdd_86_79_ble[bdd_86_79_ble["LIB_ALEA"]=="SECHERESSE"].copy()



data = bdd_86_79_ble_sech.copy()
data["month"] = data["DATE_SURVENANCE"].apply(lambda x: x.month)
data = data[["LIB_MODE_PRODUCTION",	"CULTURE_IRRIGUEE",	"TX_PERTE_RENDEMENT",	"month"]].copy()
data["month"] = data["month"].astype("str")
data["TX_PERTE_RENDEMENT"] = round(data["TX_PERTE_RENDEMENT"], 1)
data = data[~data["TX_PERTE_RENDEMENT"].isin([0.8, 0.9])].copy()


X = data[["LIB_MODE_PRODUCTION",	"CULTURE_IRRIGUEE",	"month"]].copy()
X = X.astype("str")
X = pd.get_dummies(data=X, drop_first=True)
y = data["TX_PERTE_RENDEMENT"]
y.describe()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify = y)
classifier = DecisionTreeRegressor()
classifier.fit(X_train,y_train)


# Prediction
y_pred = classifier.predict(X_test)
classifier.score(X_test,y_test)


