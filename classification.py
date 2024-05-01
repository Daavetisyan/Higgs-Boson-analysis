import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns

data = pd.read_csv(r"/home/david/PycharmProjects/HiggsBosonProject/data_boson.csv")
data_train = pd.read_csv(r"/home/david/PycharmProjects/HiggsBosonProject/test.csv")
data = data.drop(['Weight', 'Weighted_Mass'], axis=1)

data_train['PRI_jet_subleading_eta'].replace((-999.0), np.nan, inplace=True)
PRI_jet_subleading_eta_mean = data_train['PRI_jet_subleading_eta'].mean()
data_train['PRI_jet_subleading_eta'] = data_train['PRI_jet_subleading_eta'].fillna(PRI_jet_subleading_eta_mean)

data_train['PRI_jet_leading_eta'].replace((-999.0), np.nan, inplace=True)
PRI_jet_leading_eta_mean = data_train['PRI_jet_leading_eta'].mean()
data_train['PRI_jet_leading_eta'] = data_train['PRI_jet_leading_eta'].fillna(PRI_jet_leading_eta_mean)

data_train['PRI_jet_subleading_phi'].replace((-999.0), np.nan, inplace=True)
PRI_jet_subleading_phi_mean = data_train['PRI_jet_subleading_phi'].mean()
data_train['PRI_jet_subleading_phi'] = data_train['PRI_jet_subleading_phi'].fillna(PRI_jet_subleading_phi_mean)

data_train['PRI_jet_leading_phi'].replace((-999.0), np.nan, inplace=True)
PRI_jet_leading_phi_mean = data_train['PRI_jet_leading_phi'].mean()
data_train['PRI_jet_leading_phi'] = data_train['PRI_jet_leading_phi'].fillna(PRI_jet_leading_phi_mean)

data_train['PRI_jet_leading_pt'].replace((-999.0), np.nan, inplace=True)
PRI_jet_leading_pt_mean = data_train['PRI_jet_leading_pt'].mean()
data_train['PRI_jet_leading_pt'] = data_train['PRI_jet_leading_pt'].fillna(PRI_jet_leading_pt_mean)

data_train['PRI_jet_subleading_pt'].replace((-999.0), np.nan, inplace=True)
PRI_jet_subleading_pt_mean = data_train['PRI_jet_subleading_pt'].mean()
data_train['PRI_jet_subleading_pt'] = data_train['PRI_jet_subleading_pt'].fillna(PRI_jet_subleading_pt_mean)

data_train['DER_mass_jet_jet'].replace((-999.0), np.nan, inplace=True)
DER_mass_jet_jet_mean = data_train['DER_mass_jet_jet'].mean()
data_train['DER_mass_jet_jet'] = data_train['DER_mass_jet_jet'].fillna(DER_mass_jet_jet_mean)

data_train['DER_prodeta_jet_jet'].replace((-999.0), np.nan, inplace=True)
DER_prodeta_jet_jet_mean = data_train['DER_prodeta_jet_jet'].mean()
data_train['DER_prodeta_jet_jet'] = data_train['DER_prodeta_jet_jet'].fillna(DER_mass_jet_jet_mean)

data_train.loc[data_train['DER_mass_MMC'] == -999.0, 'DER_mass_MMC'] = np.sqrt(data_train.loc[data_train['DER_mass_MMC'] == -999.0, 'DER_mass_transverse_met_lep'] ** 2 + data_train.loc[data_train['DER_mass_MMC'] == -999.0, 'DER_mass_vis'] ** 2)
data_train.loc[data_train['DER_deltaeta_jet_jet'] == -999, 'DER_deltaeta_jet_jet'] = np.abs(data_train.loc[data_train['DER_deltaeta_jet_jet'] == -999, 'PRI_jet_leading_eta'] - data_train.loc[data_train['DER_deltaeta_jet_jet'] == -999, 'PRI_jet_subleading_eta'])
# data_train.loc[data['DER_deltaeta_jet_jet'] == -999, 'DER_deltaeta_jet_jet'] = np.abs(data_train.loc[data_train['DER_deltaeta_jet_jet'] == -999, 'PRI_jet_leading_eta'] - data_train.loc[data_train['DER_deltaeta_jet_jet'] == -999, 'PRI_jet_subleading_eta'])
data_train.loc[data_train['DER_lep_eta_centrality'] == -999, 'DER_lep_eta_centrality'] = np.abs(data_train.loc[data_train['DER_lep_eta_centrality'] == -999, 'PRI_lep_eta'] - data_train.loc[data_train['DER_lep_eta_centrality'] == -999, 'PRI_tau_eta'])

X = data.drop(['Label', 'EventId'], axis=1)
y = data['Label']
Eventid = data_train['EventId']
data_train = data_train.drop(['EventId'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = CatBoostClassifier(iterations=1700, learning_rate=0.05652288409977193, depth=5, l2_leaf_reg=8.53489341392174, border_count=192)

model.fit(x_train, y_train, verbose=0)
y_pred = model.predict(x_test)

y_pred1 = model.predict(data_train)
data_train['Label'] = y_pred1
print(data_train)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report
print(classification_report(y_test, y_pred))

data_train['EvenId'] = Eventid
data_train_signal = data_train[data_train['Label'] == 's']
data_train_background = data_train[data_train['Label'] == 'b']


plt.figure(figsize=(12, 9))
sns.histplot(data_train_signal['DER_mass_MMC'], kde=True, color='green')
# plt.xscale('log')
plt.title('DER_mass_MMC(signal)')
plt.show()

plt.figure(figsize=(12, 9))
sns.histplot(data_train_background['DER_mass_MMC'], kde=True, color='red')
# plt.xscale('log')
plt.title('DER_mass_MMC(background)')
plt.show()

plt.figure(figsize=(12, 9))
sns.histplot(data_train['DER_mass_MMC'], kde=True, color='blue')
# plt.xscale('log')
plt.title('DER_mass_MMC')
plt.show()

plt.figure(figsize=(12, 9))

# Plot the histogram for 'DER_mass_MMC' from each DataFrame
sns.histplot(data_train['DER_mass_MMC'], kde=False, color='blue', label='DER_mass_MMC')
sns.histplot(data_train_background['DER_mass_MMC'], kde=False, color='green', label='DER_mass_MMC (background)')
sns.histplot(data_train_signal['DER_mass_MMC'], kde=False, color='red', label='DER_mass_MMC (signal)')

# Set logarithmic scale for x-axis
# plt.xscale('log')

# Add title and labels
plt.title('Histogram of DER_mass_MMC')
plt.xlabel('DER_mass_MMC')
plt.ylabel('Frequency')
plt.legend()  # Add legend

# Display the plot
plt.show()
