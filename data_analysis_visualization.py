import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"/home/david/PycharmProjects/HiggsBosonProject/training .csv")
data = pd.DataFrame(data)
print(data.head())
print(data.describe())

# Fixing missing variables
data['PRI_jet_subleading_eta'].replace((-999.0), np.nan, inplace=True)
PRI_jet_subleading_eta_mean = data['PRI_jet_subleading_eta'].mean()
data['PRI_jet_subleading_eta'] = data['PRI_jet_subleading_eta'].fillna(PRI_jet_subleading_eta_mean)

data['PRI_jet_leading_eta'].replace((-999.0), np.nan, inplace=True)
PRI_jet_leading_eta_mean = data['PRI_jet_leading_eta'].mean()
data['PRI_jet_leading_eta'] = data['PRI_jet_leading_eta'].fillna(PRI_jet_leading_eta_mean)

data['DER_prodeta_jet_jet'].replace((-999.0), np.nan, inplace=True)
DER_prodeta_jet_jet_mean = data['DER_prodeta_jet_jet'].mean()
data['DER_prodeta_jet_jet'] = data['DER_prodeta_jet_jet'].fillna(DER_prodeta_jet_jet_mean)

data['PRI_jet_leading_pt'].replace((-999.0), np.nan, inplace=True)
PRI_jet_leading_pt_mean = data['PRI_jet_leading_pt'].mean()
data['PRI_jet_leading_pt'] = data['PRI_jet_leading_pt'].fillna(PRI_jet_leading_pt_mean)

data['PRI_jet_leading_phi'].replace((-999.0), np.nan, inplace=True)
PRI_jet_leading_phi_mean = data['PRI_jet_leading_phi'].mean()
data['PRI_jet_leading_phi'] = data['PRI_jet_leading_phi'].fillna(PRI_jet_leading_phi_mean)

data['PRI_jet_subleading_pt'].replace((-999.0), np.nan, inplace=True)
PRI_jet_subleading_pt_mean = data['PRI_jet_subleading_pt'].mean()
data['PRI_jet_subleading_pt'] = data['PRI_jet_subleading_pt'].fillna(PRI_jet_subleading_pt_mean)

data['PRI_jet_subleading_phi'].replace((-999.0), np.nan, inplace=True)
PRI_jet_subleading_phi_mean = data['PRI_jet_subleading_phi'].mean()
data['PRI_jet_subleading_phi'] = data['PRI_jet_subleading_phi'].fillna(PRI_jet_subleading_phi_mean)


# Droping label columns for corr matrix
# features = data.columns
# print(features)
# labels = data.columns[32]
# data = data.drop(labels, axis=1)
# print(data)

# correlation_matrix = data.corr()

# plt.figure(figsize=(24, 18))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Matrix")
# plt.tight_layout()

# plt.show()

count_missing_values = (data['DER_mass_MMC'] == -999.000).astype(int).sum()
print(count_missing_values)

data.loc[data['DER_mass_MMC'] == -999.0, 'DER_mass_MMC'] = np.sqrt(data.loc[data['DER_mass_MMC'] == -999.0, 'DER_mass_transverse_met_lep']**2 + data.loc[data['DER_mass_MMC'] == -999.0, 'DER_mass_vis']**2)
# data['DER_mass_MMC'] = np.sqrt(data['DER_mass_transverse_met_lep']**2 + data['DER_mass_vis']**2)
# data['DER_pt_ratio_lep_tau'] = data['PRI_lep_pt'] / data['PRI_tau_pt']
# data['DER_pt_h'] = np.sqrt(data['DER_pt_tot']**2 + data['DER_sum_pt']**2)
data.loc[data['DER_deltaeta_jet_jet'] == -999, 'DER_deltaeta_jet_jet'] = np.abs(data.loc[data['DER_deltaeta_jet_jet'] == -999, 'PRI_jet_leading_eta'] - data.loc[data['DER_deltaeta_jet_jet'] == -999, 'PRI_jet_subleading_eta'])
# data['DER_deltaeta_jet_jet'] = np.abs(data['PRI_jet_leading_eta'] - data['PRI_jet_subleading_eta'])
# data['DER_met_phi_centrality'] = np.abs(data['PRI_met_phi'] - data['PRI_lep_phi'])
data.loc[data['DER_lep_eta_centrality'] == -999, 'DER_lep_eta_centrality'] = np.abs(data.loc[data['DER_lep_eta_centrality'] == -999, 'PRI_lep_eta'] - data.loc[data['DER_lep_eta_centrality'] == -999, 'PRI_tau_eta'])
# data['DER_lep_eta_centrality'] = np.abs(data['PRI_lep_eta'] - data['PRI_tau_eta'])
# data['PRI_jet_all_pt'] = data['PRI_jet_leading_pt'] + data['PRI_jet_subleading_pt']
data['Weighted_Mass'] = data['DER_mass_MMC'] * data['Weight']


# Dividing into to signal and background datasets
data_signal = data[data['Label'] == 's']
data_background = data[data['Label'] == 'b']
print(data_signal)

# Plot histograms for each derived feature
# plt.figure(figsize=(15, 10))
# plt.subplot(3, 3, 1)
# sns.histplot(data['DER_mass_MMC'], kde=True)
# plt.title('DER_mass_MMC')
#
# plt.subplot(3, 3, 2)
# sns.histplot(data['DER_pt_ratio_lep_tau'], kde=True)
# plt.title('DER_pt_ratio_lep_tau')
#
# plt.subplot(3, 3, 3)
# sns.histplot(data['DER_pt_h'], kde=True)
# plt.title('DER_pt_h')
#
# plt.subplot(3, 3, 4)
# sns.histplot(data['DER_deltaeta_jet_jet'], kde=True)
# plt.title('DER_deltaeta_jet_jet')
#
# plt.subplot(3, 3, 5)
# sns.histplot(data['DER_met_phi_centrality'], kde=True)
# plt.title('DER_met_phi_centrality')
#
# plt.subplot(3, 3, 6)
# sns.histplot(data['DER_lep_eta_centrality'], kde=True)
# plt.title('DER_lep_eta_centrality')
#
# plt.subplot(3, 3, 7)
# sns.histplot(data['PRI_jet_all_pt'], kde=True)
# plt.title('PRI_jet_all_pt')
#
# plt.subplot(3, 3, 8)
# sns.histplot(data['Weighted_Mass'], kde=True)
# plt.title('Weighted_Mass')
#
# plt.tight_layout()
# plt.show()
#
#
plt.figure(figsize=(12, 9))
sns.histplot(data['Weighted_Mass'], kde=True)
plt.title('Weighted_Mass')
plt.show()

plt.figure(figsize=(12, 9))
sns.histplot(data['DER_mass_MMC'], kde=True)
plt.title('DER_mass_MMC')
plt.show()

print(data['DER_mass_MMC'])

plt.figure(figsize=(12, 9))
sns.histplot(data['DER_mass_MMC'], kde=True, color='blue')
plt.xscale('log')
plt.title('DER_mass_MMC')
plt.show()


plt.figure(figsize=(12, 9))
sns.histplot(data_signal['DER_mass_MMC'], kde=True, color='green')
# plt.xscale('log')
plt.title('DER_mass_MMC(signal)')
plt.show()

plt.figure(figsize=(12, 9))
sns.histplot(data_background['DER_mass_MMC'], kde=True, color='red')
# plt.xscale('log')
plt.title('DER_mass_MMC(background)')
plt.show()

# Set the figure size
plt.figure(figsize=(12, 9))

# Plot the histogram for 'DER_mass_MMC' from each DataFrame
sns.histplot(data['DER_mass_MMC'], kde=False, color='blue', label='DER_mass_MMC')
sns.histplot(data_background['DER_mass_MMC'], kde=False, color='green', label='DER_mass_MMC (background)')
sns.histplot(data_signal['DER_mass_MMC'], kde=False, color='red', label='DER_mass_MMC (signal)')

# Set logarithmic scale for x-axis
# plt.xscale('log')

# Add title and labels
plt.title('Histogram of DER_mass_MMC')
plt.xlabel('DER_mass_MMC')
plt.ylabel('Frequency')
plt.legend()  # Add legend

# Display the plot
plt.show()

plt.figure(figsize=(24, 18))
sns.histplot(data['DER_mass_vis'], kde=True, color='blue')
# plt.xscale('log')
plt.title('DER_mass_vis')
plt.show()


plt.figure(figsize=(24, 18))
sns.histplot(data_signal['DER_mass_vis'], kde=True, color='green')
# plt.xscale('log')
plt.title('DER_mass_vis(signal)')
plt.show()

plt.figure(figsize=(24, 18))
sns.histplot(data_background['DER_mass_vis'], kde=True, color='red')
# plt.xscale('log')
plt.title('DER_mass_vis(background)')
plt.show()

# Set the figure size
plt.figure(figsize=(24, 18))

# Plot the histogram for 'DER_mass_MMC' from each DataFrame
sns.histplot(data['DER_mass_vis'], kde=False, color='blue', label='DER_mass_vis')
sns.histplot(data_background['DER_mass_vis'], kde=False, color='green', label='DER_mass_vis (background)')
sns.histplot(data_signal['DER_mass_vis'], kde=False, color='red', label='DER_mass_vis (signal)')

# Set logarithmic scale for x-axis
# plt.xscale('log')

# Add title and labels
plt.title('Histogram of DER_mass_vis')
plt.xlabel('DER_mass_vis')
plt.ylabel('Frequency')
plt.legend()  # Add legend

# Display the plot
plt.show()


#fig = plt.figure(figsize=(16, 12))
#ax = fig.add_subplot(111, projection='3d')

#ax.scatter(data['DER_mass_MMC'], data['DER_mass_vis'], data['DER_mass_transverse_met_lep'], s = 40, marker = 'o', alpha = 1)
#ax.set_xlabel('DER_mass_MMC')
#ax.set_ylabel('DER_mass_vis')
#ax.set_zlabel('DER_mass_transverse_met_lep')
#ax.set_title('3D scatter plot')
#plt.show()

triples_selected = [
    ('DER_mass_MMC', 'DER_mass_vis', 'DER_mass_transverse_met_lep'),
    ('DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_pt_tot'),
    ('DER_mass_MMC', 'DER_mass_vis', 'DER_prodeta_jet_jet'),
    ('DER_mass_MMC', 'DER_mass_vis', 'DER_pt_tot'),
    ('DER_mass_MMC', 'DER_mass_vis', 'PRI_lep_pt'),
    ('DER_mass_MMC', 'DER_pt_h', 'DER_pt_tot'),
    ('DER_mass_MMC', 'DER_pt_h', 'PRI_jet_subleading_pt'),
    ('DER_mass_MMC', 'DER_deltaeta_jet_jet', 'PRI_jet_num'),
    ('DER_mass_MMC', 'DER_mass_jet_jet', 'PRI_jet_subleading_eta'),
    ('DER_mass_MMC', 'DER_prodeta_jet_jet', 'DER_pt_tot'),
    ('DER_mass_MMC', 'DER_lep_eta_centrality', 'PRI_met'),
    ('DER_mass_MMC', 'DER_lep_eta_centrality', 'PRI_jet_num'),
    ('DER_mass_MMC', 'PRI_met_phi', 'PRI_jet_subleading_pt'),
    ('DER_mass_MMC', 'PRI_jet_num', 'PRI_jet_leading_pt'),
    ('DER_mass_vis', 'DER_pt_h', 'PRI_jet_leading_pt'),
    ('DER_pt_h', 'PRI_jet_num', 'PRI_jet_leading_eta'),
    ('DER_deltaeta_jet_jet', 'DER_prodeta_jet_jet', 'PRI_jet_subleading_eta'),
    ('DER_deltaeta_jet_jet', 'DER_prodeta_jet_jet', 'PRI_jet_subleading_phi'),
    ('DER_deltaeta_jet_jet', 'DER_prodeta_jet_jet', 'PRI_jet_all_pt'),
    ('DER_deltaeta_jet_jet', 'DER_met_phi_centrality', 'PRI_jet_num'),
    ('DER_mass_jet_jet', 'DER_deltar_tau_lep', 'PRI_lep_pt'),
    ('DER_mass_jet_jet', 'PRI_tau_pt', 'PRI_jet_subleading_eta'),
    ('DER_prodeta_jet_jet', 'DER_sum_pt', 'PRI_jet_all_pt'),
    ('DER_mass_jet_jet', 'PRI_jet_leading_eta', 'PRI_jet_subleading_eta'),
    ('DER_deltar_tau_lep', 'PRI_lep_eta', 'PRI_jet_subleading_pt'),
    ('DER_pt_ratio_lep_tau', 'PRI_jet_num', 'PRI_jet_leading_pt'),
    ('DER_met_phi_centrality', 'DER_lep_eta_centrality', 'PRI_jet_num'),
    ('DER_met_phi_centrality', 'PRI_lep_eta', 'PRI_jet_num')
]

for z in triples_selected:
    fig = plt.figure(figsize = (15, 9))
    ax = fig.add_subplot(1, 2, 1, projection = '3d')
    x_b = data_background.replace(-999, np.nan)[z[0]]
    y_b = data_background.replace(-999, np.nan)[z[1]]
    z_b = data_background.replace(-999, np.nan)[z[2]]
    s1 = ax.scatter(x_b, y_b, z_b, s = 40, marker = 'o', c = y_b, alpha = 1)
    ax.set_title("Background events", fontsize = 14)
    ax.set_xlabel(z[0])
    ax.set_ylabel(z[1]) # ax.set_zlabel(z[2])
    ax = fig.add_subplot(1, 2, 2, projection = '3d')
    x_s = data_signal.replace(-999, np.nan)[z[0]]
    y_s = data_signal.replace(-999, np.nan)[z[1]]
    z_s = data_signal.replace(-999, np.nan)[z[2]]
    s2 = ax.scatter(x_s, y_s, z_s, s = 40, marker = 'o', c = y_s, alpha = 1)
    ax.set_title("Signal events", fontsize = 14)
    ax.set_xlabel(z[0])
    ax.set_ylabel(z[1])
    ax.set_zlabel(z[2])
    plt.tight_layout()

plt.show()
