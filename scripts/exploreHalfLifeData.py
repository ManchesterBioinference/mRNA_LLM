# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from Bio import SeqIO

os.chdir('/mnt/mr01-home01/m65338lb/projects/rnaDecay')

# %%
# Function to merge two Excel files based on the 'transcript' column
# Load the two Excel files into pandas DataFrames
df1 = pd.read_excel("data/mmc3.xlsx")
df2 = pd.read_excel("data/journal.pbio.3001956.s003.xlsx")

# Merge the DataFrames based on the key column and keep only common rows (inner join)
merged_df = pd.merge(df1, df2, left_on="FBtr", right_on = "tr_id", how='inner')

merged_df[["Embryo half-life","Half_life (mins)"]]

# %%
# plot the values in the two columns to check for correlation
# Extract the relevant columns
columns_of_interest = merged_df[["Embryo half-life", "Half_life (mins)"]]

# Check for missing values and handle them (optional)
columns_of_interest = columns_of_interest.dropna()

# Plot the values in the two columns to check for correlation
plt.figure(figsize=(8, 6))
plt.scatter(columns_of_interest["Embryo half-life"], columns_of_interest["Half_life (mins)"], alpha=0.5)
plt.title("Scatter plot of Embryo half-life vs Half-life (mins)")
plt.xlabel("Embryo half-life")
plt.ylabel("Half-life (mins)")
plt.grid(True)
plt.show()

# %%
# Calculate and display the correlation coefficient between the two columns
# Log transform both the x and y values
columns_of_interest_log = columns_of_interest.applymap(lambda x: np.log(x))

# Calculate and display the correlation coefficient between the log-transformed columns
correlation_log = columns_of_interest_log.corr()
print("Correlation matrix (log-transformed):")
print(correlation_log)

# Use seaborn to visualize the correlation with a regression line on log-transformed data
sns.regplot(x="Embryo half-life", y="Half_life (mins)", data=columns_of_interest_log, scatter_kws={'alpha':0.5})
plt.title("Regression plot (log-transformed): Embryo half-life vs Half-life (mins)")
plt.xlabel("Log of Embryo half-life")
plt.ylabel("Log of Half-life (mins)")
plt.grid(True)
plt.show()
# %%
# Read in the fasta file
fasta_file = "output/data/utrDecayRates.fasta"
sequences = list(SeqIO.parse(fasta_file, "fasta"))

# Extract decay rates and sequence lengths
decay_rates = [float(seq.id) for seq in sequences]
seq_lengths = [len(seq.seq) for seq in sequences]

# Create a DataFrame for easier manipulation
decay_data = pd.DataFrame({
    "Decay Rate": decay_rates,
    "Sequence Length": seq_lengths
})

# Calculate the correlation coefficient between decay rate and sequence length
correlation_decay_length = decay_data.corr()
print("Correlation matrix (Decay Rate vs Sequence Length):")
print(correlation_decay_length)

# Plot the values in a correlation plot
plt.figure(figsize=(8, 6))
sns.regplot(x="Decay Rate", y="Sequence Length", data=decay_data, scatter_kws={'alpha':0.5})
plt.title("Correlation plot: Decay Rate vs Sequence Length")
plt.xlabel("Decay Rate")
plt.ylabel("Sequence Length")
plt.grid(True)
plt.show()

#%%
import pandas as pd
newDecay = pd.read_csv("data/parameters_estimates_extended_zygotic_tr_18022025_filtered.csv")
oldDecay = pd.read_csv("data/results_zygotic_tr_extended.csv")
# %%
newDecay.index = newDecay['tr_id']
oldDecay.index = oldDecay['tr_id']
newDecay['OldD'] = oldDecay.loc[[i for i in newDecay.index if i in oldDecay.index], 'D']
print(newDecay[["OldD", "D"]])
# %%
print(newDecay.loc[newDecay['D'] == 0.1,["OldD", "D"]].to_numpy())

# %%
# plot oldD vs D
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.regplot(x="OldD", y="D", data=newDecay, scatter_kws={'alpha':0.5})
plt.title("Correlation plot: OldD vs D")
plt.xlabel("OldD")
plt.ylabel("D")
plt.grid(True)
plt.show()
# %%
