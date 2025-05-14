import pandas as pd
import matplotlib.pyplot as plt

# Lire le fichier
df = pd.read_csv("performance_results.csv")

# Nettoyage (si jamais du bruit)
df = df.dropna(subset=["Type", "Time_ms"])

# Regroupement
grouped = df.groupby("Type")["Time_ms"].apply(list)

# Graphique
plt.figure(figsize=(10, 6))
plt.boxplot(grouped, labels=grouped.index)
plt.title("Comparaison des performances (CPU vs GPU)")
plt.ylabel("Temps (ms)")
plt.grid(True)

# Sauvegarde + affichage
plt.savefig("performance_plot.png")
plt.show()
