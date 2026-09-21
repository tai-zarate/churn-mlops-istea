import matplotlib.pylot as plt
import seaborn as sns

def churn_categoria(df,columna):
  resumen = (df.groupby(columna,observed=False)["Churn"]
             .apply(lambda x: (x=="Yes").mean())
             .sort_values(ascending=False)
             .reset_index(name="Pct_abandono")
            )
  ax = sns.barplot(data=resumen,
                   x=columna,
                   y="Pct_abandono")
  ax.set_ylabel("Porcentaje abandono (%)")
  ax.set_xlabel(columna)
  ax.set_title(f"Tasa de abandono por {columna}")
  plt.tight_layout()
  plt.show()

  return resumen

churn_categoria(df,"Contract")
churn_categoria(df,"PaymentMethod")
churn_categoria(df,"InternetService")
