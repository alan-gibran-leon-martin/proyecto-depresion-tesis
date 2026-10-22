import pandas as pd

df= pd.read_csv("I:\\escritos\\proeyectodiagnostico\\metadatos\\persona.csv")

df=df[["sexo","frec_depre", "med_depre", "inte_depre"]]
antidepresivos= df[df["med_depre"]== "1"]
noantidepresivos= df[df["med_depre"]!= "1"]
corte= noantidepresivos[
    noantidepresivos["frec_depre"].isin(["1", "2", "3"]) &
    noantidepresivos["inte_depre"].isin(["2", "3"])
    ]

#Agrupar por sexo
antidepresivos = antidepresivos.groupby("sexo").size()
corte= corte.groupby("sexo").size()
conteo_total = antidepresivos.add(corte, fill_value=0)

#Agrupar por sexo el df original
divisor= df.groupby("sexo").size()

#Divición
porcentaje= conteo_total/divisor
porcentaje= porcentaje.fillna(0)
print (porcentaje)
#porcentaje.to_csv("I:\\escritos\\proeyectodiagnostico\\metadatos\\porcentajesexo.csv")