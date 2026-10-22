import pandas as pd

df= pd.read_csv("I:\\escritos\\proeyectodiagnostico\\metadatos\\persona.csv")

df=df[["nivel_inst","frec_depre", "med_depre", "inte_depre"]]
antidepresivos= df[df["med_depre"]== "1"]
noantidepresivos= df[df["med_depre"]!= "1"]
corte= noantidepresivos[
    noantidepresivos["frec_depre"].isin(["1", "2", "3"]) &
    noantidepresivos["inte_depre"].isin(["2", "3"])
    ]

#Agrupar por sexo
antidepresivos = antidepresivos.groupby("nivel_inst").size()
corte= corte.groupby("nivel_inst").size()
conteo_total = antidepresivos.add(corte, fill_value=0)

#Agrupar por sexo el df original
divisor= df.groupby("nivel_inst").size()

#Divición
porcentaje= conteo_total/divisor
porcentaje= porcentaje.fillna(0)

porcentaje.to_csv("I:\\escritos\\proeyectodiagnostico\\metadatos\\porcentajeeductaivo.csv")