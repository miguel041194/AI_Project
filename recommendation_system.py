from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Lee el archivo CSV. Debe estar en la misma carpeta que este programa.
datos = pd.read_csv("classData.csv")

# Cada clave junta los valores de G, C, B y A en ese orden.
# El valor asociado es el nombre que queremos mostrar para esa combinación.
nombres_de_falla = {
    "0000": "Sin falla",
    "1001": "LG: fase A a tierra",
    "0011": "LL: fases A-B",
    "0110": "LL: fases B-C",
    "1011": "LLG: fases A-B a tierra",
    "0111": "LLL: trifásica",
    "1111": "LLLG: trifásica a tierra",
}

# Convierte los cuatro valores de cada fila a texto y los une.
# Por ejemplo, G=1, C=0, B=0 y A=1 se convierte en el código "1001".
codigo_de_falla = (
    datos[["G", "C", "B", "A"]]
    .astype(int)
    .astype(str)
    .agg("".join, axis=1)
)

# Busca el nombre para cada código y lo guarda en una nueva columna.
# Si un código no está en el diccionario, su nombre queda vacío (NaN).
datos["falla"] = codigo_de_falla.map(nombres_de_falla)

# Cuenta e imprime cuántos registros hay para cada nombre de falla.
print("Cantidad de registros por tipo de falla:")
print(datos["falla"].value_counts().sort_index().to_string())

# Avisa si alguna combinación no está incluida en el diccionario.
codigos_sin_nombre = codigo_de_falla[datos["falla"].isna()].value_counts()
if codigos_sin_nombre.empty:
    print("\nTodos los códigos tienen un nombre.")
else:
    print("\nAviso: hay códigos sin nombre:")
    print(codigos_sin_nombre.to_string())
    print("Esos registros se excluirán del entrenamiento y de la prueba.")

# Para que el programa pueda aprender, solo usamos filas que tienen una etiqueta.
datos_para_modelo = datos.dropna(subset=["falla"]).copy()

# Estas seis mediciones son las pistas que el modelo usará para reconocer la falla.
mediciones = datos_para_modelo[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]]

# La columna "falla" contiene la respuesta correcta que queremos que aprenda.
etiquetas = datos_para_modelo["falla"]

# Separamos los datos: 80 % para aprender y 20 % para probar el modelo.
# "stratify" conserva en ambos grupos una proporción parecida de cada tipo de falla.
# "random_state" hace que la separación se repita igual cada vez que ejecutamos el programa.
mediciones_entrenamiento, mediciones_prueba, fallas_entrenamiento, fallas_reales = (
    train_test_split(
        mediciones,
        etiquetas,
        test_size=0.2,
        random_state=42,
        stratify=etiquetas,
    )
)

# KNN usa distancias para encontrar los ejemplos más parecidos.
# Como las corrientes y tensiones tienen escalas distintas, StandardScaler
# las pone en escalas comparables antes de calcular esas distancias.
# Al unir ambos pasos en una tubería, la escala se aprende solo del entrenamiento.
modelo = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5),
)

# El modelo aprende los patrones de las mediciones de entrenamiento y sus etiquetas.
modelo.fit(mediciones_entrenamiento, fallas_entrenamiento)

# Le pedimos que prediga las fallas del grupo de prueba, que no usó para aprender.
fallas_predichas = modelo.predict(mediciones_prueba)

# La exactitud es la fracción de predicciones correctas; se muestra como porcentaje.
porcentaje_de_aciertos = accuracy_score(fallas_reales, fallas_predichas)
print(f"\nPorcentaje de aciertos en los datos de prueba: {porcentaje_de_aciertos:.2%}")

# La matriz compara las fallas reales (filas) con las predichas (columnas).
# Los valores de la diagonal son los aciertos; los demás indican confusiones.
nombres_de_clase = sorted(etiquetas.unique())
valores_matriz = confusion_matrix(
    fallas_reales,
    fallas_predichas,
    labels=nombres_de_clase,
)
matriz = pd.DataFrame(
    valores_matriz,
    index=nombres_de_clase,
    columns=nombres_de_clase,
)
matriz.index.name = "Falla real"
matriz.columns.name = "Falla predicha"

print("\nMatriz de confusión:")
print(matriz.to_string())

# Para comparar otra forma de clasificar, agrupamos LLL y LLLG bajo un nombre común.
# Esto no cambia las etiquetas originales ni los resultados del primer modelo.
grupos_de_falla = {
    "LLL: trifásica": "Trifásica",
    "LLLG: trifásica a tierra": "Trifásica",
}

# Cambiamos solo las etiquetas de entrenamiento y prueba para este experimento.
fallas_entrenamiento_agrupadas = fallas_entrenamiento.replace(grupos_de_falla)
fallas_reales_agrupadas = fallas_reales.replace(grupos_de_falla)

# Medimos qué pasaría si agrupáramos las predicciones del primer modelo,
# sin volver a entrenarlo. Así podemos separar el efecto de agrupar etiquetas
# del efecto de enseñar al modelo la nueva categoría.
fallas_predichas_agrupadas = pd.Series(
    fallas_predichas,
    index=fallas_reales.index,
).replace(grupos_de_falla)
aciertos_original_agrupado = accuracy_score(
    fallas_reales_agrupadas,
    fallas_predichas_agrupadas,
)

# Creamos y entrenamos un segundo KNN con las mismas mediciones y la misma
# separación de entrenamiento/prueba, pero con LLL y LLLG ya agrupadas.
modelo_agrupado = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5),
)
modelo_agrupado.fit(
    mediciones_entrenamiento,
    fallas_entrenamiento_agrupadas,
)
fallas_agrupadas_predichas = modelo_agrupado.predict(mediciones_prueba)
aciertos_modelo_agrupado = accuracy_score(
    fallas_reales_agrupadas,
    fallas_agrupadas_predichas,
)

# Mostramos los tres porcentajes para que la comparación sea más clara.
print("\nComparación al juntar LLL y LLLG:")
print(f"Modelo original, 6 categorías: {porcentaje_de_aciertos:.2%}")
print(
    "Modelo original contando LLL y LLLG como una sola: "
    f"{aciertos_original_agrupado:.2%}"
)
print(
    "Modelo KNN reentrenado con 5 categorías: "
    f"{aciertos_modelo_agrupado:.2%}"
)

# Esta matriz muestra el resultado del segundo KNN con las cinco categorías.
nombres_de_clase_agrupados = sorted(
    etiquetas.replace(grupos_de_falla).unique()
)
matriz_agrupada = pd.DataFrame(
    confusion_matrix(
        fallas_reales_agrupadas,
        fallas_agrupadas_predichas,
        labels=nombres_de_clase_agrupados,
    ),
    index=nombres_de_clase_agrupados,
    columns=nombres_de_clase_agrupados,
)
matriz_agrupada.index.name = "Falla real"
matriz_agrupada.columns.name = "Falla predicha"

print("\nMatriz de confusión del modelo con 5 categorías:")
print(matriz_agrupada.to_string())

# Calculamos la corriente residual firmada sumando las corrientes de las tres fases.
# Se conserva el signo para que el modelo reciba la suma Ia + Ib + Ic, no su valor absoluto.
mediciones_con_residual = mediciones.copy()
mediciones_con_residual["Ir"] = mediciones_con_residual[["Ia", "Ib", "Ic"]].sum(axis=1)

# Usamos las mismas filas de entrenamiento y prueba que en el modelo original,
# para que la comparación sea justa y se mantengan las seis categorías originales.
mediciones_residuales_entrenamiento = mediciones_con_residual.loc[
    mediciones_entrenamiento.index
]
mediciones_residuales_prueba = mediciones_con_residual.loc[mediciones_prueba.index]

# Creamos otro KNN con las seis mediciones originales más la corriente residual.
modelo_con_residual = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5),
)
modelo_con_residual.fit(
    mediciones_residuales_entrenamiento,
    fallas_entrenamiento,
)
fallas_con_residual_predichas = modelo_con_residual.predict(
    mediciones_residuales_prueba
)
aciertos_modelo_con_residual = accuracy_score(
    fallas_reales,
    fallas_con_residual_predichas,
)

# La tabla compara tres experimentos: el original, el de categorías agrupadas
# y el original de seis categorías con una séptima medición.
tabla_comparacion = pd.DataFrame(
    [
        {
            "Experimento": "KNN original: 6 mediciones, 6 categorías",
            "Aciertos": porcentaje_de_aciertos,
        },
        {
            "Experimento": "KNN agrupado: 6 mediciones, 5 categorías",
            "Aciertos": aciertos_modelo_agrupado,
        },
        {
            "Experimento": "KNN con residual: 7 mediciones, 6 categorías",
            "Aciertos": aciertos_modelo_con_residual,
        },
    ]
)
tabla_comparacion["Aciertos"] = tabla_comparacion["Aciertos"].map(
    lambda valor: f"{valor:.2%}"
)

print("\nComparación de los tres experimentos:")
print(tabla_comparacion.to_string(index=False))