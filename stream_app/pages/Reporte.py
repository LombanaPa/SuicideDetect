import streamlit as st
import os
import pandas as pd

# Función para buscar archivos en la carpeta
def buscar_y_leer_archivos(codigo, carpeta):
    data = []
    for archivo in os.listdir(carpeta):
        if codigo in archivo:
            ruta_archivo = os.path.join(carpeta, archivo)
            data.append(pd.read_csv(ruta_archivo))
    data = pd.concat(data)
    return data

# Interfaz de Streamlit
st.title("Buscador de Archivos")

# Ingreso de código
codigo = st.text_input("Ingresa el código a buscar:")

# Carpeta donde se buscarán los archivos
carpeta = "/Users/plombana/Documents/Universidad/ProyectodeGrado/SuicideDetect/stream_app/pages/src/results/"  # Cambia esto a la ruta de tu carpeta

# Botón para ejecutar la búsqueda
if st.button("Buscar Archivos"):
    if codigo:
        df_archivos = buscar_y_leer_archivos(codigo, carpeta)
        if df_archivos.shape[0]>0:
            # Crear DataFrame
            st.success("Archivos encontrados:")
            st.dataframe(df_archivos)  # Mostrar el DataFrame en Streamlit
        else:
            st.warning("No se encontraron archivos que coincidan con el código.")
    else:
        st.error("Por favor, ingresa un código válido.")

