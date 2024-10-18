# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:27:26 2023

@author: Pablo Lombana
"""

import streamlit as st

st.set_page_config(page_title="Multipage App",
                   page_icon="👈", layout='wide')

st.title("Sitio de Autoayuda y seguimiento Kamala")
st.write("""

         ## Objetivo de la Aplicación

        En momentos de crisis, es fundamental contar con un espacio seguro donde puedas expresar tus sentimientos y recibir apoyo. Esta aplicación está diseñada para ofrecerte herramientas y recursos que te ayuden a sobrellevar momentos difíciles y a encontrar esperanza en medio del dolor.
        
        ### Funciones Principales
        
        1. **Chat de Apoyo**: Conéctate con profesionales de la salud mental que están disponibles para escucharte y ofrecerte apoyo emocional.
        2. **Recursos de Autoayuda**: Accede a una variedad de artículos, ejercicios de mindfulness y técnicas de relajación que pueden ayudarte a manejar tus emociones.
        
        ### Modo de Uso
        
        1. **Conversa**: Inicia una conversacion con Kamala y cuentale lo que tú quieras, como está tu día, que ha pasado los últimos meses... lo que quieras
        
        ### Recuerda
        
        No estás solo. Hay personas que se preocupan por ti y que están dispuestas a ayudarte. Esta aplicación es un primer paso hacia el apoyo que mereces y una vida más plena. Si sientes que la situación se vuelve abrumadora, por favor busca ayuda de inmediato.
        
        **Tu bienestar es nuestra prioridad.**
        
        Creado por
        **Pablo Andrés Lombana**

         """)


from pages.src.app.model.generate_embeddings import Embeddings
from pages.src.app.conf.parameters import MODEL_EMBEDDINGS
from pages.src.app.data.process_data import Preprocessing
import numpy as np


if 'instance_embeddings' not in st.session_state:
    name_session = np.random.randint(0, 9999999999)
    st.sidebar.success(f"Tu número de session corresponde a: {name_session}")
    st.session_state['name_session'] = name_session
    st.sidebar.success("Espera un momento mientras se cargan los componentes ⏱️")
    instance_embeddings = Embeddings(MODEL_EMBEDDINGS)
    instance_preprocessing = Preprocessing()
    st.session_state['instance_embeddings'] = instance_embeddings
    st.session_state['instance_preprocessing'] = instance_preprocessing
    st.sidebar.success("Listo 🥸 ahora selecciona la siguiente Pagina.... 🚀")

# def ReadingData():
#   df = pd.read_csv("Ofertas_finales.csv",sep="|")
#  df.rename(columns={'ID_Subtipo':'Tipo'},inplace=True)
# return df
# ofertas = ReadingData()
# st.session_state['ofertas'] = ofertas

# st.sidebar.success("Select a page above")