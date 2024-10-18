import pandas as pd
from openai import OpenAI
import streamlit as st

#from SuicideDetect.src.app.model.predict import predict
from pages.src.app.model.predict import predict
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_behavior(text,name_session):
    instance_preprocessing = st.session_state['instance_preprocessing']
    instance_embeddings = st.session_state['instance_embeddings']
    text_prepro = instance_preprocessing.__call__(text)
    embeddings_step = np.array(instance_embeddings.get_embeddings(text_prepro))
    prediction = np.array(predict(embeddings_step.reshape(1, 768))[0])
    name_pred = np.random.randint(0,9999999999)
    pd.DataFrame({'session':name_session,
                  'prediction': prediction,
                  'number_pred': name_pred,
                  'texto':text}).to_csv(f"/Users/plombana/Documents/Universidad/ProyectodeGrado/SuicideDetect/stream_app/pages/src/results/{name_session}_{name_pred}.csv", index = False, decimal=",")
    return prediction

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chat de Autoayuda Kamala")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hola soy Kamala, me alegra que busques esta linea para hablar... cuentame en que puedo ayudarte?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    predict_behavior(st.session_state.messages[-1]['content'], st.session_state['name_session'])
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
