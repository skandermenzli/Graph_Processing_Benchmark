import flask
import streamlit as st
from flask import Flask, request
import pandas as pd
from pfa_model import PfaModel


st.title('Benchmark For Graph processing tools on a single machine')

st.subheader('Greetings')

st.write("")

st.markdown("<span style='font-size:20px;'>This is a ML model that determines which single machine graph processing tools is better equiped to performe a graph algorithm on a particular graph."
            ":</span>", unsafe_allow_html=True)

st.markdown("<span style='font-size:20px;'>The list of tools to choose from are:</span>", unsafe_allow_html=True)

st.markdown("""
<ul style='font-size: 18px; font-weight:bold;'>
  <li>GraphChi</li>
  <li>MMAP</li>
  <li>Ligra</li>
  <li>X-stream</li>
</ul>
""", unsafe_allow_html=True)


st.markdown("<span style='font-size:20px;'>This was developped as part of our End of year project at INSAT, the team is composed of:</span>", unsafe_allow_html=True)

st.markdown("""
<ul style='font-size: 18px; font-weight:bold;'>
  <li> 🤷‍♂️Skander Menzli</li>
  <li> 🧮 Mohamed Farouk Drira</li>
  <li>👑 Racem Benrhayem</li>
  <li>💈 Mohamed Bouarda</li>
</ul>
""", unsafe_allow_html=True)

st.markdown("<span style='font-size:20px;'>Under the supervion of:</span>", unsafe_allow_html=True)

st.markdown("""
<ul style='font-size: 18px; font-weight:bold;'>
  <li> 👩🏼‍🏫Lilia Sfaxi</li>
  <li> 👩🏻‍🏫 Mariem Loukil</li>
  
</ul>
""", unsafe_allow_html=True)

app = Flask(__name__)
@app.route("/predict",methods=['POST'])
def hello_world():
    test_data = pd.DataFrame({
        # 'OS':[os_name],
        # 'disk': [disk_gb],
        'cpu': [float(request.form['cpu'])],
        'ram': [float(request.form['ram'])],
        'Graph_size(MB)': [float(request.form['graph_size'])],
        'Graph_nodes(vertices)': [int(request.form['graph_nodes'])],
        'Graph_edges': [int(request.form['graph_edges'])]
    })
    algo = request.form['algo']
    machines = pd.read_csv("csv/machines.csv")
    graphs = pd.read_csv("csv/graphs.csv")
   # print("flassssssssssssk:", flask.__version__)
    model = PfaModel(0.99,0.01)
    model.train(graphs, machines)
    #print("tessssssst",test_data)

    name, expected_time,config = model.predict(test_data, algo)
    return {'name':name, 'expected_time':expected_time}
    #return "<p>Hello, World!</p>"




