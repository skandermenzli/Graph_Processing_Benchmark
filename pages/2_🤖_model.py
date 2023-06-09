from pfa_model import PfaModel
import pandas as pd
import streamlit as st

st.title('Benchmark Model')

st.markdown("<span style='font-size:20px; font-weight:bold'>Enter your corresponding Graph and machine configurations and our model will select the appropiate framework for your case"
            "</span>", unsafe_allow_html=True)


machines = pd.read_csv("csv/machines.csv")
graphs = pd.read_csv("csv/graphs.csv")
model = PfaModel(0.5,0.5)
model.train(graphs,machines)

#columns
col1, col2,= st.columns([1,1])

    #Input fields in the first and second column

with col2:
    cpu = st.text_input("CPU cores:")
    algo = st.selectbox("Select Algorithm:", ["BFS", "PR10", "Triangle Counting", "Connected components"])
    nbr_edges = st.text_input("Number of edges:")

with col1:
        ram = st.text_input("Available ram:")
        size = st.text_input("Graph size(GB):")
        nbr_nodes = st.text_input("Number of nodes:")
        if st.button('Submit'):
            test_data = pd.DataFrame({
                # 'OS':[os_name],
                # 'disk': [disk_gb],
                'cpu': [float(cpu)],
                'ram': [float(ram)],
                'Graph_size(MB)': [float(size)],
                'Graph_nodes(vertices)': [int(nbr_nodes)],
                'Graph_edges': [int(nbr_edges)]
            })

            #st.write(test_data)
            name,time,config = model.predict(test_data,algo)

            st.write("The model compared frameworks performances on the following configuration configuration")
            st.dataframe(config)

            st.write("The best framework for your case is: <b>" + name + "</b>", unsafe_allow_html=True)

            st.write("with estimated time in seconds of :",time)




# Submit button in the middle column




