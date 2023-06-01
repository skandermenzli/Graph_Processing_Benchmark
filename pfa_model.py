import os
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd


class PfaModel:

    def __init__(self, ram_weight, time_weight):
        # Initialize the model with given hyperparameters
        self.ram_weight = ram_weight
        self.time_weight = time_weight
        self.scalers = {}
        self.knn = NearestNeighbors(n_neighbors=1)
        self.res = pd.DataFrame()

    def preprocess(self, df_graph, df_machine):

        numeric_cols1 = df_graph.select_dtypes(include='number').columns
        numeric_cols2 = df_machine.select_dtypes(include='number').columns
        for col in numeric_cols1:
            scaler = StandardScaler()
            df_graph[col] = scaler.fit_transform(df_graph[col].values.reshape(-1, 1))
            print(scaler.mean_)
            self.scalers[col] = scaler
        train_graphs = df_graph.loc[:, ['Graph_name', 'Graph_size(MB)', 'Graph_nodes(vertices)', 'Graph_edges']]

        for col in numeric_cols2:
            scaler = StandardScaler()
            df_machine[col.replace('machine_', '')] = scaler.fit_transform(df_machine[col].values.reshape(-1, 1))
            self.scalers[col.replace('machine_', '')] = scaler
        train_machines = df_machine.loc[:, ['cpu', 'ram', 'machine_cpu', 'machine_ram']]

        train_graphs['key'] = 0
        train_machines['key'] = 0
        train_data = train_machines.merge(train_graphs, on='key', how='outer')
        self.res = train_data[['Graph_name', 'machine_cpu', 'machine_ram']]
        #print(train_data)
        train_data = train_data.drop(['key', 'Graph_name', 'machine_cpu', 'machine_ram'], axis=1)

        # print(names)

        return train_data

    def train(self, graphs, machines):
        train_data = self.preprocess(graphs, machines)
        self.knn.fit(train_data)
        print("model trained succesfully!")
        return train_data

    def predict(self, x,algo):
        for col in x.columns:
            scaler = self.scalers[col]
            x[col] = scaler.transform(x[col].values.reshape(-1, 1))
        print(x)
        distances, indices = self.knn.kneighbors(x)
        #print("predict")
        print("distance: ", distances)
        print("indice", indices)
        print(self.res.iloc[[indices[0, 0]]])
        config = self.res.iloc[[indices[0, 0]]]
        
        my_path = os.path.abspath(os.path.dirname(__file__))

        graphchi = pd.read_csv(os.path.join(my_path,'csv/PFA-GraphChi.csv'))
        mmap = pd.read_csv(os.path.join(my_path,'csv/PFA-MMAP.csv'))
        
        ligra = pd.read_csv(os.path.join(my_path, "./csv/PFA-Ligra.csv"))
        tools = [mmap, graphchi, ligra]
        names = ['mmap', 'graphchi', 'ligra']
        scores = list()
        min_names = list()
        min_times = list()
        i = 0
        if(config.iloc[0,0]=='soc-sinawbeio-260M_edge' and algo=='BFS'):
            print("ttttttttt")
            config.iloc[0, 0]='bn-humanJung-2015_87101705_200M'

        print(config.iloc[0, 1])
        print(config.iloc[0, 2])
        print(config.iloc[0, 0])
        for tool in tools:
            test = tool[
                        (tool['Graph_name'] == config.iloc[0, 0])&(tool['Algorithm'] == algo)]
            print(test)
            line = tool[(tool['machine_cpu'] == config.iloc[0, 1]) &
                        (tool['machine_ram'] == config.iloc[0, 2]) &
                        (tool['Graph_name'] == config.iloc[0, 0]) &
                        (tool['Algorithm'] == algo)]
            print('line',line)
            if line.empty:
                i = i + 1
                continue


            min_names.append(names[i])
            min_times.append(float(line.iloc[0]['exec_time(s)']))
            i = i + 1



            score = float(line.iloc[0]['exec_time(s)']) * self.time_weight + float(
                line.iloc[0]['peak_memory']) * self.ram_weight
            scores.append(score)


        print("scores:", scores)

        min_value = min(scores)
        min_index = scores.index(min_value)
        print(min_names)
        #return min_names[min_index],config

        return min_names[min_index], min_times[min_index],config