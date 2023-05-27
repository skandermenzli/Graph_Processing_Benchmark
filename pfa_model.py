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
        print(train_data)
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
        print("predict")
        print("distance: ", distances)
        print("indice", indices)
        print(self.res.iloc[[indices[0, 0]]])
        config = self.res.iloc[[indices[0, 0]]]
        # return row
        # print(type(row))

        graphchi = pd.read_csv('csv/GraphChi.csv')
        mmap = pd.read_csv('csv/Mmap.csv')
        tools = [mmap, graphchi]
        names = ['mmap', 'graphchi', 'ligra']
        scores = list()
        print("aaaaaaaaaa")
        for tool in tools:
            line = tool[(tool['machine_cpu'] == config.iloc[0, 1]) &
                        (tool['machine_ram'] == config.iloc[0, 2]) &
                        (tool['Graph_name'] == config.iloc[0, 0]) &
                        (tool['Algorithm'] == algo)]
            if line.empty:
                continue
            score = float(line.iloc[0]['exec_time(s)']) * self.time_weight + float(
                line.iloc[0]['peak_memory']) * self.ram_weight
            scores.append(score)

        print("scores:", scores)

        """ chi = graphchi[(graphchi['machine_cpu']==config.iloc[0,1])&
                     (graphchi['machine_ram']==config.iloc[0,2])&
                     (graphchi['Graph_name']==config.iloc[0,0])&
                     (graphchi['Algorithm']=='PR10')]
        mm = mmap[(mmap['machine_cpu']==config.iloc[0,1])&
                     (mmap['machine_ram']==config.iloc[0,2])&
                     (mmap['Graph_name']==config.iloc[0,0])&
                     (mmap['Algorithm']=='PR10')]

        mm_score = float(mm.iloc[0]['exec_time(s)'])*self.time_weight + mm.iloc[0]['peak_memory']*self.ram_weight
        print("mmap score:",mm_score)
        chi_score = chi.iloc[0]['exec_time(s)']*self.time_weight + chi.iloc[0]['peak_memory']*self.ram_weight
        print("chi_socre:",chi_score)"""

        min_value = min(scores)
        min_index = scores.index(min_value)
        return names[min_index],config