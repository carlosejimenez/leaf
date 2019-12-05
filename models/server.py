import numpy as np
from clustering import *
from math import floor

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY

class Server:
    
    def __init__(self, client_model):
        self.client_model = client_model
        self.model = client_model.get_params()
        self.selected_clients = []
        # self.updates = []
        self.gradients = []
        self.personalized_models = None
        self.clusterer = None

    def get_dense_gradients(self, client_grads, k=6):
        if type(client_grads[0]) != tuple:
            client_grads = [client_grads]
        return list(map(lambda x: x[2][k].flatten(), client_grads))

    def create_personalized_models(self, avail_clients, personalization_percent, num_epochs, batch_size, minibatch, cluster_algo, num_clusters):
        print('i got to personalized modelss')
        num_clients = floor(len(avail_clients) * personalization_percent)
        self.select_clients(0, possible_clients=avail_clients, num_clients=num_clients)
        self.train_model(num_epochs=num_epochs, batch_size=batch_size, minibatch=minibatch)
        grads = self.get_dense_gradients(self.gradients)
        self.clusterer = Clustering(grads, cluster_algo, num_clusters)
        self.cluster_count = self.clusterer.number_clusters()
        self.personalized_models = [self.model.copy() for _ in range(self.cluster_count)]

    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """

        self.gradients = []

        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}

        if self.personalized_models is not None:
            for _ in range(len(self.personalized_models)):
                self.gradients.append([])

        for c in clients:
            c.model.set_params(self.model)
            # comp, num_samples, update, grads = c.train(num_epochs, batch_size, minibatch)
            comp, num_samples, grads = c.train(num_epochs, batch_size, minibatch)

            if self.personalized_models is not None:
                client_tuple = (num_samples, c.id, grads)
                processed_grad = self.get_dense_gradients(client_tuple)
                cluster_id = self.clusterer.predict(processed_grad)[0]
                c.model.set_params(self.personalized_models[cluster_id])
                comp, num_samples, grads = c.train(num_epochs, batch_size, minibatch)

            sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp

            # self.updates.append((num_samples, update))
            if self.personalized_models is not None:
                self.gradients[cluster_id].append((num_samples, c.id, grads))
            else:
                self.gradients.append((num_samples, c.id, grads))

        return sys_metrics

    def update_model(self):
        # total_weight = 0.
        # base = [0] * len(self.updates[0][1])
        # for (client_samples, client_model) in self.updates:
        #     total_weight += client_samples
        #     for i, v in enumerate(client_model):
        #         base[i] += (client_samples * v.astype(np.float64))
        # averaged_soln = [v / total_weight for v in base]

        if self.personalized_models is None:
            avg_gradients = self.get_averaged_gradients(self.gradients)
            self.model = [m - p for m, p in zip(self.model, avg_gradients)]
        else:
            for cluster_id in range(len(self.personalized_models)):
                curr_grads = self.gradients[cluster_id]
                if len(curr_grads) == 0:
                    continue
                avg_gradients = self.get_averaged_gradients(curr_grads)
                self.personalized_models[cluster_id] = [m - p for m, p in zip(self.personalized_models[cluster_id], avg_gradients)]

        # self.model = averaged_soln  # self.model -= lr * avg_gradients
        # self.updates = []
        #self.gradients = []


    def test_model(self, clients_to_test, set_to_use='test', num_epochs=1, batch_size=10, minibatch=None):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            client.model.set_params(self.model)

            if self.personalized_models is not None:
                _, num_samples, grads = client.train(num_epochs, batch_size, minibatch)
                client_tuple = (num_samples, client.id, grads)
                processed_grad = self.get_dense_gradients(client_tuple)
                cluster_id = self.clusterer.predict(processed_grad)[0]
                client.model.set_params(self.personalized_models[cluster_id])

            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics
        
        return metrics

    def get_averaged_gradients(self, gradients):
        grads = [np.zeros_like(g) for g in gradients[0][2]]
        total_weight = 0
        for (client_samples, c_id, client_grads) in gradients:
            total_weight += client_samples
            for i, grad in enumerate(client_grads):
                grads[i] += client_samples * grad * self.client_model.lr
        avg_gradients = [g / total_weight for g in grads]

        return avg_gradients

    def get_client_gradients(self):
        client_grads = self.gradients.copy()
        for i in range(len(client_grads)):
            client_grads[i][2] = np.concatenate(list(map(lambda x: x.flatten().reshape(-1), client_grads[i][2])))
        return client_grads

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model.set_params(self.model)
        model_sess = self.client_model.sess
        return self.client_model.saver.save(model_sess, path)

    def close_model(self):
        self.client_model.close()