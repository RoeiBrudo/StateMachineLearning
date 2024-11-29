class AutomataWrapper:
    def __init__(self, automata, X_disc, pred_disc):
        self.automata = automata
        self.X_disc = X_disc
        self.pred_disc = pred_disc

    def predict(self, X):
        data_clusters = self.X_disc.predict_from_vectors(X, to_vectors=False)
        automata_prediction = self.automata.predict(data_clusters)
        return self.pred_disc.get_clusters_representatives([automata_prediction])


