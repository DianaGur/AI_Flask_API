from sklearn.cluster import DBSCAN

def cluster_embeddings(embeddings):
    model = DBSCAN(metric='cosine', eps=0.5, min_samples=2)
    labels = model.fit_predict(embeddings)
    return labels
