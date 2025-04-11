from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Use PCA or Autoencoder to embed sequences
pca = PCA(n_components=2)
seq_features = pca.fit_transform(padded_seqs)

kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(seq_features)

# Visualize
import matplotlib.pyplot as plt
plt.scatter(seq_features[:, 0], seq_features[:, 1], c=clusters)
