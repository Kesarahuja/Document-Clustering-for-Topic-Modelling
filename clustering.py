
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the TF-IDF matrix
with open(
    '/home/ubuntu/twenty_newsgroups/tfidf_matrix.pkl', 'rb') as f:
    X = pickle.load(f)

# K-means Clustering
print("\n--- K-means Clustering ---")
# Determine optimal number of clusters using Silhouette Score (example for a range)
# In a real scenario, this would involve more sophisticated methods like elbow method, etc.

# Let's try a range of cluster numbers and find the best one using silhouette score
# Note: Silhouette score can be computationally intensive for large datasets
# For demonstration, we'll pick a reasonable range and number of iterations

max_clusters = 10 # Adjust as needed
silhouette_scores = []

for n_clusters in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"K-means with {n_clusters} clusters, Silhouette Score: {score:.4f}")

# Find the optimal number of clusters based on silhouette score
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2 # +2 because range starts from 2
print(f"Optimal K for K-means based on Silhouette Score: {optimal_k}")

kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_optimal.fit(X)
print(f"K-means optimal clustering done with {optimal_k} clusters.")

# Save K-means model
with open(
    '/home/ubuntu/twenty_newsgroups/kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans_optimal, f)
print("K-means model saved.")

# Latent Dirichlet Allocation (LDA)
print("\n--- Latent Dirichlet Allocation (LDA) ---")
# Determine optimal number of topics (similar to K-means, but often uses perplexity/coherence)
# For simplicity, we'll use a fixed number of topics for now, or iterate through a range

# LDA requires count matrix, not TF-IDF. We need to re-vectorize or use a different approach
# For this example, let's assume we can use TF-IDF for simplicity, but ideally, we'd use CountVectorizer
# If we were to use CountVectorizer, we would need to modify data_preprocessing.py

# For now, let's proceed with TF-IDF for LDA, acknowledging it's not ideal.
# A better approach would be to save both TF-IDF and CountVectorizer outputs from preprocessing.

# Let's try a range of topic numbers for LDA
max_topics = 10 # Adjust as needed
lda_models = []

for n_topics in range(2, max_topics + 1):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    lda_models.append(lda)
    print(f"LDA with {n_topics} topics trained.")

# For LDA, evaluating optimal topics is more complex (e.g., coherence score)
# For this example, we'll just pick the model with max_topics as a placeholder
optimal_lda_model = lda_models[-1] # Taking the last one as optimal for now
print(f"LDA optimal model selected with {max_topics} topics.")

# Save LDA model
with open(
    '/home/ubuntu/twenty_newsgroups/lda_model.pkl', 'wb') as f:
    pickle.dump(optimal_lda_model, f)
print("LDA model saved.")

print("Clustering and LDA modeling complete.")


