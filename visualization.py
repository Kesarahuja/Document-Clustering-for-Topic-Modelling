import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from sklearn.datasets import load_files

# Load data and models
with open(
    '/home/ubuntu/twenty_newsgroups/tfidf_matrix.pkl', 'rb') as f:
    X = pickle.load(f)
with open(
    '/home/ubuntu/twenty_newsgroups/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open(
    '/home/ubuntu/twenty_newsgroups/kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)
with open(
    '/home/ubuntu/twenty_newsgroups/lda_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)

data_path = '/home/ubuntu/twenty_newsgroups/20_newsgroups'
newsgroups_data = load_files(data_path, encoding=
    'latin1', random_state=42)

# --- Visualization 1: K-means Cluster Distribution ---
print("Generating K-means Cluster Distribution...")
plt.figure(figsize=(10, 6))
sns.countplot(x=kmeans_model.labels_)
plt.title('K-means Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Number of Documents')
plt.savefig('/home/ubuntu/twenty_newsgroups/kmeans_cluster_distribution.png')
plt.close()
print("K-means Cluster Distribution saved to kmeans_cluster_distribution.png")

# --- Visualization 2: LDA Topic Distribution ---
print("Generating LDA Topic Distribution...")
lda_document_topic_distribution = lda_model.transform(X)
lda_predicted_topics = lda_document_topic_distribution.argmax(axis=1)
plt.figure(figsize=(10, 6))
sns.countplot(x=lda_predicted_topics)
plt.title('LDA Topic Distribution')
plt.xlabel('Topic')
plt.ylabel('Number of Documents')
plt.savefig('/home/ubuntu/twenty_newsgroups/lda_topic_distribution.png')
plt.close()
print("LDA Topic Distribution saved to lda_topic_distribution.png")

# --- Visualization 3: Word Clouds for LDA Topics ---
print("Generating Word Clouds for LDA Topics...")
tfidf_feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda_model.components_):
    wordcloud = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies({
        tfidf_feature_names[i]: topic[i] for i in topic.argsort()[:-50 - 1:-1]
    })
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'LDA Topic {topic_idx} Word Cloud')
    plt.savefig(f'/home/ubuntu/twenty_newsgroups/lda_topic_{topic_idx}_wordcloud.png')
    plt.close()
print("Word Clouds for LDA Topics saved.")

# --- Visualization 4: t-SNE/PCA for Document Clustering ---
print("Generating t-SNE/PCA for Document Clustering...")
# Reduce dimensions for visualization
# Using PCA first for dimensionality reduction before t-SNE for better performance
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X.toarray()) # Convert sparse matrix to dense for PCA

tsne = TSNE(n_components=2, random_state=42, n_iter_without_progress=300)
X_tsne = tsne.fit_transform(X_pca)

# Create DataFrame for plotting
df_tsne = pd.DataFrame(X_tsne, columns=['Component 1', 'Component 2'])
df_tsne['KMeans_Cluster'] = kmeans_model.labels_
df_tsne['LDA_Topic'] = lda_predicted_topics
df_tsne['True_Label'] = newsgroups_data.target
df_tsne['True_Label_Name'] = [newsgroups_data.target_names[i] for i in newsgroups_data.target]

# Plot t-SNE with K-means clusters
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='Component 1', y='Component 2', hue='KMeans_Cluster', palette='viridis', 
    data=df_tsne, legend='full', alpha=0.7, s=10
)
plt.title('t-SNE Visualization of K-means Clusters')
plt.savefig('/home/ubuntu/twenty_newsgroups/tsne_kmeans_clusters.png')
plt.close()
print("t-SNE Visualization of K-means Clusters saved to tsne_kmeans_clusters.png")

# Plot t-SNE with LDA topics
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='Component 1', y='Component 2', hue='LDA_Topic', palette='viridis', 
    data=df_tsne, legend='full', alpha=0.7, s=10
)
plt.title('t-SNE Visualization of LDA Topics')
plt.savefig('/home/ubuntu/twenty_newsgroups/tsne_lda_topics.png')
plt.close()
print("t-SNE Visualization of LDA Topics saved to tsne_lda_topics.png")

# Plot t-SNE with True Labels
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='Component 1', y='Component 2', hue='True_Label_Name', palette='tab20', 
    data=df_tsne, legend='full', alpha=0.7, s=10
)
plt.title('t-SNE Visualization of True Labels')
plt.savefig('/home/ubuntu/twenty_newsgroups/tsne_true_labels.png')
plt.close()
print("t-SNE Visualization of True Labels saved to tsne_true_labels.png")

print("All visualizations generated.")

