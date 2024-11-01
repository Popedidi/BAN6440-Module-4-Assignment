publicBucket = "covid19-lake"  # the bucket reference
import boto3
from botocore import \
    UNSIGNED  # You'll need this to connect as anonymous. You could also pass your access key and secret
from botocore.client import Config
import pandas as pd
from IPython.display import display

s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Verify Access control
acl = s3_client.get_bucket_acl(Bucket=publicBucket)
owner = acl["Owner"]
grants = acl["Grants"]
print("Bucket owned by ", owner)
print("Bucket grants:")
for grant in grants:
    grantee = grant["Grantee"]
    permission = grant["Permission"]
    print("Grantee=", grantee, ", Permission=", permission)


    # List the objects in the bucket
    def list_bucket_objects(**kwargs):
        response = s3_client.list_objects_v2(**kwargs)
        continuation_token = response.get("NextContinuationToken")
        for obj in response.get("Contents"):
            key = obj.get("Key")
            size = obj.get("Size")
            storageclass = obj.get("StorageClass")
            print("Object found with key=", key, ", size=", size, ", S3 storage class=", storageclass)
        return continuation_token
args = dict(Bucket=publicBucket, MaxKeys=10)
continuation = list_bucket_objects(**args)
args["ContinuationToken"] = continuation
continuation = list_bucket_objects(**args)

args = dict(Bucket=publicBucket, MaxKeys=50, Prefix='rearc-covid-19')
list_bucket_objects(**args)

obj = s3_client.get_object(Bucket=publicBucket, Key="rearc-covid-19-testing-data/csv/us_daily/us_daily.csv")
df = pd.read_csv(obj.get("Body"))
display(df)

df.dtypes

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Prepare the data
# Select relevant numerical features for clustering
features = ['positive', 'negative', 'hospitalizedCurrently', 'inIcuCurrently', 'onVentilatorCurrently']
X = df[features].copy()

# Handle missing values
X = X.fillna(X.mean())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters using elbow method
inertias = []
silhouette_scores = []
K = range(2, 8)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')

# Plot silhouette scores
plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'rx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.tight_layout()
plt.show()

# Apply K-means with optimal k (let's use 3 clusters for this example)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze cluster characteristics
cluster_means = df.groupby('Cluster')[features].mean()
print("\nCluster Characteristics:")
print(cluster_means)

# Visualize clusters using first two features
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
                      c=df['Cluster'], cmap='viridis')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title('K-means Clustering Results')
plt.colorbar(scatter)
plt.show()

# Print cluster sizes
print("\nCluster Sizes:")
print(df['Cluster'].value_counts())


# Unit Tests
def test_kmeans_clustering():
    # Test cluster assignments
    assert df['Cluster'].nunique() == optimal_k, f"Expected {optimal_k} clusters but got {df['Cluster'].nunique()}"
    assert df['Cluster'].isin(range(optimal_k)).all(), "Cluster labels should be in range 0 to optimal_k-1"

    # Test cluster sizes
    cluster_sizes = df['Cluster'].value_counts()
    assert len(cluster_sizes) == optimal_k, "Each cluster should have at least one data point"

    # Test that features were properly scaled
    assert abs(X_scaled.mean()) < 0.0001, "Scaled features should have mean close to 0"
    assert abs(X_scaled.std() - 1) < 0.0001, "Scaled features should have std close to 1"

    # Test silhouette scores are in valid range
    assert all(-1 <= score <= 1 for score in silhouette_scores), "Silhouette scores should be between -1 and 1"

    return "All tests passed!"


# Run tests and display results
print("\nRunning unit tests...")
try:
    test_result = test_kmeans_clustering()
    print(test_result)
except AssertionError as e:
    print(f"Test failed: {str(e)}")
