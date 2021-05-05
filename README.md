# 109-2_Microprocessor-Based-System-Lab
## Midterm Project 期中專案

## 第十二組
### 資工二 108590049 符芷琪
### 資工二108590050 李浩銘

#### Algorithms
> BoW Model(SIFT + KMeans) -> SVM Model

#### K-Means Algorithm
There are 3 K-Means Algorithm, `cv2.kmeans()`, `sklearn.cluster.KMeans()`, `scipy.cluster.vq.kmeans()`, respectively.

`cv2.kmeans()`
    criteria = cv2.kmeans(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1);
    flags = cv2.KMEANS_RANDOM_CENTERS;

    compactness, labels, centers = cv2.kmeans(features, util.K_MEANS_CLUSTERS, None, criteria, 20, flags);

    return centers;

`sklearn.cluster.KMeans()`
    kmeans = KMeans(n_clusters=util.K_MEANS_CLUSTERS, random_state=0).fit(features);

    return kmeans.cluster_centers_;

`scipy.cluster.vq.kmeans()`
    centers, variance = kmeans(features, util.K_MEANS_CLUSTERS);

    return centers;

The accuracy of 'scipy.cluster.vq.kmeans()' is the best, but the algorithm does not accept 'random_state' as a parameter.
The output will be different every time.
Therefore, do not use 'scipy.cluster.vq.kmeans()' on the developing, use `cv2.kmeans()` or `sklearn.cluster.KMeans()` instead.

## LICENSE
[GNU General Public License v3.0](./LICENSE)
