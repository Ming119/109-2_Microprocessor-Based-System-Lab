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
Therefore, do not use 'scipy.cluster.vq.kmeans()' on the developing, use `sklearn.cluster.KMeans()` instead.
For `cv2.kmeans()`, please use `cv.KMEANS_PP_CENTERS` as the `flags` on the developing.


## Reference
[Rock Paper Scissors Dataset](http://www.laurencemoroney.com/rock-paper-scissors-dataset/)(CC By 2.0 @Laurence Moroney lmoroney@gmail.com / laurencemoroney.com)
[DrGFreeman/rps-cv](https://github.com/DrGFreeman/rps-cv)([MIT LICENSE](https://github.com/DrGFreeman/rps-cv/blob/master/LICENSE))
[使用OpenCV与sklearn实现基于词袋模型(Bag of Word)的图像分类预测与搜索](https://cloud.tencent.com/developer/article/1165870)
[SVM+Sift+K-means实现图像分类（python）](https://blog.csdn.net/weixin_42486554/article/details/103732613)


## LICENSE
[GNU General Public License v3.0](./LICENSE)
