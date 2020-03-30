from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def pca(config, X):
    scaled_data = StandardScaler().fit_transform(X)
    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    per_var = np.round(pca.explained_variance_ratio_ * 100, 1)
    labels = ['PC' + str(i) for i in range(1, (len(per_var)+1))]

    # Plot a scree graph
    plot_path = config.get('General', 'path') + '/plots'
    fig, ax = plt.subplots(1, 1, figsize=(40, 6))
    ax.bar(x=range(1, (len(per_var)+1)), height=per_var, tick_label=labels)
    ax.set_xticklabels(labels, rotation=90, ha="right")
    ax.set_ylabel('Percentage of explained variance')
    ax.set_xlabel('Principal component')
    ax.set_title('Scree plot')
    fig.savefig(plot_path + '/pca_scree_test.png')

    pca_df = pd.DataFrame(pca_data, columns=labels)
    loading_scores = pd.Series(pca.components_[0], index=X.columns)
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    top_50_x = sorted_loading_scores[:50].index.values
    # print(pca.explained_variance_ratio_)
    # print(loading_scores[top_50_x])

    # How many dimensions do we need for 95% variance ratio
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    # print(d)

    # Plot a scree graph with 95% cutoff
    fig, ax = plt.subplots(1, 1, figsize=(40, 6))
    ax.plot(labels, cumsum)
    ax.set_xlabel("Dimensions")
    ax.set_ylabel("Explained Variance")
    ax.grid()
    ax.plot([d, d], [0, 0.95], "k:")
    ax.plot([0, d], [0.95, 0.95], "k:")
    fig.savefig(plot_path + '/pca_scree_test2.png')
