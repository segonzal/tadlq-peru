from itertools import combinations
from itertools import chain
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import pygraphviz as pgv
import scipy.cluster.hierarchy as sch
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import roc_curve, RocCurveDisplay, accuracy_score


def join_multiindex(df, k='\t'):
    df = df.columns.to_frame().reset_index(drop=True)
    for c in df.columns:
        df.loc[df.loc[:, c].str.startswith('Unnamed:'), c] = ""
    return df.agg(lambda x: k.join(x), axis=1)


def plot_violin(df, cx, cy, ax, palette, h=2, dh=None):
    groups = df[cx].unique()
    # palette = [palette[c] for c in groups]
    sns.violinplot(data=df, y=cy, x=cx, hue=cx, ax=ax, inner='quart', palette=palette, legend=False)

    t = kruskal(*[df.loc[df[cx] == g, cy] for g in groups], nan_policy='omit', axis=0, keepdims=False)

    if dh is None:
        dh = h

    y = df[cy].max() + h
    for x1, x2 in combinations(range(len(groups)), 2):
        stat, pval = mannwhitneyu(df.loc[df[cx] == groups[x1], cy], df.loc[df[cx] == groups[x2], cy])
        if pval <= 0.0001:
            label = '***'
        elif pval <= 0.001:
            label = '**'
        elif pval <= .05:
            label = '*'
        else:
            label = 'ns'
        if label != 'ns':
            ax.plot([x1, x1, x2, x2], [y, y + dh * .5, y + dh * .5, y], lw=1.5, c='k')
            ax.text((x1 + x2) * .5, y + dh * .5, label, ha='center', va='bottom')
            y += 2.5 * dh
    return t


def cluster_corr(corr_array, inplace=False):
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


def plot_three_components(X, y, cmap, label=None, style=None, markers=None, ncols=3):
    fig = plt.figure(figsize=(8, 8))

    axs = np.array([None] * 4).reshape((2, 2))

    axs[0, 0] = fig.add_subplot(2, 2, 1)
    axs[0, 1] = fig.add_subplot(2, 2, 2)
    axs[1, 0] = fig.add_subplot(2, 2, 3)
    axs[1, 1] = fig.add_subplot(2, 2, 4, projection='3d')
    axs[1, 1].set_box_aspect(aspect=None, zoom=0.7)

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, style=style, ax=axs[0, 0], palette=cmap, markers=markers)
    sns.scatterplot(x=X[:, 2], y=X[:, 1], hue=y, style=style, ax=axs[0, 1], palette=cmap, markers=markers)
    sns.scatterplot(x=X[:, 0], y=X[:, 2], hue=y, style=style, ax=axs[1, 0], palette=cmap, markers=markers)

    if markers is None:
        markers = {None: 'o'}

    if style is not None:
        style = np.array(style)

    # unique_y = np.unique(y).tolist()
    # hue = [unique_y.index(j) for j in y]

    c = np.array([cmap[j] for j in y])
    for m in markers:
        if m is None:
            _x = X[:, 0]
            _y = X[:, 1]
            _z = X[:, 2]
            _c = c
        else:
            mask = style == m
            _x = X[mask, 0]
            _y = X[mask, 1]
            _z = X[mask, 2]
            _c = c[mask]
        axs[1, 1].scatter(_x, _y, _z, c=_c, marker=markers[m])

    if label is not None:
        axs[0, 0].set_xlabel(label[0])
        axs[0, 0].set_ylabel(label[1])

        axs[0, 1].set_xlabel(label[2])
        axs[0, 1].set_ylabel(label[1])

        axs[1, 0].set_xlabel(label[0])
        axs[1, 0].set_ylabel(label[2])

        axs[1, 1].set_xlabel(label[0])
        axs[1, 1].set_ylabel(label[1])
        axs[1, 1].set_zlabel(label[2])

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower right', ncols=ncols)

    for ax in axs.flatten()[:3]:
        ax.get_legend().remove()

    return fig, axs


def plot_correlation_matrix(df, ordered=False, sz=1.5):
    sz *= len(df.columns)
    corr_ = df.corr()

    if ordered:
        corr_ = cluster_corr(corr_)
        # corrsum = (-1 * np.sum(corr_, axis=0)).tolist()
        # idx = np.arange(len(corrsum)).tolist()
        # idx.sort(key=lambda i: corrsum[i])

        # corr_ = df.loc[:, df.columns[idx]].corr()

    fig, ax = plt.subplots(figsize=(sz, sz))
    g = sns.heatmap(data=corr_,
                    annot=True,
                    center=0,
                    square=True,
                    ax=ax,
                    vmin=0,
                    vmax=1,
                    cbar_kws={'shrink': 0.7, 'format': mtick.PercentFormatter(1.0, decimals=None)})


def explined_variance_plot(pca, n_factors=None, ax=None, max_comp=None):
    compmat = pca.components_.T
    w, h = compmat.shape[:2]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 2))

    x_ticks = list(map(lambda x: f"PC-{x + 1:d}", range(pca.n_components)))
    y_a = np.cumsum(pca.explained_variance_ratio_)
    y_b = pca.explained_variance_ratio_

    if max_comp is not None:
        x_ticks = x_ticks[:max_comp]
        y_a = y_a[:max_comp]
        y_b = y_b[:max_comp]

    ax.plot(x_ticks, y_a, label='Cumulative\nexplained variance')
    ax.bar(ax.get_xticks(), y_b, color='c', label='Individual\nexplained variance')

    if n_factors is None:
        n_factors = np.argmax(np.abs(np.cumsum(pca.explained_variance_ratio_) - 0.8) < 0.1)

    ax.axhline(y=[0.8], linestyle='--', color='red')
    ax.axvline(x=n_factors - 1, linestyle='--', color='yellow')

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=None))
    ax.tick_params(axis='x', labelrotation=90)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plot_components(pca, n_factors, cols, group_columns=None):
    xlabels = list(map(lambda x: f"PC-{x + 1:d}", range(n_factors)))

    if group_columns is None:
        ylabels = cols
    else:
        group_columns_inv = {v: k for k, vlist in group_columns.items() for v in vlist}
        ylabels = [group_columns_inv[c] + '\n' + c for c in cols]

    comp = pca.components_.T[:, :n_factors]
    vmax = np.abs(comp).max()
    fig, ax = plt.subplots(figsize=(len(cols), n_factors))
    g = sns.heatmap(data=comp.T,
                    center=0,
                    vmin=-vmax,
                    vmax=vmax,
                    annot=True,
                    fmt=".2f",
                    xticklabels=ylabels,
                    yticklabels=xlabels,
                    ax=ax)


def get_groups(aSeries):
    m = aSeries.ne(aSeries.shift())
    res = pd.DataFrame({'index': aSeries[m], 'starts': np.arange(len(aSeries))[m]}).drop('index', axis=1)
    res['ends'] = res['starts'].shift(-1, fill_value=len(aSeries))
    return res


def check_nulls(df, ylabel, figsize=None, aspect='auto', cmap=None):
    vmap = {'NULL': 0, 'OK': 1, 'NA/DK': 2}

    if cmap is None:
        cmap = plt.get_cmap(name='Paired')

    k = np.ones((*df.shape, 1)) * vmap['OK']
    m = np.ones((*df.shape, 1)) * np.float32(cmap(vmap['OK']))[np.newaxis, :]

    k[df.isna()] = vmap['NULL']
    m[df.isna(), :] = cmap(vmap['NULL'])
    
    for i, c in enumerate(df.columns):
        if df[c].dtype.kind in 'biufc':
            m[df[c] < 0, i, :] = np.float32(cmap(vmap['NA/DK']))[np.newaxis, :]
            k[df[c] < 0, i] = vmap['NA/DK']
    
    ycol = df.columns.to_list().index(ylabel)
    
    idx = list(range(df.shape[0]))
    idx.sort(key=lambda i: (df.iloc[i, ycol], ''.join(map(str, k[i, :]))))
    
    m = m[idx, :]
    k = k[idx, :]
    df = df.iloc[idx, :]

    df = df.set_index(ylabel)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    # im = ax.imshow(m, aspect=aspect)
    im = ax.matshow(m, aspect=aspect, cmap=cmap)

    ax.set_xticks(np.arange(len(df.columns)) + 0.5, labels=df.columns)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="left", rotation_mode="anchor")

    unique_vals = df.index.unique()
    ylabel = df.index.to_series().map({k: v for v, k in enumerate(unique_vals)}).astype(np.uint8)
    ylabel = get_groups(ylabel)

    ax.set_yticks(ylabel['ends'], labels=['' for _ in ylabel['ends']], minor=False)

    ylabel, ypos = zip(*[(lbl, start + (end - start) / 2) for lbl, (start, end) in ylabel.iterrows()])

    ax.set_yticks(ypos, labels=ylabel, minor=True)

    ax.legend(handles=[mpatches.Patch(color=cmap(v), label=k) for k, v in vmap.items()], loc='center left',
              bbox_to_anchor=(1, 0.5))


def run_classifier(X_train, y_train, X_test, y_test, cols, pca=None, max_depth=10, n_estimators=10, random_seed=0, color_palette=None):
    classifier = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features='sqrt',
                                        random_state=random_seed)
    classifier.fit(X_train, y_train)

    # Feature importances
    fig, ax = plt.subplots()

    sorted_idx = classifier.feature_importances_.argsort()[::-1]
    sns.barplot(y=[cols[i] for i in sorted_idx], x=classifier.feature_importances_[sorted_idx], ax=ax)

    # Class feat imp
    # plot_class_feature_importances(X_test, y_test, cols, classifier.feature_importances_)

    y_pre = classifier.predict(X_test)

    mask = (y_pre != y_test)
    acc = accuracy_score(y_test, y_pre)
    # acc = np.sum(~mask) / len(mask)
    print(f"Acc: {acc}")

    # Plot PCA
    if pca is not None:
        x_pca = pca.transform(X_test)

        style = ['Wrong' if m else 'Correct' for m in mask]
        markers = {'Correct': 'o', 'Wrong': 'X'}

        fig, axs = plot_three_components(x_pca, y_test, color_palette, style=style, markers=markers,
                                         label=['PCA-1', 'PCA-2', 'PCA-3'])

    return classifier


def get_best_estimator(classifier, X_test, y_test):
    cls = classifier.classes_.tolist()
    y_test = np.float32([cls.index(i) for i in y_test])
    acc = []
    for e in classifier.estimators_:
        y_pred = e.predict(X_test)
        mask = (y_pred != y_test)
        acc.append(np.sum(~mask) / len(mask))
    i = np.argmax(acc)
    print('Max Acc:', acc[i])
    return classifier.estimators_[i]


def plot_decision_Tree(tree, feature_names, threshold_mean, threshold_std, label=None, class_names=None):
    if class_names is None:
        class_names = tree.classes_

    stack = [0]

    thresholds = [(t * threshold_std[i]) + threshold_mean[i] for i, t in zip(tree.feature, tree.threshold)]

    nodes = []
    edges = []
    is_leaf = []

    while len(stack) > 0:
        node_id = stack.pop()

        n_label = f"node_{node_id}"

        if tree.children_left[node_id] != tree.children_right[node_id]:
            nodes.append((n_label, f"{feature_names[tree.feature[node_id]]} <= {thresholds[node_id]:.2f}"))
            is_leaf.append(False)
            edges.append((n_label, f"node_{tree.children_right[node_id]}"))
            edges.append((n_label, f"node_{tree.children_left[node_id]}"))

            stack.append(tree.children_right[node_id])
            stack.append(tree.children_left[node_id])
        else:
            node_values = tree.value[node_id][0]
            if np.count_nonzero(node_values) == 1:
                content = class_names[np.where(node_values != 0)[0][0]]
            else:
                t = node_values.sum()
                content = " ".join([f"{n}={v / t:.0%}" for n, v in zip(class_names, node_values)])
            nodes.append((n_label, content))
            is_leaf.append(True)

    A = pgv.AGraph(directed=True, strict=True, rankdir="TB")
    for (n, l), il in zip(nodes, is_leaf):
        A.add_node(n, label=l, shape='ellipse' if il else 'box')
    for ns, ne in edges:
        A.add_edge(ns, ne)

    if label is not None:
        A.graph_attr["label"] = label

    A.layout("dot")
    return A.draw(format='png')


def train_random_forest(X, y, test_size, max_depth, n_estimators, n_splits):
    #classes = {c:i for i, c in enumerate(sorted(y.unique()))}
    #y = y.map(classes)

    # Reserve a portion of the dataset to validate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, stratify=y)
    
    clf = classifier = RandomForestClassifier(max_depth=max_depth,
                                              n_estimators=n_estimators,
                                              max_features='sqrt',
                                              random_state=0)
    cv = StratifiedKFold(n_splits=n_splits,
                         shuffle=True,
                         random_state=0)
    
    model =  cross_validate(clf, X_train, y_train, cv=cv, scoring='balanced_accuracy', return_estimator=True, return_train_score=True, return_indices=True)
    
    # From all the estimators, pick the ones with best accuracy on the validation set
    y_test = y_test.to_numpy()
    trees = []
    acc = []
    for est in model['estimator']:
        #classes_ = list(est.classes_)
        
        for tree in est.estimators_:
            y_pred = est.predict(X_test)
            # y_pred = [classes_[i] for i in y_pred.astype(int)]
            tree.classes_ = est.classes_
            trees.append(tree)
            acc.append(accuracy_score(y_test, y_pred))
    acc = np.float32(acc)
    return model, trees, acc


def get_estimator_with_max_acc_and_least_leaves(val_acc, trees, ax=None):
    # Pick from the threes with max accuracy, the one with the least leaves
    max_acc = np.max(val_acc)
    max_acc_idx = np.where(val_acc == max_acc)[0].astype(int)
    tree_leaves = [trees[i].tree_.n_leaves for i in max_acc_idx]
    
    #ax = sns.histplot(x=tree_leaves, ax=ax)
    #ax.set(xlabel='Number of Leaves', ylabel='Count')
    
    return max_acc_idx[np.argmin(tree_leaves)]


def get_aggregate_feature_importances(trees):
    agg_feature_importances = sum([est.feature_importances_ for est in trees])
    return agg_feature_importances / np.sum(agg_feature_importances)

