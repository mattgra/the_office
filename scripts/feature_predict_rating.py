import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from core import feature_engineering as fe

if __name__ == "__main__":

    df = fe.get_merged_dataframes()
    df = fe.extract_features(df)
    df = df.fillna(0)
    df['rating'] = pd.qcut(df['rating'], 3, labels=['low', 'medium', 'high'])

    # 1) Train / Test split
    df_train, df_test = train_test_split(df, test_size=0.2)

    # 2) PCA Analysis (with scaling beforehand - otherwise dimensions have different variance)
    x_train = df_train.drop(columns="rating").values
    y_train = df_train["rating"].values
    x_test = df_test.drop(columns="rating").values
    y_test = df_test["rating"].values

    # 2A) Standardizing the features
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # 2B) PCA
    pca = PCA(n_components=2)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    # principalDf = pd.DataFrame(data=principalComponents, columns=["principal component 1", "principal component 2"])


    # =================

    # Code source: Gaël Varoquaux
    #              Andreas Müller
    # Modified for documentation by Jaques Grobler
    # License: BSD 3 clause
    #
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import ListedColormap
    # from sklearn.model_selection import train_test_split
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.datasets import make_moons, make_circles, make_classification
    # from sklearn.neural_network import MLPClassifier
    # from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.svm import SVC
    # from sklearn.gaussian_process import GaussianProcessClassifier
    # from sklearn.gaussian_process.kernels import RBF
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    # from sklearn.naive_bayes import GaussianNB
    # from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    #
    # h = 0.02  # step size in the mesh
    #
    # names = [
    #     "Nearest Neighbors",
    #     "Linear SVM",
    #     "RBF SVM",
    #     "Gaussian Process",
    #     "Decision Tree",
    #     "Random Forest",
    #     "Neural Net",
    #     "AdaBoost",
    #     "Naive Bayes",
    #     "QDA",
    # ]
    #
    # classifiers = [
    #     KNeighborsClassifier(3),
    #     SVC(kernel="linear", C=0.025),
    #     SVC(gamma=2, C=1),
    #     GaussianProcessClassifier(1.0 * RBF(1.0)),
    #     DecisionTreeClassifier(max_depth=5),
    #     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #     MLPClassifier(alpha=1, max_iter=1000),
    #     AdaBoostClassifier(),
    #     GaussianNB(),
    #     QuadraticDiscriminantAnalysis(),
    # ]
    #
    # figure = plt.figure(figsize=(27, 9))
    # i = 1
    #
    # X_train = x_train
    # X_test = x_test
    # X = np.concatenate([X_train,X_test])
    #
    # x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    # y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #
    # # just plot the dataset first
    # cm = plt.cm.RdBu
    # cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    # ax = plt.subplot(1, len(classifiers) + 1, i)
    # ax.set_title("Input data")
    # # Plot the training points
    # c_train = y_train.map({'low': 0, 'medium': 1, 'high': 2})
    # ax.scatter(X_train[:, 0], X_train[:, 1], c=c_train, cmap=cm_bright, edgecolors="k")
    # # Plot the testing points
    # c_test = y_test.map({'low': 0, 'medium': 1, 'high': 2})
    # ax.scatter(X_test[:, 0], X_test[:, 1], c=c_test, cmap=cm_bright, alpha=0.6, edgecolors="k")
    # ax.set_xlim(xx.min(), xx.max())
    # ax.set_ylim(yy.min(), yy.max())
    # ax.set_xticks(())
    # ax.set_yticks(())
    # # iterate over classifiers
    # for name, clf in zip(names, classifiers):
    #     i+=1
    #
    #     print("Running", name)
    #     ax = plt.subplot(1, len(classifiers) + 1, i)
    #     clf.fit(X_train, y_train)
    #     score = clf.score(X_test, y_test)
    #
    #     # Plot the decision boundary. For that, we will assign a color to each
    #     # point in the mesh [x_min, x_max]x[y_min, y_max].
    #     if hasattr(clf, "decision_function"):
    #         Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    #     else:
    #         Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    #
    #     # Put the result into a color plot
    #     # Z = Z.reshape(xx.shape)
    #     # ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
    #
    #     # Plot the training points
    #     ax.scatter(X_train[:, 0], X_train[:, 1], c=c_train, cmap=cm_bright, edgecolors="k")
    #     # Plot the testing points
    #     ax.scatter(
    #         X_test[:, 0],
    #         X_test[:, 1],
    #         c=c_test,
    #         cmap=cm_bright,
    #         edgecolors="k",
    #         alpha=0.6,
    #     )
    #
    #     ax.set_xlim(xx.min(), xx.max())
    #     ax.set_ylim(yy.min(), yy.max())
    #     ax.set_xticks(())
    #     ax.set_yticks(())
    #     ax.set_title(name)
    #     ax.text(
    #         xx.max() - 0.3,
    #         yy.min() + 0.3,
    #         ("%.2f" % score).lstrip("0"),
    #         size=15,
    #         horizontalalignment="right",
    #     )
    #
    # plt.tight_layout()
    # plt.show()
