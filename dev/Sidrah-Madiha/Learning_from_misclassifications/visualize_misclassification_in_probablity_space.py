import matplotlib.pyplot as plt
import numpy as np


def visualize_missclassify_in_probablity_space(
    classifier, X_train, X_test, y_train, y_test
):
    """ returns subplots for each class showing probablity space and missclassified points on it"""

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
    #                      np.arange(y_min, y_max, 0.1))
    #     probas = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    #     probas= probas.reshape(xx.shape)
    #     plt.contourf(xx, yy, probas, cmap=pl.cm.Paired)
    #     plt.axis('off')

    plt.figure(figsize=(5 * 4, 2))
    plt.subplots_adjust(bottom=0.2, top=0.95)
    #     xx = np.linspace(3, 9,len(X_test)) #int(len(X_test)/2)
    #     yy = np.linspace(1, 5, len(X_test)).T #int(len(X_test)/2)
    #     xx, yy = np.meshgrid(xx, yy)
    #     Xfull = np.c_[xx.ravel(), yy.ravel()]
    probas = classifier.predict_proba(X_test)
    n_classes = np.unique(y_pred).size
    for k in range(n_classes):
        plt.subplot(1, n_classes, k + 1)
        plt.title("Class %d" % k)
        imshow_handle = plt.imshow(
            np.reshape(probas[:, k], (-1, 2)),
            extent=(x_min, x_max, y_min, y_max),
            origin="lower",
        )
        plt.xticks(())
        plt.yticks(())
        idx = (y_pred != y_test) & (y_pred == k)
        if idx.any():
            plt.scatter(
                X_test[idx, 0], X_test[idx, 1], marker="o", c="w", edgecolor="k"
            )
    ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax, orientation="horizontal")

    plt.show()
