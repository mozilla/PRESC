"""Basic general exploration of datasets.

This module accepts a database filename in CSV format and outputs a
series of generic dataset descriptors. It assumes that the label is in
the last column of the dataset.

Example:
    python3 data_exploration.py ../../datasets/generated.csv
"""

import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report


def load_file(dataset_file):
    """Load CSV as pandas dataset.

    Parameters:
        datasetfile (str): Path to dataset file.

    Returns:
        dataset (DataFrame): Pandas dataset.
    """
    dataset_name = os.path.basename(dataset_file)
    dataset_name = dataset_name.split(".")[0]

    dataset = pd.read_csv(dataset_file)

    return dataset, dataset_name


def create_folder(dataset_name):
    """Creates a folder with the dataset file name, if it doesn't exist.

    Parameters:
        dataset_name (str): Dataset filename without path nor extension.

    Returns:
        None
    """
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)


def table_descriptors(dataset):
    """Basic descriptors and statistics of the dataset.

    This module displays basic information on the dataset: shape,
    number of columns, column names, number of classes, classes names,
    number of samples for each class, the largest ratio if it's 1.1 or
    larger and reports if the sample is imbalanced, dataset head with
    the columns and the first and last rows, and basic statistics on
    the columns.

    Parameters:
        dataset (DataFrame): Pandas dataset.

    Returns:
        None
    """
    print("\nDataset shape: ", dataset.shape)

    print("\nThere are %s columns: " % len(list(dataset)))
    print(list(dataset))

    labels = list(dataset.iloc[:, -1].unique())
    same_type = all(isinstance(x, type(labels[0])) for x in labels)
    if same_type:
        labels = sorted(labels)

    print("\nThis dataset has %s classes: " % len(labels))
    print(labels)

    rows_with_label = []
    for label_name in labels:
        rows_with_label += [len(dataset.loc[dataset.iloc[:, -1] == label_name])]
    print("Each class has %s elements, respectively." % rows_with_label)

    # Reports on imbalance of the sample
    imbalanced, maximum_imbalance = weight_advisor(dataset)
    if imbalanced:
        print(
            "Your dataset is imbalanced. There is at least one class "
            f"with {maximum_imbalance:.1f} times more samples than another."
        )
    elif maximum_imbalance >= 1.1:
        print(
            f"There is a class with {maximum_imbalance:.1f} times more "
            "samples than another."
        )

    print("\nDataset head: ", dataset.head)

    print("\nDataset descriptive statistics: \n", dataset.describe())


def weight_advisor(dataset):
    """Reports on imbalance of the sample.

    Research on imbalanced classes often considers imbalanced to
    mean a minority class of 10% to 20%. This function provides a
    warning if this proportion is reached.

        Parameters:
            dataset (DataFrame): Pandas dataset.

        Returns:
            imbalanced (bool): True if ratio (maximum_imbalance) is >= 4.
            maximum_imbalance (float): Largest ratio between the
                number of samples of classes.
    """
    labels = list(dataset.iloc[:, -1].unique())
    same_type = all(isinstance(x, type(labels[0])) for x in labels)
    if same_type:
        labels = sorted(labels)

    rows_with_label = []
    for label_name in labels:
        rows_with_label += [len(dataset.loc[dataset.iloc[:, -1] == label_name])]

    imbalanced = False
    maximum_imbalance = 1.0

    for volumeA in rows_with_label:
        for volumeB in rows_with_label:

            if (volumeA / volumeB) > maximum_imbalance:
                maximum_imbalance = volumeA / volumeB
            if (volumeB / volumeA) > maximum_imbalance:
                maximum_imbalance = volumeB / volumeA

    if maximum_imbalance >= 4.0:
        imbalanced = True

    return imbalanced, maximum_imbalance


def labels_to_colors(dataset):
    """Obtain classes and create its corresponding color map.

    This function determines the classes in the datset, sorts them, and
    creates a color map assigning a color tuple to each clas.

    Parameters:
        dataset (DataFrame): Pandas dataset.

    Returns:
        labels (list): Ordered list of the different classes.
        labels_color_RGBA (list): List of an ordered color map
            corresponding to the classes.
        labels_color_dictionary (dict): Dictionary with a mapping
            assigning a color tuple to each class.
    """
    # Choose color map for the labels
    colormap = mpl.cm.get_cmap("twilight_shifted")

    # Create an indexed dictionary for the list of labels
    labels = list(dataset.iloc[:, -1].unique())
    same_type = all(isinstance(x, type(labels[0])) for x in labels)
    if same_type:
        labels = sorted(labels)

    labels_color_RGBA = [
        colormap((x + 0.5) / len(labels)) for x in range(0, len(labels))
    ]

    labels_color_dictionary = dict(zip(labels, labels_color_RGBA))

    return labels, labels_color_RGBA, labels_color_dictionary


def scaling_plot(dataset, dataset_name):
    """Plot comparing the scaling of the different parameters.

    Figure depicting the range of values for all parameters of the
    dataset. The figure is both displayed and also saved to a folder
    named after the dataset.

    Parameters:
        dataset (DataFrame): Pandas dataset.
        dataset_name (str): Dataset filename without path nor extension.

    Returns:
        None
    """
    parameters_min_max = pd.concat(
        [dataset.describe().iloc[3], dataset.describe().iloc[7]], axis=1, sort=False,
    )

    # Invert the vertical ordering for the plot
    parameters_min_max = parameters_min_max[::-1]

    # Add a column with the calculated interval of each variable
    parameters_min_max.loc[:, "interval"] = (
        parameters_min_max["max"] - parameters_min_max["min"]
    )
    print(parameters_min_max[::-1])

    # Plots the variable intervals
    fig, ax = plt.subplots()
    ax.barh(
        parameters_min_max.index,
        parameters_min_max["max"] - parameters_min_max["min"],
        left=parameters_min_max["min"],
        height=0.4,
    )

    ax.grid(b=True, which="major", axis="x", markevery=slice(0, -1, 1))
    plt.tick_params(
        axis="y", bottom=True, left=False, direction="in", labelleft=True, pad=20,
    )
    ax.set_title("Comparison of parameter scaling", pad=10, fontsize=11)

    if parameters_min_max["max"].max() > 100:
        plt.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    plt.tight_layout()

    # Creates a folder and saves the plot to disk
    create_folder(dataset_name)
    plt.savefig("%s/%s_scaling_plot.svg" % (dataset_name, dataset_name))
    plt.show()


def histograms(dataset, dataset_name, stacked_classes, grid=3, bins=11):
    """Stacked or overlapping histograms of the dataset variables.

    Figure depicting the frequency of values for the variables on the
    database, with a different colour for the contribution of each
    class. The histograms can be stacked, with the contributions piled
    on top of each other, or overlapping, where the distributions may
    be in front of each other.

    There is a limit on the number of histograms to display at the same
    time for datasets with a lot of variables (grid**2). This function
    only displays the first ones of the dataset.

    The figure is both displayed and also saved to a folder named after
    the dataset.

    Parameters:
        dataset (DataFrame): Pandas dataset.
        dataset_name (str): Dataset filename without path nor extension.
        stacked_classes (bool): True yields stacked histograms, False
            yields overlapping histograms.
        grid (int): Number of rows and columns of subplots to display
            per page. The recommended values are 2 or 3.
        bins (int): Number of bins of the histograms. The recommended
            values are between 11 and 21.

    Return:
        None
    """
    print(
        f"\nYou chose a {grid}x{grid} grid, the histograms will be "
        f"calculated maximum {grid**2} at a time.\n"
    )

    # Obtain label list and create a color mapping for the dataset labels
    labels, labels_color_RGBA, labels_color_dictionary = labels_to_colors(dataset)

    # Determine name of column with the classes
    label_title = list(dataset)[-1]

    # Determine the number of parameters to plot and limit if necessary
    max_number_parameters = grid ** 2
    number_parameters = len(list(dataset))

    if number_parameters > max_number_parameters:
        number_parameters = max_number_parameters

    # Determine the necessary rows in the figure
    row_width = grid
    if number_parameters % row_width == 0:
        rows_of_histograms = int(number_parameters / row_width)
    else:
        rows_of_histograms = int(number_parameters / row_width) + 1

    # Generate a figure with each parameter histogram as a subplot
    figure, axs = plt.subplots(
        rows_of_histograms, row_width, figsize=(12, 9 * (rows_of_histograms / grid)),
    )

    # Iteration over parameters
    plot_data = []
    for subplot_index in range(row_width * rows_of_histograms):
        subplotX = int(subplot_index / row_width)
        subplotY = subplot_index % row_width

        if rows_of_histograms == 1:
            plot_data += [axs[subplotY]]
        else:
            plot_data += [axs[subplotX, subplotY]]

        if subplot_index > (number_parameters - 1):
            figure.delaxes(axs[subplotX, subplotY])

        else:
            # Create list with dataset separated in classes
            data_separated_labels = []
            for label_name in labels:
                data_separated_labels += [
                    dataset[(dataset[label_title] == label_name)].iloc[:, subplot_index]
                ]
                data_separated_labels[-1] = data_separated_labels[-1].replace(
                    {True: 1, False: 0}
                )

            # Plot stacked histograms single row
            if stacked_classes and rows_of_histograms == 1:
                axs[subplotY].hist(
                    data_separated_labels,
                    bins=bins,
                    histtype="barstacked",
                    label=list(dataset)[subplot_index],
                    edgecolor="black",
                    color=labels_color_RGBA,
                )

                if all(
                    isinstance(x, (int, float)) for x in dataset.iloc[:, subplot_index]
                ):
                    if dataset.iloc[:, subplot_index].max() > 100000:
                        axs[subplotY].ticklabel_format(
                            axis="x", style="sci", scilimits=(-2, 2)
                        )

            # Plot stacked histograms multiple rows
            elif stacked_classes and rows_of_histograms > 1:
                axs[subplotX, subplotY].hist(
                    data_separated_labels,
                    bins=bins,
                    histtype="barstacked",
                    label=list(dataset)[subplot_index],
                    edgecolor="black",
                    color=labels_color_RGBA,
                )
                if all(
                    isinstance(x, (int, float)) for x in dataset.iloc[:, subplot_index]
                ):
                    if dataset.iloc[:, subplot_index].max() > 100000:
                        axs[subplotX, subplotY].ticklabel_format(
                            axis="x", style="sci", scilimits=(-2, 2)
                        )

            # Plot overlapping histograms single row
            elif not stacked_classes and rows_of_histograms == 1:
                axs[subplotY].hist(
                    data_separated_labels,
                    bins=bins,
                    histtype="stepfilled",
                    label=list(dataset)[subplot_index],
                    alpha=0.6,
                    edgecolor="black",
                    color=labels_color_RGBA,
                )
                if all(
                    isinstance(x, (int, float)) for x in dataset.iloc[:, subplot_index]
                ):
                    if dataset.iloc[:, subplot_index].max() > 100000:
                        axs[subplotY].ticklabel_format(
                            axis="x", style="sci", scilimits=(-2, 2)
                        )

            # Plot overlapping histograms multiple rows
            else:
                axs[subplotX, subplotY].hist(
                    data_separated_labels,
                    bins=bins,
                    histtype="stepfilled",
                    label=list(dataset)[subplot_index],
                    alpha=0.6,
                    edgecolor="black",
                    color=labels_color_RGBA,
                )
                if all(
                    isinstance(x, (int, float)) for x in dataset.iloc[:, subplot_index]
                ):

                    if dataset.iloc[:, subplot_index].max() > 100000:
                        axs[subplotX, subplotY].ticklabel_format(
                            axis="x", style="sci", scilimits=(-2, 2)
                        )

            if rows_of_histograms == 1:
                plt.setp(axs[subplotY], xlabel=list(dataset)[subplot_index])
            else:
                plt.setp(axs[subplotX, subplotY], xlabel=list(dataset)[subplot_index])

        if rows_of_histograms == 1:
            plt.setp(axs[subplotY], ylabel="Frequency")
        else:
            plt.setp(axs[subplotX, subplotY], ylabel="Frequency")

        # Prepare the class markers (circles) for the legend

        # For stacked histograms
        label_markers = []
        for label_index in range(len(labels)):
            if stacked_classes:
                label_markers += [
                    mpl.lines.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=labels[label_index],
                        markerfacecolor=labels_color_RGBA[label_index],
                        markersize=10,
                    )
                ]

            # For overlapping histograms
            else:
                label_markers += [
                    mpl.lines.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        alpha=0.6,
                        label=labels[label_index],
                        markerfacecolor=labels_color_RGBA[label_index],
                        markersize=10,
                    )
                ]

        # Display the legend of the top right subplot.
        if rows_of_histograms == 1:
            axs[row_width - 1].legend(
                label_markers, labels, bbox_to_anchor=(1.8, 0.8), title="Classes",
            )

        else:
            axs[0, row_width - 1].legend(
                label_markers, labels, bbox_to_anchor=(1.8, 0.8), title="Classes",
            )

        create_folder(dataset_name)

        # Setting up a title and saving the figure
        if stacked_classes and rows_of_histograms == 1:
            axs[1].set_title("Parameter histograms (stacked)", pad=20, fontsize=11)
            plt.tight_layout()
            plt.savefig("%s/%s_histograms_stacked.svg" % (dataset_name, dataset_name))
        elif stacked_classes and rows_of_histograms > 1:
            axs[0, 1].set_title("Parameter histograms (stacked)", pad=20, fontsize=11)
            plt.tight_layout()
            plt.savefig("%s/%s_histograms_stacked.svg" % (dataset_name, dataset_name))
        elif not stacked_classes and rows_of_histograms == 1:
            axs[1].set_title("Parameter histograms (overlapped)", pad=20, fontsize=11)
            plt.tight_layout()
            plt.savefig(
                "%s/%s_histograms_overlapped.svg" % (dataset_name, dataset_name)
            )
        else:
            axs[0, 1].set_title(
                "Parameter histograms (overlapped)", pad=20, fontsize=11
            )
            plt.tight_layout()
            plt.savefig(
                "%s/%s_histograms_overlapped.svg" % (dataset_name, dataset_name)
            )
    plt.show()


def histograms_specific(
    dataset, dataset_name, stacked_classes, list_of_variables, grid=3, bins=11
):
    """Stacked or overlapping histograms of a specific list of variables.

    Similar to "histograms()", but in this case it displays only the
    histograms specified in a list.

    The figure is both displayed and also saved to a folder named after
    the dataset.

    Parameters:
        dataset (DataFrame): Pandas dataset.
        dataset_name (str): Dataset filename without path nor extension.
        stacked_classes (bool): True yields stacked histograms, False
            yields overlapping histograms.
        list_of_variables (list): Ordered list with the names of the
            selected variables.
        grid (int): Number of rows and columns of subplots to display
            per page. The recommended values are 2 or 3.
        bins (int): Number of bins of the histograms. The recommended
            values are between 11 and 21.

    Return:
        None
    """
    # Move the selected variables to the front of the list in the
    # desired order
    list_of_variables_inverted = list_of_variables[::-1]

    columns = list(dataset)
    for variable in list_of_variables_inverted:
        columns.insert(0, columns.pop(columns.index(variable)))

    # Remove from the list the variables that were not selected
    for index_variable in range(len(columns) - 1, len(list_of_variables), -1):
        columns.pop(index_variable - 1)

    # Build a subset of the dataset with the selected variables
    dataset_temporal = dataset.loc[:, columns]

    histograms(dataset_temporal, dataset_name, stacked_classes, grid=grid, bins=bins)


def histograms_all(dataset, dataset_name, stacked_classes, grid=3, bins=11):
    """Stacked or overlapping histograms of all variables.

    Similar to "histograms()", but in this case it displays the
    histograms of all variables in the sample, although grouping them
    in sets of grid**2 subplots.

    The figures are both displayed and also saved to a folder named
    ater the dataset.

    Parameters:
        dataset (DataFrame): Pandas dataset.
        dataset_name (str): Dataset filename without path nor extension.
        stacked_classes (bool): True yields stacked histograms, False
            yields overlapping histograms.
        grid (int): Number of rows and columns of subplots to display
            per page. The recommended values are 2 or 3.
        bins (int): Number of bins of the histograms. The recommended
            values are between 11 and 21.

    Return:
        None
    """
    list_of_variables = list(dataset)[:-1]

    # These are the number of pages completely filled with plots:
    full_pages = int(len(list_of_variables) / (grid ** 2))

    # If the last page is not full it will be handled differently
    if len(list_of_variables) % (grid ** 2) == 0:
        exact = True
    else:
        exact = False

    initial = list(range(grid ** 2))

    # Prepares a list for each full page to call the histogram builder
    for page_number in range(full_pages):
        list_temporal = [
            list_of_variables[(i + page_number * grid ** 2)] for i in initial
        ]
        histograms_specific(
            dataset, dataset_name, stacked_classes, list_temporal, grid=grid, bins=bins,
        )

    # Prepares a list specific for the last page if it's not full
    if not exact:
        list_temporal = []
        modulus = len(list_of_variables) % (grid ** 2)
        for n in range(modulus):
            list_temporal += [list_of_variables[full_pages * (grid ** 2) + n]]
        histograms_specific(
            dataset, dataset_name, stacked_classes, list_temporal, grid=grid, bins=bins,
        )


def projections(dataset, dataset_name, grid=3):
    """2D variable projections color-coded for the different classes.

    Figure depicting the projection of dataset variables, with a
    different colour for the contribution of each class. The figure is
    both displayed and also saved to a folder named after the dataset.

    There is a limit on the number of variables to display at the same
    time for datasets with a lot of variables (grid**2).

    Parameters:
        dataset (DataFrame): Pandas dataset.
        dataset_name (str): Dataset filename without path nor extension.
        grid (int): Maximum number of parameters to show in the same
            figure. The recommended value is 2-4.

    Return:
        None
    """
    # Obtain label list and create a color mapping for the dataset labels
    labels, labels_color_RGBA, labels_color_dictionary = labels_to_colors(dataset)

    # Determine name of column with the classes
    label_title = list(dataset)[-1]

    # Determine the number of parameters to plot and limit if necessary
    number_parameters = len(list(dataset))
    print(f"\nThere are {number_parameters-1} variables in this dataset.")

    max_number_parameters = grid
    if number_parameters > max_number_parameters:
        number_parameters = max_number_parameters
        print(
            "Only the crossed projections of %s parameters at a time "
            "will be plotted." % max_number_parameters
        )

    # Generate a figure with each parameter histogram as a subplot
    figure, axs = plt.subplots(
        number_parameters, number_parameters, figsize=(12, 10), tight_layout=True,
    )

    # Iteration over parameter combinations
    for columnY in range(number_parameters):
        for columnX in range(number_parameters):
            axs[columnY, columnX].scatter(
                dataset.iloc[:, columnX],
                dataset.iloc[:, columnY],
                c=dataset[label_title].map(labels_color_dictionary),
            )
            if all(isinstance(x, (int, float)) for x in dataset.iloc[:, columnX]):
                if dataset.iloc[:, columnX].max() > 100000:
                    axs[columnY, columnX].ticklabel_format(
                        axis="x", style="sci", scilimits=(-2, 2)
                    )
            plt.setp(axs[-1, columnX], xlabel=dataset.columns[columnX])

        if all(isinstance(x, (int, float)) for x in dataset.iloc[:, columnY]):
            if dataset.iloc[:, columnY].max() > 100000:
                axs[columnY, columnX].ticklabel_format(
                    axis="y", style="sci", scilimits=(-2, 2)
                )
        plt.setp(axs[columnY, 0], ylabel=dataset.columns[columnY])

        # Prepare markers of labels for legend
        label_markers = []
        for label_index in range(len(labels)):
            label_markers += [
                mpl.lines.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=labels[label_index],
                    markersize=10,
                    markerfacecolor=labels_color_RGBA[label_index],
                )
            ]

        # Display the legend of the top right subplot.
        axs[0, number_parameters - 1].legend(
            label_markers, labels, title="Classes", bbox_to_anchor=(1.0, 0.8)
        )

        axs[0, 1].set_title("Parameter space projections", pad=20, fontsize=11)
    create_folder(dataset_name)
    plt.savefig(f"{dataset_name}/{dataset_name}_parameter_projections.svg")
    plt.show()


def projections_specific(dataset, dataset_name, list_of_variables):
    """Selected 2D variable projections color-coded for different classes.

    Similar to "projections()", but in this case it displays only the
    histograms specified in a list.

    Parameters:
        dataset (DataFrame): Pandas dataset.
        dataset_name (str): Dataset filename without path nor extension.
        list_of_variables (list): Ordered list with the names of the
            selected variables. Maximum number of parameters to
            display in the same figure. It is recommended to calculate
            the projections for 2-4 variables.

    Return:
        None
    """
    # Move the selected variables to the front of the list in the
    # desired order
    list_of_variables_inverted = list_of_variables[::-1]
    columns = list(dataset)
    for variable in list_of_variables_inverted:
        columns.insert(0, columns.pop(columns.index(variable)))

    # Build the dataset with those variables
    dataset_temporal = dataset.loc[:, columns]

    projections(dataset_temporal, dataset_name, len(list_of_variables))


def projections_all(dataset, dataset_name, grid=2):
    """2D variable projections color-coded by class for all variables.

    Similar to "projections()", but in this case it displays some cross
    projections for all variables. It doesn't show ALL projections for
    ALL variables, but the desired combinations can be easily selected
    with "projections_specific()".

    There is a limit on the number of variables to display at the same
    time for datasets with a lot of variables (grid**2).

    Parameters:
        dataset (DataFrame): Pandas dataset.
        dataset_name (str): Dataset filename without path nor extension.
        grid (int): Maximum number of parameters to
            display in the same figure. The recommended value is 2-4.

    Return:
        None
    """
    list_of_variables = list(dataset)[:-1]
    # These are the number of pages completely filled with plots:
    full_pages = int(len(list_of_variables) / grid)

    # If the last page is not full it will be handled differently
    if len(list_of_variables) % grid == 0:
        exact = True
    else:
        exact = False

    # Prepares a list for each full page to call the projections builder
    initial = list(range(grid))
    for page_number in range(full_pages):
        list_temporal = [list_of_variables[(i + page_number * grid)] for i in initial]
        projections_specific(dataset, dataset_name, list_temporal)

    # Prepares a list specific for the last page if it's not full
    if not exact:
        list_temporal = []
        modulus = len(list_of_variables) % grid
        for n in range(modulus):
            list_temporal += [list_of_variables[full_pages * grid + n]]
        projections_specific(dataset, dataset_name, list_temporal)


def explore_test_split_ratio(dataset, dataset_name, num_test_fractions, random_tries):
    """Explore test-train split ratios with a linear SVC model.

    This function trains a linear SVC model with the given dataset,
    rescaling the variables and compensating during training with
    weights for any imbalance of the sample. It systematically trains
    the model dividing the data into the training and test subsets in
    different proportions.

    Parameters:
        dataset (DataFrame): Pandas dataset.
        num_test_fractions (int): Number of different test data
            fractions to explore (fractions between 0 and 1).
        random_tries (int): Number of randomised trainings to carry out
            for each test data fraction.

    Returns:
        test_sizes (numpy array): List of explored test data fractions.
        averages (numpy array): Average score of the randomised
            trainings for each test fraction.
        standard_deviations (numpy array): Standard deviation of the
            score of the randomised trainings for each test fraction.
    """
    # Load sample dataset
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    # Generate a list with the requested number of test data fractions
    test_sizes = np.linspace(
        1.0 / (num_test_fractions + 1), 1.0, num=num_test_fractions, endpoint=False,
    )

    counter = 0
    score = []
    for fraction in test_sizes:
        score_random = []
        for random_number in range(random_tries):

            # Roughly estimate duration of calculation
            if counter == 0:
                start = datetime.now()
            elif counter == 1:
                interval1 = datetime.now() - start
                total_time = interval1 * num_test_fractions * random_tries / 1.5
                iterations = num_test_fractions * random_tries - counter
                print(
                    f"{interval1} (hh:mm:ss.ss) for {counter} iteration/s. \n"
                    f"There are {iterations} iterations left. \n"
                    f"Estimated total running time: {total_time} (hh:mm:ss.ss)"
                )
            counter += 1

            # Split data into test and train subsets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=fraction, random_state=random_number
            )

            # Rescale all variables to (-1,1)
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            classifier = SVC(
                kernel="linear", decision_function_shape="ovo", class_weight="balanced",
            )
            classifier.fit(X_train_scaled, y_train)

            # Can this line be deleted?
            y_predicted = classifier.predict(X_test_scaled)  # noqa

            score_random += [classifier.score(X_test_scaled, y_test)]

        score += [score_random]

    # Gather average scores and standard deviations of all fractions
    score_summary = []
    for scores_list in score:
        score_summary += [(np.mean(scores_list), np.std(scores_list))]

    averages = np.array([score_summary[x][0] for x in range(len(score_summary))])
    standard_deviations = np.array(
        [score_summary[x][1] for x in range(len(score_summary))]
    )

    real_total_time = datetime.now() - start
    print(f"Real total running time: {real_total_time} (hh:mm:ss.ss)")

    # Plot and save figure
    figure, axs = plt.subplots(1, 2, figsize=(15, 6))

    mpl.rc("font", size=12)
    mpl.rc("axes", titlesize=12)

    axs[0].plot(test_sizes, averages, color="slateblue", linewidth=2.0)
    axs[0].fill_between(
        test_sizes,
        averages - standard_deviations,
        averages + standard_deviations,
        color="lavender",
    )
    axs[0].set_xlabel("Test subset fraction")
    axs[0].set_ylabel("Average score")

    axs[1].scatter(test_sizes, standard_deviations)
    axs[1].set_xlabel("Test subset fraction")
    axs[1].set_ylabel("Score standard deviation")

    create_folder(dataset_name)
    plt.savefig(
        f"{dataset_name}/{dataset_name}_train-test-exploration_"
        f"{num_test_fractions}_{random_tries}.svg"
    )
    plt.show()

    print(
        "\nIndex of point where score has the smallest standard "
        f"deviation: {standard_deviations.argmin()}"
    )
    print(
        "\nTest fraction where score has smallest standard "
        f"deviation: {test_sizes[standard_deviations.argmin()]:.4f}"
        "\nAverage score at test fraction with the smallest standard "
        f"deviation: {averages[standard_deviations.argmin()]:.4f}"
    )

    return test_sizes, averages, standard_deviations


def main():
    """Sample of the module features."""

    dataset, dataset_name = load_file(sys.argv[1])
    if dataset_name == "winequality":
        dataset = dataset.drop(columns=["quality"])

    print("\nThese are basic descriptors of the dataset: \n")
    table_descriptors(dataset)

    print(
        "\nIs the sample imbalanced? Should I use weights while "
        "training the model?\n"
    )
    print(weight_advisor(dataset))

    print(
        "\nDo I have to take care of reescaling the variables "
        "before training the model?\n"
    )
    scaling_plot(dataset, dataset_name)

    list_to_plot = [list(dataset)[1]] + [list(dataset)[2]] + [list(dataset)[0]]
    print(list_to_plot)

    histograms_specific(dataset, dataset_name, True, list_to_plot)
    histograms_all(dataset, dataset_name, True)
    histograms_all(dataset, dataset_name, False, bins=15)

    projections_specific(dataset, dataset_name, list_to_plot)
    projections_all(dataset, dataset_name, grid=4)

    # Should score_summary be tested to ensure the split is correct?
    score_summary = explore_test_split_ratio(dataset, dataset_name, 10, 5)  # noqa


if __name__ == "__main__":
    main()
