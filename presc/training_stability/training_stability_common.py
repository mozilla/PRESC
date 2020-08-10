import matplotlib as mpl
import matplotlib.pyplot as plt


def show_averages_and_variations(
    x,
    averages,
    standard_deviations,
    x_name="Test subset fraction",
    metric_name="accuracy",
):
    """Displays averages and their standard deviations.

    This function allows in general to visualize data corresponding to many repetitions, by showing the averages as a function of a variable and their corresponding standard deviations.

    Parameters:
        x (numpy array): Values of the x-axis variable.
        averages (numpy array): Values of the metric averages (y axis).
        standard_deviations (numpy array): Standard deviations from the
            metric average values (y axis).
        x_name (str): Variable name that will be used to label the x axis
            of the graph.
        metric_name (srt): Name of the metric or score that has been
            repeatedly computed, and that will be used in the labeling of
            the y axis.
    """
    figure, axs = plt.subplots(1, 2, figsize=(15, 6))

    mpl.rc("font", size=12)
    mpl.rc("axes", titlesize=12)

    axs[0].plot(x, averages, color="slateblue", linewidth=2.0)
    axs[0].fill_between(
        x,
        averages - standard_deviations,
        averages + standard_deviations,
        color="lavender",
    )
    axs[0].set_xlabel(x_name)
    axs[0].set_ylabel("Average " + metric_name)

    axs[1].scatter(x, standard_deviations)
    axs[1].set_xlabel(x_name)
    axs[1].set_ylabel(metric_name.capitalize() + " standard deviation")

    plt.show(block=False)

    print(
        "\nIndex of point where " + metric_name + " has the smallest standard "
        f"deviation: {standard_deviations.argmin()}"
    )
    print(
        "\n" + x_name + " where " + metric_name + " has smallest standard "
        f"deviation: {x[standard_deviations.argmin()]:.4f}"
        "\nAverage " + metric_name + " at test fraction with the smallest standard "
        f"deviation: {averages[standard_deviations.argmin()]:.4f}"
    )
