import matplotlib.pyplot as plt


def visualize_spread_of_metric(df):
    """ takes in df table such that column 0 is value X, and rest of the columns have repeated runs of value y"""
    df["y_mean"] = df.iloc[:, 1:].mean(axis=1)
    df["y_min"] = df.iloc[:, 1:].min(axis=1)
    df["y_max"] = df.iloc[:, 1:].max(axis=1)
    fig, ax = plt.subplots()
    t = df["y_mean"]
    t1 = df["y_max"]
    t2 = df["y_min"]
    s = df.iloc[:, 0]

    ax.plot(s, t, "-", label="mean")
    ax.plot(s, t1, color="red", label="max")
    ax.plot(s, t2, color="green", label="min")
    ax.legend()
    ax.set_ylabel("values of y")
    ax.set_xlabel("values of X")
    ax.set_title("spread and mean of y over x ")
    ax.fill_between(s, t1, t2, alpha=0.2)
