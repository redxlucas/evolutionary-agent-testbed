from matplotlib import pyplot as plt

class Plotter:
    """
    Base class for experiment plotters.
    """

    def __init__(
            self,
            figsize=(10, 6)
    ):
        self.figure, self.axes = plt.subplots(figsize=figsize)

    def plot_line(
            self,
            x,
            y,
            label=None
    ):
        self.axes.plot(x, y, label=label)

    def configure(
        self,
        title=None,
        xlabel=None,
        ylabel=None
    ):
        if title:
            self.axes.set_title(title)

        if xlabel:
            self.axes.set_xlabel(xlabel)

        if ylabel:
            self.axes.set_ylabel(ylabel)

        self.axes.grid(True)
        self.axes.legend()

    def show(self):
        self.figure.tight_layout()
        plt.show()

    def save(self, path):
        self.figure.tight_layout()
        self.figure.savefig(path, dpi=300)