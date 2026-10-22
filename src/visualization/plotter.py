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
        layout_rect = getattr(self.figure, "layout_rect", None)
        if layout_rect is None:
            self.figure.tight_layout()
        else:
            self.figure.subplots_adjust(
                left=layout_rect[0],
                bottom=layout_rect[1],
                right=layout_rect[2],
                top=layout_rect[3],
            )
        plt.show()

    def save(self, path):
        self.figure.tight_layout()
        self.figure.savefig(path, dpi=300)
