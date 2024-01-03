import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


class Plotter:
    def __init__(
        self,
        figsize_nrow=10,
        n_cols=2,
        figsize_ncols_multiplier=3,
        hspace=0.95,
        wspace=0.3,
        add_value=False,
        tick_rotation=45,
    ):
        self.figsize_nrow = figsize_nrow
        self.n_cols = n_cols
        self.figsize_ncols_multiplier = figsize_ncols_multiplier
        self.hspace = hspace
        self.wspace = wspace
        self.add_value = add_value
        self.tick_rotation = tick_rotation

    def generate_plots_from_series(self, *series, sort_value=False, y_label='Number of Documents', x_label='Intervals', title_suffix='Distribution'):
        num_rows = (len(series) + 1) // self.n_cols

        fig, axs = plt.subplots(
            num_rows,
            self.n_cols,
            figsize=(self.figsize_nrow, self.figsize_ncols_multiplier * num_rows),
            sharey=False,
        )
        if num_rows == 1:
            axs = axs.reshape(1, self.n_cols)

        for i, s in enumerate(series):
            s = s.sort_values(ascending=False) if sort_value else s.sort_index()
            
            row, col = divmod(i, self.n_cols)
            ax = axs[row, col]  # Select the subplot using 2D indexing
            x_labels = [str(key) for key in s.keys()]
            values = list(s.values)
            ax.bar(x_labels, values, width=0.5, color="blue")
            if self.add_value:
                for j, v in enumerate(values):
                    ax.text(j, v + 0.1, str(v), color="black", ha="center")
            ax.set_title(f"{s.index.name} {title_suffix}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.tick_params(axis="x", rotation=self.tick_rotation)

        for i in range(len(series), num_rows * self.n_cols):
            axs.flatten()[i].axis("off")

        plt.tight_layout()
        plt.subplots_adjust(hspace=self.hspace, wspace=self.wspace)

        return plt

    def get_word_cloud(data, width=800, height=800, background_color="white"):
        world_cloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            min_font_size=10,
        ).generate(" ".join(data))

        return world_cloud
