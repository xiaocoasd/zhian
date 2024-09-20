import matplotlib.pyplot as plt
from abc import ABC
import os
import numpy as np
from scipy.interpolate import griddata
from matplotlib import animation

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


class LogitsPlt(ABC):

    def __init__(
        self,
        data,
        act_data=None,
    ) -> None:
        super().__init__()
        self.data = data
        self.act_data = act_data

    def plt(
        self,
        is_2d,
        is_3d,
        path,
        title,
        x_label="x_value",
        y_label="y_value",
        z_label="z_value",
        test_episode=10,
    ):

        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if is_2d:

            if len(self.data[0]) == 1:
                if len(self.data) > 200:

                    x_values = np.arange(200)
                    y_values = []
                    for i in range(200):
                        x_values[i] += 1
                        y_values.append(self.data[i][0])

                else:
                    data_2d = self.data
                    x_values = np.arange(len(data_2d))
                    y_values = [point[0] for point in data_2d]

                    for i in range(len(data_2d)):
                        x_values[i] += 1
                plt.figure()
                plt.plot(
                    x_values,
                    y_values,
                    marker="o",
                    markersize=3,
                    linewidth=0.8,
                )
                plt.title(title)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.savefig(path, dpi=300)
                return

            # act_data = [item for sublist in self.act_data for item in sublist]
            # data_t = [item for sublist in self.data for item in sublist]
            # self.data = data_t
            act_data = self.act_data
            labels = None

            act_data_t = act_data
            if isinstance(act_data[0], list) is False:
                act_data_t = [[value] for value in act_data]

            dbscan = DBSCAN(eps=0.3, min_samples=5)

            labels = dbscan.fit_predict(act_data_t)

            if len(self.data[0]) > 2:
                data_std = StandardScaler().fit_transform(self.data)
                # print("###############")
                # print(self.data)
                # print(data_std)
                data_dimen = TSNE(
                    n_components=2,
                    perplexity=(len(data_std) - 1) if len(data_std) <= 30 else 30,
                )
                print(data_dimen)
                data_2d = data_dimen.fit_transform(data_std)
            else:
                data_2d = self.data

            # print(data_2d)

            x_values = [point[0] for point in data_2d]
            y_values = [point[1] for point in data_2d]

            plt.figure()
            plt.scatter(
                x_values,
                y_values,
                c=labels,
                cmap=plt.cm.Set1,
                edgecolors="k",
                s=20,
            )
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)

            plt.savefig(path, dpi=300)
            # plt.show()

        elif is_3d:
            if len(self.data[0]) > 3:
                data_3d = self.data_adapter(
                    self.data,
                    is_2d=False,
                    is_3d=True,
                )
            else:
                data_3d = self.data
            x_values = [point[0] for point in data_3d]
            y_values = [point[1] for point in data_3d]
            z_values = [point[2] for point in data_3d]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            xx = np.linspace(min(x_values), max(x_values), 100)
            yy = np.linspace(min(y_values), max(y_values), 100)
            xx, yy = np.meshgrid(xx, yy)
            zz = griddata((x_values, y_values), z_values, (xx, yy), method="cubic")

            ax.plot_surface(xx, yy, zz, cmap="viridis")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)

            def update(frame, ax):
                ax.view_init(elev=10.0, azim=frame)
                return fig

            ani = animation.FuncAnimation(
                fig,
                update,
                frames=np.arange(0, 360, 1),
                fargs=(ax,),
                interval=50,
            )

            ani.save(path, writer="pillow", fps=15)
            # plt.savefig(path, dpi=300)
            # plt.show()
        return 0

    def data_adapter(self, data, is_2d, is_3d):

        return 0
