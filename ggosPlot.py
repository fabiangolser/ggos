"""GGOS Lab.

This is the class used to plot the results.
Inspired by https://stackoverflow.com/questions/16132798/python-animation-graph
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class GgosPlot:

    def __init__(self, data, plot_range):
        self.__data = data
        self.__plot_range = plot_range
        self.__fig = plt.figure(figsize=(20, 12))

        if self.__data.ndim == 1:
            self.__length = len(self.__data)
            self.__ax = plt.axes(xlim=(0, self.__length), ylim=(np.min(self.__data), np.max(self.__data)))
        elif self.__data.ndim == 2:
            print('2-dimensional plot: ', np.shape(self.__data))
            self.__length = len(self.__data[:, 0])
            self.__ax = plt.axes(xlim=(np.min(self.__data[:, 0]), np.max(self.__data[:, 0])),
                                 ylim=(np.min(self.__data[:, 1]), np.max(self.__data[:, 1])))
            # self.__line_2, = self.__ax.plot([], [], lw=1, color="r")

        self.__line, = self.__ax.plot([], [], lw=2)

    def plot(self, block=False, show=True):
        # fig = plt.figure(figsize=(20, 12))
        # ax = plt.axes(xlim=(0, len(self.__data)), ylim=(np.min(self.__data), np.max(self.__data)))
        # self.__line, = ax.plot([], [], lw=2)
        # self.__line_2, = ax.plot([], [], lw=1, color="r")
        if self.__data.ndim == 1:
            plt.plot(range(0, self.__length), self.__data, 'b')
        elif self.__data.ndim == 2:
            plt.plot(self.__data[:, 0], self.__data[:, 1], 'b')
        # plt.plot(range(0, 9000), part_data_2/5, 'r')
        if show:
            plt.show(block=block)

    def __init_plot(self):
        # initialization function: plot the background of each frame
        self.__line.set_data([], [])
        return self.__line,

    def __animate(self, i):
        # animation function.  This is called sequentially
        plot_end = i
        plot_start = i - self.__plot_range
        if i < self.__plot_range:
            plot_start = 0
        if i > self.__length - 1:
            plot_end = self.__length - 1
        if self.__data.ndim == 1:
            x = range(plot_start, plot_end)
            y = self.__data[plot_start:plot_end]
        elif self.__data.ndim == 2:
            x = self.__data[:, 0][plot_start:plot_end]
            y = self.__data[:, 1][plot_start:plot_end]
        self.__line.set_data(x, y)
        # y_2 = part_data_2[plot_start:plot_end]
        # __line_2.set_data(x, y_2 / 5)
        return self.__line,  # __line_2,

    def animate(self, interval=1, save=False, fps=20):
        # call the animator.  blit=True means only re-draw the parts that have changed.
        animated_graph = animation.FuncAnimation(self.__fig, self.__animate, init_func=self.__init_plot, frames=self.__length + self.__plot_range, interval=interval, blit=True)
        if save:
            animation_writer = animation.writers['ffmpeg']
            writer = animation_writer(fps=fps, metadata=dict(artist='GGOS'), bitrate=1800)
            animated_graph.save('test.mp4', writer=writer)
        else:
            plt.show(block=True)

    def show(self, save=False):
        if save:
            plt.savefig('Figure_1.pdf')
        plt.show()
