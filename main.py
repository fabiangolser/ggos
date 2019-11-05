"""GGOS Lab.

This is the main function of our lab exercise.
Here data gets loaded and plotted.
"""
import sys
import numpy as np
import GGOS_pro1 as g_data
import ggosPlot as g_plot


def main(argv=sys.argv):
    # read test data
    test_data = g_data.data()
    test_data.read_given_txt("./lab01_data/", "potentialCoefficientsTides.txt")
    test_data.read_isdc_files("OAM", ['2005'])

    p = np.zeros([9000, 5])
    g = np.zeros([9000, 6])
    for i in range(9000):
        p[i, :] = test_data.getter("tc", 2005, i)
        g[i, :] = test_data.getter("isdc", 2005, i)

    # plot test data
    plot_range = 300

    # plot test data and save plot
    first_test_plot = g_plot.GgosPlot(p[:, 0], plot_range)
    first_test_plot.plot()
    first_test_plot.show(True)

    # plot test data animated
    second_test_plot = g_plot.GgosPlot(p, plot_range)
    second_test_plot.animate()
    second_test_plot.show()

    # plot test data animated and save as video
    third_test_plot = g_plot.GgosPlot(g, plot_range)
    third_test_plot.animate(0.5, True, 100)


if __name__ == "__main__":
    sys.exit(main())
