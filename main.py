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

    z = 500
    a = np.zeros([z, 3])
    b = np.zeros([z, 3])
    c = np.zeros([z, 3])
    f = np.zeros([z, 6])

    for i in range(z):
        a[i, :] = np.transpose(test_data.moon(i))
        b[i, :] = np.transpose(test_data.sun(i))
        c[i, :] = np.transpose(test_data.earth_rottation(i))
        f[i, :] = np.transpose(test_data.aam(i))

    # plot test data
    plot_range = 300

    three_d_test_plot = g_plot.GgosPlot(a, plot_range)
    three_d_test_plot.plot()
    #three_d_test_plot.animate()
    three_d_test_plot.show()

    # plot test data and save plot
    first_test_plot = g_plot.GgosPlot(a[:, 0], plot_range)
    first_test_plot.plot()
    first_test_plot.show(True)

    # plot test data animated
    arr2D = np.delete(b, [2, 3], axis=1)    # use only first 2 columns
    second_test_plot = g_plot.GgosPlot(arr2D, plot_range)
    second_test_plot.animate()
    second_test_plot.show()

    # plot test data animated and save as video
    arr2Df = np.delete(f, [2, 3, 4, 5], axis=1)  # use only first 2 columns
    third_test_plot = g_plot.GgosPlot(arr2Df, plot_range)
    third_test_plot.animate(0.5, True, 100)


if __name__ == "__main__":
    sys.exit(main())
