"""
GGOS Lab

This is the main function of our lab exercise.
Here data gets loaded, computed and plotted.
"""
import sys
import numpy as np
import GGOS_pro1 as g_data
import ggosPlot as g_plot
import ggos_erm as erm


def main(argv=sys.argv):
    """ here the different test methods are called """
    data = g_data.Data() # g_data.Data()>> constructor >> erstellt Objekt data = g_data.Data() >>> speichert Obejekt aut Variable data
    #data = test_data()
    #test_plot(data)
    test_calculation(data)


def test_data(length=10000):
    """ test data """
    # sun.txt    moon.txt     earthRotationVector.txt
    # potentialCoefficientsAOHIS.txt     potentialCoefficientsTides.txt
    test = g_data.Data()
    z = length
    a = np.zeros([z, 3])
    b = np.zeros([z, 3])
    c = np.zeros([z, 3])
    d = np.zeros([z, 5])
    e = np.zeros([z, 5])

    f = np.zeros([z, 6])
    g = np.zeros([z, 6])
    h = np.zeros([z, 6])
    j = np.zeros([z, 3])
    for i in range(z):
        a[i, :] = np.transpose(test.moon(i))
        b[i, :] = np.transpose(test.sun(i))
        c[i, :] = np.transpose(test.earth_rotation(i))
        d[i, :] = np.transpose(test.pc_aohis(i))
        e[i, :] = np.transpose(test.pc_tide(i))

        f[i, :] = np.transpose(test.aam(i))
        g[i, :] = np.transpose(test.aom(i))
        h[i, :] = np.transpose(test.ham(i))
        j[i, :] = np.transpose(test.slam(i))
    return test


def test_calculation(data):
    """ test calculations """
    len_max = 87649 - 8760
    test_erm = erm.RotationModel(data)

    test_erm.omega_dot(0)
    test_erm.polar_motion(0)

    w_dot = test_erm.omega_dot(1)
    polar = test_erm.polar_motion(1)
#    polar_ref = test_erm.poar_motion(0)
    for index in range(2, int(len_max/10)):
        w_dot = np.append(w_dot, test_erm.omega_dot(index), axis=0)
        polar = np.append(polar, test_erm.polar_motion(index), axis=0)
        print('omega_dot({}) = {}'.format(index-1, w_dot[index-1]))
        #polar = np.append(polar, [[1, index]], axis=0)
        #polar_ref = np.append(polar_ref, test_erm.polar_motion(index, True), axis=0)

    # plot polar
    plot_polar = g_plot.GgosPlot(polar, 3000)
    plot_polar.plot()
    #plot_polar.animate(0.2)
    plot_polar.show('polar_motion_10')

    # plot w_dot
    #w_dot = np.delete(w_dot, 2, 1)
    #plot_w_dot = g_plot.GgosPlot(w_dot, 3000)
    #plot_w_dot.plot()
    # plot_polar.animate(0.2)
    #plot_w_dot.show('plot_w_dot_10')

    # plot polar ref
    #plot_polar_ref = g_plot.GgosPlot(polar_ref, 3000)
    #plot_polar_ref.plot()
    #plot_polar_ref.animate(0.2)
    #plot_polar_ref.show('polar_motion_ref_10')


def test_plot(data):
    """ test plot data """
    z = 1000
    a = np.zeros([z, 3])
    b = np.zeros([z, 3])

    for i in range(z):
        a[i, :] = np.transpose(data.moon(i))
        b[i, :] = np.transpose(data.earth_rotation(i))

    plot_range = 300

    three_d_test_plot = g_plot.GgosPlot(a, plot_range)
    three_d_test_plot.plot()
    # three_d_test_plot.animate()
    three_d_test_plot.show()

    # plot test data and save plot
    first_test_plot = g_plot.GgosPlot(a[:, 0], plot_range)
    first_test_plot.plot()
    first_test_plot.show(True)

    # plot test data animated
    arr2D = np.delete(b, [2, 3], axis=1)  # use only first 2 columns
    second_test_plot = g_plot.GgosPlot(arr2D, plot_range)
    second_test_plot.animate()
    second_test_plot.show()

    # plot test data animated and save as video
    arr2Df = np.delete(b, [2, ], axis=1)  # use only first 2 columns
    third_test_plot = g_plot.GgosPlot(arr2Df, plot_range)
    third_test_plot.animate(0.5, True, 100)


if __name__ == "__main__":
    sys.exit(main())
