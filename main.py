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


def test_data(length=1000):
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

    g_plot.GgosPlot(a).plot()
    g_plot.GgosPlot(b).plot()
    g_plot.GgosPlot(c).plot()
    d = np.delete(d, [0, 1, 2], 1)
    g_plot.GgosPlot(d).plot()
    # g_plot.GgosPlot(e).plot()
    # g_plot.GgosPlot(f).plot()
    # g_plot.GgosPlot(g).plot()
    # g_plot.GgosPlot(h).plot()
    plt = g_plot.GgosPlot(j)
    plt.plot()
    plt.show()

    return test


def test_calculation(data):
    """ test calculations """
    len_max = 87649 - 8760
    plot_length = 1000
    test_erm = erm.RotationModel(data)

    ''' index = 0 '''
    test_erm.omega_dot(0)
    test_erm.polar_motion(0)
    test_erm.delta_lod(0)

    ''' index = 1 '''
    w_dot_v, f_invers_v, M_v, DT_G_Dt_w_v, w_x_Tw_v, w_x_h_v, dh_v, w_v, T_v, TG_v, TR_v = test_erm.omega_dot(1)
    polar = test_erm.polar_motion(1)
    d_lod = test_erm.delta_lod(1)
    #    polar_ref = test_erm.poar_motion(0)

    ''' index = 2 ... plot_length '''
    for index in range(2, int(plot_length)):
        w_dot, f_invers, M, DT_G_Dt_w, w_x_Tw, w_x_h, dh, w, T, TG, TR = test_erm.omega_dot(index)
        w_dot_v = np.append(w_dot_v, w_dot, axis=0)
        f_invers_v = np.append(f_invers_v, f_invers, axis=0)
        M_v = np.append(M_v, M, axis=0)
        DT_G_Dt_w_v = np.append(DT_G_Dt_w_v, DT_G_Dt_w, axis=0)
        w_x_Tw_v = np.append(w_x_Tw_v, w_x_Tw, axis=0)
        w_x_h_v = np.append(w_x_h_v, w_x_h, axis=0)
        dh_v = np.append(dh_v, dh, axis=0)
        w_v = np.append(w_v, w, axis=0)
        T_v = np.append(T_v, T, axis=0)
        TG_v = np.append(TG_v, TG, axis=0)
        TR_v = np.append(TR_v, TR, axis=0)

        polar = np.append(polar, test_erm.polar_motion(index), axis=0)
        # print('omega_dot({}) = {}'.format(index-1, w_dot[index-1]))
        # polar = np.append(polar, [[1, index]], axis=0)
        # polar_ref = np.append(polar_ref, test_erm.polar_motion(index, True), axis=0)
        d_lod = np.append(d_lod, test_erm.delta_lod(index))

    ''' intermediate plots '''
    g_plot.GgosPlot(w_dot_v, plot_length, 'w_dot_v').plot()
    ###g_plot.GgosPlot(f_invers_v, plot_length, 'f_invers_v').plot()
    #g_plot.GgosPlot(M_v, plot_length, 'M_v').plot()
    g_plot.GgosPlot(DT_G_Dt_w_v, plot_length, 'DT_G_Dt_w_v').plot()
    g_plot.GgosPlot(w_x_Tw_v, plot_length, 'w_x_Tw_v').plot()
    g_plot.GgosPlot(w_x_h_v, plot_length, 'w_x_h_v').plot()
    #g_plot.GgosPlot(dh_v, plot_length, 'dh_v').plot()
    g_plot.GgosPlot(w_v, plot_length, 'w_v').plot()
    ###g_plot.GgosPlot(T_v, plot_length, 'T_v').plot()
    ###g_plot.GgosPlot(TG_v, plot_length, 'TG_v').plot()
    ###g_plot.GgosPlot(TR_v, plot_length, 'TR_v').plot()

    ''' final plots '''
    # delta LOD
    g_plot.GgosPlot(d_lod, plot_length, 'delta_LOD').plot()

    # plot polar
    plot_polar = g_plot.GgosPlot(polar, plot_length, 'polar_motion_10')
    plot_polar.plot()
    # plot_polar.animate(0.2)
    plot_polar.show('polar_motion_10')


    ''' reference plots '''
    # plot w_dot
    # w_dot = np.delete(w_dot, 2, 1)
    # plot_w_dot = g_plot.GgosPlot(w_dot, plot_length)
    # plot_w_dot.plot()
    # plot_polar.animate(0.2)
    # plot_w_dot.show('plot_w_dot_10')

    # plot polar ref
    # plot_polar_ref = g_plot.GgosPlot(polar_ref, plot_length)
    # plot_polar_ref.plot()
    # plot_polar_ref.animate(0.2)
    # plot_polar_ref.show('polar_motion_ref_10')


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
