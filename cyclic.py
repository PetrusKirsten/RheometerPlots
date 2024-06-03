import wx
from rheoplots.plotting import DynamicCompression
from matplotlib import pyplot as plt
import warnings
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

warnings.simplefilter('ignore')

if __name__ == "__main__":
    def argconfig():
        wx.app()
        # Parse command line arguments
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

        # Argumentos to DynamicCompression.__init__
        parser.add_argument(
            '-p', '--datapath', type=str,
            help='Datafile path.')
        parser.add_argument(
            '-np', '--n_points', default=196, type=int,
            help='Numbers of points per period.')
        parser.add_argument(
            '-f', '--fig_size', default=(34, 14), type=tuple,
            help='Size of the figure chart.')

        # Arguments to DynamicCompression.cyclic_plot
        parser.add_argument(
            '-ps', '--peak_size', default=3, type=int,
            help='Peak range for analysis.')
        parser.add_argument(
            '-is', '--initial_strain', default=10, type=float,
            help='Initial strain value for linear/elastic region.')
        parser.add_argument(
            '-fs', '--final_strain', default=18, type=float,
            help='Final strain value for linear/elastic region.')
        parser.add_argument(
            '-st', '--stress_period', default=True, type=bool,
            help='If True, plot the stress X strain.')
        parser.add_argument(
            '-pe', '--peak_period', default=True, type=bool,
            help='If True, plot the peak X period.')
        parser.add_argument(
            '-ym', '--ym_period', default=True, type=bool,
            help='If True, plot the ym X period.')
        parser.add_argument(
            '-ar', '--ratio', default=(3, 2), type=tuple,
            help='Charts width aspect ratio.')
        parser.add_argument(
            '-pt', '--plot_t', default=False, type=bool,
            help='If True, plot as stress X oscillation time.')
        parser.add_argument(
            '-pp', '--plot_peak', default=False, type=bool,
            help='If True, plot the peak region in stress x strain.')
        parser.add_argument(
            '-pf', '--plot_fit', default=False, type=bool,
            help='If True, plot the fitted curve in stress x strain.')
        parser.add_argument(
            '-cs', '--color_series', default='dodgerblue', type=str,
            help='Series markers color.')
        parser.add_argument(
            '-cl', '--color_linear', default='crimson', type=str,
            help='Linear/elastic region and linear fitting colors.')

        # Arguments to file config
        parser.add_argument(
            '-n', '--file_name', default='cyclic_plot.png', type=str,
            help='File name to save the plot. Must specify the format (e.g. .png or .pdf).')
        parser.add_argument(
            '-d', '--file_dpi', default=300, type=int,
            help='Figure resolution in dots per inch.')
        parser.add_argument(
            '-s', '--file_style', default='seaborn-v0_8-ticks', type=str,
            help='Select oredefined matplotlib styles. '
                 'The names of the available styles can be found in the list matplotlib.style.available.')

        args = vars(parser.parse_args())
        return tuple(args.values())


    args_values = argconfig()

    plt.style.use(args_values[17])

    data = DynamicCompression(
        args_values[0],
        args_values[1],
        args_values[2],
    )

    fig = DynamicCompression.cyclic_plot(
        data,
        args_values[3],
        args_values[4],
        args_values[5],
        args_values[6],
        args_values[7],
        args_values[8],
        args_values[9],
        args_values[10],
        args_values[11],
        args_values[12],
        args_values[13],
        args_values[14],
    )

    plt.show()

    fig.savefig(args_values[15], dpi=args_values[16])
    plt.close(fig)
