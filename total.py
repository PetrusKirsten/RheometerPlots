from rheoplots.plotting import DynamicCompression
from matplotlib import pyplot as plt
import warnings
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

warnings.simplefilter('ignore')

if __name__ == "__main__":
    def argconfig():
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
            '-fs', '--fig_size', default=(34, 14), type=tuple,
            help='Size of the figure chart.')

        # Argumentos to DynamicCompression.total_plot
        parser.add_argument(
            '-ns', '--normal_wave', default=False, type=bool,
            help='If True, plot fitted sine wave from height oscillation.')
        parser.add_argument(
            '-ds', '--damped_wave', default=False, type=bool,
            help='If True, plot fitted damped wave from stress oscillation')
        parser.add_argument(
            '-as', '--abs_wave', default=False, type=bool,
            help='If True, plot fitted absolute wave from stress oscillation.')
        parser.add_argument(
            '-ph', '--plot_h', default=False, type=bool,
            help='If True, plot the experimental height values.')
        parser.add_argument(
            '-pt', '--plot_t', default=False, type=bool,
            help='If True, plot the experimental height values.')
        parser.add_argument(
            '-c1', '--colorax1', default='dodgerblue', type=str,
            help='Left axis (stress) color.')
        parser.add_argument(
            '-c2', '--colorax2', default='silver', type=str,
            help='Right axis (height) color.')

        # Arguments to file config
        parser.add_argument(
            '-n', '--file_name', default='total_plot.png', type=str,
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

    plt.style.use(args_values[12])

    data = DynamicCompression(
        args_values[0],
        args_values[1],
        args_values[2],
    )

    fig = DynamicCompression.plotTotal(
        data,
        args_values[3],
        args_values[4],
        args_values[5],
        args_values[6],
        args_values[7],
        args_values[8],
        args_values[9]
    )
    plt.show()
    fig.savefig(args_values[10], dpi=args_values[11])
    plt.close(fig)
