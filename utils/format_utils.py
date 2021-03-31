import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# Define function for string formatting of scientific notation
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        exponent = int(np.floor(np.log10(abs(num))))
    coeff = round(num / float(10 ** exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\times 10^{{{1:d}}}$".format(coeff, exponent, precision)


# def convert_to_exponent_of_ten(v):
#     if v == 0:
#         return v
#     else:
#         try:
#             return '$10^{' + str(int(np.log10(v))) + '}$'
#             # return '{:.1e}'.format(v)
#         except:
#             return v

def format_matplotlib(text_fontsize=10, default_fontsize=14, legend_fontsize=10):
    sns.set(style='darkgrid')
    # plt.style.use('tableau-colorblind10')
    font = {
        'family': 'normal',
        'weight': 'bold'
    }
    matplotlib.rc('font', **font)
    params = {'axes.labelsize': default_fontsize, 'axes.titlesize': default_fontsize, 'font.size': text_fontsize,
              'legend.fontsize': legend_fontsize,
              'xtick.labelsize': default_fontsize, 'ytick.labelsize': default_fontsize}
    matplotlib.rcParams.update(params)


format_matplotlib()
