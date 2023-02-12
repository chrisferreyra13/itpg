"""Compute connectivity based on O-Info."""
# Authors: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 01/2023
import logging
import os

import xarray as xr

from itpg.hoi import conn_hoi, plot_oinfo
from cfutils.workflow import setup_output_folder, get_input_filepath
from cfutils.pyutils import setup_logging
from itpg.data import generate_data, set_lin_relationship


def read_data(data_filepath):
    ...


def generate_test_data():
    """Generate random data with linear relationships."""
    n_trials = 300
    n_roi = 6
    n_times = 600
    # win amplitude
    alpha = 0.6

    # the last item is the dependent var
    # ex: (0,1,2) => X_2 = X_0 + X_1
    relationships = [
        ("0,1,2", slice(200, 300)),
    ]

    # generate and setup data
    logging.info(f'Generating test data')
    x, trials, times, roi = generate_data(n_trials, n_roi, n_times)
    x = set_lin_relationship(x, relationships, alpha)

    return x, trials, times, roi


def pipeline_conn_oinfo(dataset_name, filename, output_path):
    """Compute O-Info connectivity for brain data."""
    # setup input data and output folder
    # init output folder

    # get input path
    logging.info('Getting input path')
    data_filepath = get_input_filepath(dataset_name, filename)

    # read data
    # logging.info(f'Reading data file {data_filepath}')
    # data = read_data(data_filepath)

    data, trials, times, roi = generate_test_data()

    # format data
    data = xr.DataArray(data, dims=('trials', 'roi', 'times'),
                        coords=(trials, roi, times))

    # compute o-info
    logging.info('Computing O-info connnectivity')
    oinfo = conn_hoi(data, minsize=3, maxsize=5, y=None, roi='roi',
                     times='times')

    # plot result
    logging.info('Plotting results')
    plot_oinfo(oinfo)  # , output_path)


if __name__ == '__main__':
    # set variables
    analysis_name = 'conn_oinfo'
    dataset_name = 'dataset-test'
    filename = 'test.txt'

    output_path = setup_output_folder(analysis_name)

    # init logging
    log_filepath = os.path.join(
        output_path, analysis_name+'.log')
    setup_logging(log_filepath)

    try:
        pipeline_conn_oinfo(dataset_name, filename, output_path)
    except Exception as ex:
        logging.error(ex)
        logging.error('Error when running pipeline')

    logging.info('Succesful')
