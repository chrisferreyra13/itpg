"""Setup functions for analysis."""
# Authors: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 2022

import os


def setup_output_folder(analysis_name):
    """Create output folder with name versioning."""
    output_path = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_folder_name = get_name_version(analysis_name, output_path)
    output_path = os.path.join(output_path, output_folder_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    return output_path


def get_name_version(name, path):
    """Construct file/folder name checking previous versions."""
    items = os.listdir(path)
    # check if file exits
    items = [os.path.splitext(i)[0] for i in items if name in i]
    if len(items) == 0:
        suffix = ''
    else:
        # check if there are different versions
        last_version = 0
        for i in items:
            i = i.split('_')
            if i[-1].isnumeric():
                if int(i[-1]) > last_version:
                    last_version = int(i[-1])

        suffix = '_'+str(last_version+1)

    name = name+suffix

    return name


def get_figs_path(output_path, fig_name):
    """Construct figures path to ./output/output_name/figures."""
    figs_path = os.path.join(output_path, 'figures')
    if not os.path.exists(figs_path):
        os.mkdir(figs_path)

    fig_name_path = os.path.join(figs_path, fig_name)

    return fig_name_path
