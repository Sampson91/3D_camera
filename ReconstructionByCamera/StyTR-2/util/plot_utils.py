"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pandas
import numpy as np
import seaborn as seaborn
import matplotlib.pyplot as plt

from pathlib import Path, PurePath


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'),
              exponential_weighted_smoothing_column=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    function_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(
                f"{function_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{function_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, directory in enumerate(logs):
        if not isinstance(directory, PurePath):
            raise ValueError(
                f"{function_name} - non-Path object in logs argument of {type(directory)}: \n{directory}")
        if not directory.exists():
            raise ValueError(
                f"{function_name} - invalid directory in logs argument:\n{directory}")
        # verify log_name exists
        function = Path(directory / log_name)
        if not function.exists():
            print(
                f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {function}")
            return

    # load log file(s) and plot
    log_files = [pandas.read_json(Path(parameter_) / log_name, lines=True) for
                 parameter_ in logs]

    figure, axials = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for log_file, color in zip(log_files,
                               seaborn.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_evaluation = pandas.DataFrame(
                    np.stack(log_file.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=exponential_weighted_smoothing_column).mean()
                axials[j].plot(coco_evaluation, c=color)
            else:
                log_file.interpolate().ewm(
                    com=exponential_weighted_smoothing_column).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axials[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for axial, field in zip(axials, fields):
        axial.legend([Path(parameter_).name for parameter_ in logs])
        axial.set_title(field)


def plot_precision_recall(files, naming_scheme='iteration'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [file_.parts[-3] for file_ in files]
    elif naming_scheme == 'iteration':
        names = [file_.stem for file_ in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    figure, axials = plt.subplots(ncols=2, figsize=(16, 5))
    for file_, color, name in zip(files, seaborn.color_palette("Blues",
                                                               n_colors=len(
                                                                       files)),
                                  names):
        data = torch.load(file_)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        precision2 = precision.mean()
        recall2 = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={precision2 * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * precision2 * recall2 / (precision2 + recall2 + 1e-8):0.3f}'
              )
        axials[0].plot(recall, precision, c=color)
        axials[1].plot(recall, scores, c=color)

    axials[0].set_title('Precision / Recall')
    axials[0].legend(names)
    axials[1].set_title('Scores / Recall')
    axials[1].legend(names)
    return figure, axials
