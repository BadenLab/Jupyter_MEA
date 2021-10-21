# -*- coding: utf-8 -*-
"""
Backbone functions, which dont do the actual analysis but some other stuff like
finding the file ending, creating a file selection button etc.
@ Marvin Seifert 2021
"""
# Imports
from ipywidgets import widgets
import traitlets
from tkinter import Tk, filedialog
import numpy as np
from multiprocessing import Pool
import pandas as pd


def get_file_ending(any_file):
    """
    File ending function
    This function finds the ending of a file and returns the characters after the dot.

    Parameters
    ----------
    any_file : str:
        The file you want to get the file ending from
    Returns
    -------
    format: str: The str which contains only the part of the input string that is the
    after the dot.
    """
    dot_position = any_file.find(".")
    if dot_position == -1:
        print("Named file is not a file, (maybe folder?), return 0")
        return 0
    format = any_file[dot_position:]
    return format


class SelectFilesButton(widgets.Button):
    """
    File selection Button
    This class creates a file selection Button which on click opens a Windows explorer window
    for file selection

    """

    out = widgets.Output()

    def __init__(self, text):
        """
        Initialize the object

        Parameters
        ----------
        text: str: The text that should appear on the button

        """

        super(SelectFilesButton, self).__init__()
        # Add the selected_files trait
        self.add_traits(files=traitlets.traitlets.List())
        # Create the button.
        self.description = "Select " + text
        self.icon = "square-o"
        self.style.button_color = "orange"
        # Set on click behavior.
        self.on_click(self.select_files)

    @staticmethod
    def select_files(b):
        """Generate instance of tkinter.filedialog.

        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button
        """

        with b.out:
            try:
                # Create Tk root
                root = Tk()
                # Hide the main window
                root.withdraw()
                # Raise the root to the top of all windows.
                root.call("wm", "attributes", ".", "-topmost", True)
                # List of selected fileswill be set to b.value
                b.files = filedialog.askopenfilename(multiple=True)

                b.description = "File Selected"
                b.icon = "check-square-o"
                b.style.button_color = "lightgreen"
            except:
                pass


def bisection(array, value):
    """
    Bisection function
    Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
        and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
        to indicate that ``value`` is out of range below and above respectively.

    Parameters
    ----------
    array: np.array: The array containing a sequence of increasing values.

    value: int, float: The value that the array is compared against

    Returns
    -------
    position : The position of the nearest array value to value or zero if the
    value is out of range
    """
    n = len(array)
    if value < array[0]:
        return -1
    elif value > array[n - 1]:
        return -1
    jl = 0  # Initialize lower
    ju = n - 1  # and upper limits.
    while ju - jl > 1:  # If we are not yet done,
        jm = (ju + jl) >> 1  # compute a midpoint with a bitshift
        if value >= array[jm]:
            jl = jm  # and replace either the lower limit
        else:
            ju = jm  # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if value == array[0]:  # edge cases at bottom
        return 0
    elif value == array[n - 1]:  # and top
        return n - 1
    else:
        return jl + 1


def select_stimulus(max):
    """
    Creates a Jupyter widget that allows to select an integer between zeros and zero
    a maximum

    Parameters
    ----------
    max: integer: the maximum to which a number can be selected

    Returns
    -------
    A widget object
    """
    Select_stimulus_w = widgets.BoundedIntText(
        value=0,
        min=0,
        max=max,
        step=1,
        description="Which stimulus do you want to analyse?",
        disabled=False,
    )
    return Select_stimulus_w


def parallelize_dataframe(df, func, n_cores=4):
    """
    Function to split a dataframe into n parts and run the same function on all the
    parts of the dataframe on different cores in parallel.

    Parameters
    ----------
    df: pandas DataFrame: The dataframe
    func: function: the function that shall be maped onto the dataframe
    n_cores: int: default: 4: The number of cores which will be assigned with a
    part of the dataframe

    Returns
    -------
    df: pandas DataFrame: The recombined dataframe containing the results

    """

    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
