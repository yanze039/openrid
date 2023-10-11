import os
import torch
import inspect
import logging
from openmm import unit
import openmmtools as mmtools
import mpiplus


def is_terminal_verbose():
    """Check whether the logging on the terminal is configured to be verbose.

    This is useful in case one wants to occasionally print something that is not really
    relevant to yank's log (e.g. external library verbose, citations, etc.).

    Returns
    -------
    is_verbose : bool
        True if the terminal is configured to be verbose, False otherwise.
    """

    # If logging.root has no handlers this will ensure that False is returned
    is_verbose = False

    for handler in logging.root.handlers:
        # logging.FileHandler is a subclass of logging.StreamHandler so
        # isinstance and issubclass do not work in this case
        if type(handler) is logging.StreamHandler and handler.level <= logging.DEBUG:
            is_verbose = True
            break

    return is_verbose


def config_root_logger(verbose, log_file_path=None):
    """
    Setup the the root logger's configuration.

    The log messages are printed in the terminal and saved in the file specified
    by log_file_path (if not None) and printed. Note that logging use sys.stdout
    to print logging.INFO messages, and stderr for the others. The root logger's
    configuration is inherited by the loggers created by logging.getLogger(name).

    Different formats are used to display messages on the terminal and on the log
    file. For example, in the log file every entry has a timestamp which does not
    appear in the terminal. Moreover, the log file always shows the module that
    generate the message, while in the terminal this happens only for messages
    of level WARNING and higher.

    Parameters
    ----------
    verbose : bool
        Control the verbosity of the messages printed in the terminal. The logger
        displays messages of level logging.INFO and higher when verbose=False.
        Otherwise those of level logging.DEBUG and higher are printed.
    log_file_path : str, optional, default = None
        If not None, this is the path where all the logger's messages of level
        logging.DEBUG or higher are saved.

    """

    class TerminalFormatter(logging.Formatter):
        """
        Simplified format for INFO and DEBUG level log messages.

        This allows to keep the logging.info() and debug() format separated from
        the other levels where more information may be needed. For example, for
        warning and error messages it is convenient to know also the module that
        generates them.
        """

        # This is the cleanest way I found to make the code compatible with both
        # Python 2 and Python 3
        simple_fmt = logging.Formatter('%(asctime)-15s: %(message)s')
        default_fmt = logging.Formatter('%(asctime)-15s: %(levelname)s - %(name)s - %(message)s')

        def format(self, record):
            if record.levelno <= logging.INFO:
                return self.simple_fmt.format(record)
            else:
                return self.default_fmt.format(record)

    # Check if root logger is already configured
    n_handlers = len(logging.root.handlers)
    if n_handlers > 0:
        root_logger = logging.root
        for i in range(n_handlers):
            root_logger.removeHandler(root_logger.handlers[0])

    # If this is a worker node, don't save any log file
    import mpiplus
    mpicomm = mpiplus.get_mpicomm()
    if mpicomm:
        rank = mpicomm.rank
    else:
        rank = 0

    # Create different log files for each MPI process
    if rank != 0 and log_file_path is not None:
        basepath, ext = os.path.splitext(log_file_path)
        log_file_path = '{}_{}{}'.format(basepath, rank, ext)

    # Add handler for stdout and stderr messages
    terminal_handler = logging.StreamHandler()
    terminal_handler.setFormatter(TerminalFormatter())
    if rank != 0:
        terminal_handler.setLevel(logging.WARNING)
    elif verbose:
        terminal_handler.setLevel(logging.DEBUG)
    else:
        terminal_handler.setLevel(logging.INFO)
    logging.root.addHandler(terminal_handler)

    # Add file handler to root logger
    file_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    if log_file_path is not None:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(file_format))
        logging.root.addHandler(file_handler)

    # Do not handle logging.DEBUG at all if unnecessary
    if log_file_path is not None:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(terminal_handler.level)

    # Setup critical logger file if a logfile is specified
    # No need to worry about MPI due to it already being set above
    if log_file_path is not None:
        basepath, ext = os.path.splitext(log_file_path)
        critical_log_path = basepath + "_CRITICAL" + ext
        # Create the critical file handler to only create the file IF a critical message is sent
        critical_file_handler = logging.FileHandler(critical_log_path, delay=True)
        critical_file_handler.setLevel(logging.CRITICAL)
        # Add blank lines to space out critical errors
        critical_file_format = file_format + "\n\n\n"
        critical_file_handler.setFormatter(logging.Formatter(critical_file_format))
        logging.root.addHandler(critical_file_handler)



def quantity_from_string(expression, compatible_units=None):
    """Create a Quantity object from a string expression.

    All the functions in the standard module math are available together
    with most of the methods inside the ``simtk.unit`` module.

    Parameters
    ----------
    expression : str
        The mathematical expression to rebuild a Quantity as a string.
    compatible_units : simtk.unit.Unit, optional
       If given, the result is checked for compatibility against the
       specified units, and an exception raised if not compatible.

       `Note`: The output is not converted to ``compatible_units``, they
       are only used as a unit to validate the input.

    Returns
    -------
    quantity
        The result of the evaluated expression.

    Raises
    ------
    TypeError
        If ``compatible_units`` is given and the quantity in expression is
        either unit-less or has incompatible units.

    Examples
    --------
    >>> expr = '4 * kilojoules / mole'
    >>> quantity_from_string(expr)
    Quantity(value=4.000000000000002, unit=kilojoule/mole)

    >>> expr = '1.0*second'
    >>> quantity_from_string(expr, compatible_units=unit.femtosecond)
    Quantity(value=1.0, unit=second)

    """
    # Retrieve units from unit module.
    if not hasattr(quantity_from_string, '_units'):
        units_tuples = inspect.getmembers(unit, lambda x: isinstance(x, unit.Unit))
        quantity_from_string._units = dict(units_tuples)

    # Eliminate nested quotes and excess whitespace
    try:
        expression = expression.strip('\'" ')
    except AttributeError:
        raise TypeError('The expression {} must be a string defining units, '
                        'not a {} instance'.format(expression, type(expression)))

    # Handle a special case of the unit when it is just "inverse unit",
    # e.g. Hz == /second
    if expression[0] == '/':
        expression = '(' + expression[1:] + ')**(-1)'

    # Evaluate expressions.
    quantity = mmtools.utils.math_eval(expression, variables=quantity_from_string._units)

    # Check to make sure units are compatible with expected units.
    if compatible_units is not None:
        try:
            is_compatible = quantity.unit.is_compatible(compatible_units)
        except AttributeError:
            raise TypeError("String {} does not have units attached.".format(expression))
        if not is_compatible:
            raise TypeError("Units of {} must be compatible with {}"
                            "".format(expression, str(compatible_units)))

    return quantity


import MDAnalysis as mda
import numpy as np

def prep_dihedral(conf):
    u = mda.Universe(conf)
    dihedral_selection = []
    for res in u.residues:
        if res.resname == "SOL" or res.resname == "NA" or res.resname == "CL":
            continue
        if res.phi_selection() is not None:
            dihedral_selection.append([ii.index for ii in res.phi_selection()])
        if res.psi_selection() is not None:
            dihedral_selection.append([ii.index for ii in res.psi_selection()])
        
    return np.array(dihedral_selection)


def calc_dihedral(conf):
    u = mda.Universe(conf)
    dihedral_selection = []
    for res in u.residues:
        if res.resname == "SOL" or res.resname == "NA" or res.resname == "CL":
            continue
        if res.phi_selection() is not None:
            dihedral_selection.append([[ii.index for ii in res.phi_selection()], res.phi_selection().dihedral.value()])
        if res.psi_selection() is not None:
            dihedral_selection.append([[ii.index for ii in res.psi_selection()], res.psi_selection().dihedral.value()])
        
    return dihedral_selection


def jit_script_to(model, output_path):
    torch.jit.script(model).save(output_path)
    return output_path


def set_barrier():
    try:
        mpicomm = mpiplus.get_mpicomm()
        mpicomm.barrier()
    except AttributeError:
        pass