import numpy as np

def is_positive(x):
    return (not is_array_like(x) and _is_numeric(x) and x > 0)

def is_positive_or_zero(x):
    return (not is_array_like(x) and _is_numeric(x) and x >= 0)

def is_array_like(x):
    return isinstance(x, (tuple, list, np.ndarray))

def is_positive_integer(x):
    return (not is_array_like(x) and _is_integer(x) and x > 0)

def is_positive_integer_or_zero(x):
    return (not is_array_like(x) and _is_integer(x) and x >= 0)

def is_string(x):
    return isinstance(x, str)

def is_none(x):
    return isinstance(x, type(None))

def is_dict(x):
    return isinstance(x, dict)

def _is_numeric(x):
    return isinstance(x, (float, int))

def is_numeric_array(x):
    if is_array_like(x) and np.asarray(x).size > 0:
        try:
            np.asarray(x, dtype=float)
            return True
        except (ValueError, TypeError):
            return False
    return False

def _is_integer(x):
    return (_is_numeric(x) and (float(x) == int(x)))

# will intentionally accept 0, 1 as well
def is_bool(x):
    return (x in (True, False))

def is_non_zero_integer(x):
    return (_is_integer(x) and x != 0)

def _is_positive_array(x):
    if is_numeric_array(x) and (np.asarray(x, dtype = float) > 0).all():
        return True
    return False

def _is_positive_or_zero_array(x):
    if is_numeric_array(x) and (np.asarray(x, dtype = float) >= 0).all():
        return True
    return False

def _is_integer_array(x):
    if is_numeric_array(x):
        if (np.asarray(x, dtype = float) == np.asarray(x, dtype = int)).all():
            return True
    return False

def is_positive_integer_array(x):
    return (_is_integer_array(x) and _is_positive_array(x))


def is_positive_integer_or_zero_array(x):
    return (_is_integer_array(x) and _is_positive_or_zero_array(x))

# ------------- ** Checking inputs ** --------------------------

def check_global_descriptor(x):
    """
    This function checks that the data passed through x corresponds to the descriptor in a numpy array of shape
    (n_samples, n_features) containing floats.

    :param x: array like
    :return: numpy array of floats of shape (n_samples, n_features)
    """

    if not is_array_like(x):
        raise InputError("x should be array like.")

    x = np.asarray(x)

    if len(x.shape) != 2:
        raise InputError("x should be an array with 2 dimensions. Got %s" % (len(x.shape)))

    return x

def check_local_descriptor(x):
    """
    This function checks that the data passed through x corresponds to the descriptor in a numpy array of shape
    (n_samples, n_atoms, n_features) containing floats.

    :param x: array like
    :return: numpy array of floats of shape (n_samples, n_atoms, n_features)
    """

    if not is_array_like(x):
        raise InputError("x should be array like.")

    x = np.asarray(x)

    if len(x.shape) != 3:
        raise InputError("x should be an array with 3 dimensions. Got %s" % (len(x.shape)))

    return x

def check_y(y):
    """
    This function checks that y is a one dimensional array of floats.

    :param y: array like
    :return: numpy array of shape (n_samples, 1)
    """
    if not is_array_like(y):
        raise InputError("y should be array like.")

    y = np.atleast_2d(y).T

    return y

def check_sizes(x, y=None, dy=None, classes=None):
    """
    This function checks that the different arrays have the correct number of dimensions.

    :param x: array of 3 dimensions
    :param y: array of 1 dimension
    :param dy: array of 3 dimensions
    :param classes: array of 2 dimensions
    :return: None
    """

    if is_none(dy) and is_none(classes):

        if x.shape[0] != y.shape[0]:
            raise InputError("The descriptor and the properties should have the same first number of elements in the "
                             "first dimension. Got %s and %s" % (x.shape[0], y.shape[0]))

    elif is_none(y) and is_none(dy):
        if is_none(classes):
            raise InputError("Only x is not none.")
        else:
            if x.shape[0] != classes.shape[0]:
                raise InputError("Different number of samples in the descriptor and the classes: %s and %s." % (x.shape[0], classes.shape[0]))
            if len(x.shape) == 3:
                if x.shape[1] != classes.shape[1]:
                    raise InputError("The number of atoms in the descriptor and in the classes is different: %s and %s." % (x.shape[1], classes.shape[1]))

    elif is_none(dy) and not is_none(classes):

        if x.shape[0] != y.shape[0] or x.shape[0] != classes.shape[0]:
            raise InputError("All x, y and classes should have the first number of elements in the first dimension. Got "
                             "%s, %s and %s" % (x.shape[0], y.shape[0], classes.shape[0]))

        if len(x.shape) == 3:
            if x.shape[1] != classes.shape[1]:
                raise InputError("x and classes should have the same number of elements in the 2nd dimension. Got %s "
                                 "and %s" % (x.shape[1], classes.shape[1]))

    else:

        if x.shape[0] != y.shape[0] or x.shape[0] != dy.shape[0] or x.shape[0] != classes.shape[0]:
            raise InputError("All x, y, dy and classes should have the first number of elements in the first dimension. Got "
                             "%s, %s, %s and %s" % (x.shape[0], y.shape[0], dy.shape[0], classes.shape[0]))

        if x.shape[1] != dy.shape[1] or x.shape[1] != classes.shape[1]:
            raise InputError("x, dy and classes should have the same number of elements in the 2nd dimension. Got %s, %s "
                             "and %s" % (x.shape[1], dy.shape[1], classes.shape[1]))

def check_dy(dy):
    """
    This function checks that dy is a three dimensional array with the 3rd dimension equal to 3.

    :param dy: array like
    :return: numpy array of floats of shape (n_samples, n_atoms, 3)
    """

    if is_none(dy):
        approved_dy = dy
    else:
        if not is_array_like(dy):
            raise InputError("dy should be array like.")

        dy = np.asarray(dy)

        if len(dy.shape) != 3:
            raise InputError("dy should be an array with 3 dimensions. Got %s" % (len(dy.shape)))

        if dy.shape[-1] != 3:
            raise InputError("The last dimension of the array dy should be 3. Got %s" % (dy.shape[-1]))

        approved_dy = dy

    return approved_dy

def check_classes(classes):
    """
    This function checks that the classes is a numpy array of shape (n_samples, n_atoms) of ints
    :param classes: array like
    :return: numpy array of ints of shape (n_samples, n_atoms)
    """

    if is_none(classes):
        approved_classes = classes
    else:
        if not is_array_like(classes):
            raise InputError("classes should be array like.")

        if not is_positive_integer_array(classes):
            raise InputError("classes should be an array of ints.")

        classes = np.asarray(classes)

        if len(classes.shape) != 2:
            raise InputError("classes should be an array with 2 dimensions. Got %s" % (len(classes.shape)))
        approved_classes = classes

    return approved_classes








#
#def _is_numeric_array(x):
#    try:
#        arr = np.asarray(x, dtype = float)
#        return True
#    except (ValueError, TypeError):
#        return False
#
#def _is_numeric_scalar(x):
#    try:
#        float(x)
#        return True
#    except (ValueError, TypeError):
#        return False
#
#def is_positive(x):
#    if is_array(x) and _is_numeric_array(x):
#        return _is_positive_scalar(x)
#
#def _is_positive_scalar(x):
#    return float(x) > 0
#
#def _is_positive_array(x):
#    return np.asarray(x, dtype = float) > 0
#
#def is_positive_or_zero(x):
#    if is_numeric(x):
#        if is_array(x):
#            return is_positive_or_zero_array(x)
#        else:
#            return is_positive_or_zero_scalar(x)
#    else:
#        return False
#
#def is_positive_or_zero_array(x):
#
#
#def is_positive_or_zero_scalar(x):
#    return float(x) >= 0
#
#def is_integer(x):
#    if is_array(x)
#        return is_integer_array(x)
#    else:
#        return is_integer_scalar(x)
#
## will intentionally accept floats with integer values
#def is_integer_array(x):
#    if is_numeric(x):
#        return (np.asarray(x) == np.asarray(y)).all()
#    else:
#        return False
#
## will intentionally accept floats with integer values
#def is_integer_scalar(x):
#    if is_numeric(x):
#        return int(float(x)) == float(x)
#    else:
#        return False
#
#
#def is_string(x):
#    return isinstance(x, str)
#
#def is_positive_integer(x):
#    return (is_numeric(x) and is_integer(x) and is_positive(x))
#
#def is_positive_integer_or_zero(x):
#    return (is_numeric(x) and is_integer(x) and is_positive_or_zero(x))
#
#def is_negative_integer(x):
#    if is_integer(x):
#        return not is_positive(x)
#    else:
#        return False
#
#def is_non_zero_integer(x):
#    return (is_positive_integer(x) or is_negative_integer(x))


# Custom exception to raise when we intentinoally catch an error
# This way we can test that the right error was raised in test cases
class InputError(Exception):
    pass
    #def __init__(self, msg, loc):
    #    self.msg = msg
    #    self.loc = loc
    #def __str__(self):
    #    return repr(self.msg)

def ceil(a, b):
    """
    Returns a/b rounded up to nearest integer.

    """
    return -(-a//b)
