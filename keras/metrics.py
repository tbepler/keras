import numpy as np
from . import backend as K
from .utils.generic_utils import get_from_module

def _weighted_masked_mean(array, weights, mask):
    if weights is None and mask is None:
        return K.mean(array)
    if weights is None:
        weights = 1
    else:
        while weights.ndim < array.ndim:
            weights = K.expand_dims(weights, dim=-1)
    if mask is None:
        mask = 1
    else:
        while mask.ndim < array.ndim:
            mask = K.expand_dims(mask, dim=-1)
    weights *= mask
    return (weights*array).sum()/weights.sum()

def binary_accuracy(y_true, y_pred, weights=None, mask=None):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems
    '''
    correct = K.equal(y_true, K.round(y_pred))
    return _weighted_masked_mean(correct, weights, mask)


def categorical_accuracy(y_true, y_pred, weights=None, mask=None):
    '''Calculates the mean accuracy rate across all predictions for
    multiclass classification problems
    '''
    correct = K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1))
    return _weighted_masked_mean(correct, weights, mask)


def sparse_categorical_accuracy(y_true, y_pred, weights=None, mask=None):
    '''Same as categorical_accuracy, but useful when the predictions are for
    sparse targets
    '''
    correct = K.equal(K.max(y_true, axis=-1), K.cast(K.argmax(y_pred, axis=-1), K.floatx()))
    return _weighted_masked_mean(correct, weights, mask)


def top_k_categorical_accuracy(y_true, y_pred, weights=None, mask=None, k=5):
    '''Calculates the top-k categorical accuracy rate, i.e. success when the
    target class is within the top-k predictions provided
    '''
    correct = K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k)
    return _weighted_masked_mean(correct, weights, mask)

def circular_mse(y_true, y_pred, weights=None, mask=None):
    d1 = K.square(y_true - y_pred)
    d2 = K.square(2*np.pi - K.maximum(y_true, y_pred) + K.minimum(y_true, y_pred))
    return _weighted_masked_mean(K.minimum(d1, d2), weights, mask)


def circular_mae(y_true, y_pred, weights=None, mask=None):
    d1 = K.abs(y_true - y_pred)
    d2 = K.abs(2*np.pi - K.maximum(y_true, y_pred) + K.minimum(y_true, y_pred))
    return _weighted_masked_mean(K.minimum(d1, d2), weights, mask)


def mean_squared_error(y_true, y_pred, weights=None, mask=None):
    '''Calculates the mean squared error (mse) rate
    between predicted and target values
    '''
    sq_err = K.square(y_pred - y_true)
    return _weighted_masked_mean(sq_error, weights, mask)


def mean_absolute_error(y_true, y_pred, weights=None, mask=None):
    '''Calculates the mean absolute error (mae) rate
    between predicted and target values
    '''
    abs_err = K.abs(y_pred - y_true)
    return _weighted_masked_mean(abs_err, weights, mask)


def mean_absolute_percentage_error(y_true, y_pred, weights=None, mask=None):
    '''Calculates the mean absolute percentage error (mape) rate
    between predicted and target values
    '''
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf))
    return 100. * _weighted_masked_mean(diff, weights, mask)


def mean_squared_logarithmic_error(y_true, y_pred, weights=None, mask=None):
    '''Calculates the mean squared logarithmic error (msle) rate
    between predicted and target values
    '''
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    sq_diff = K.square(first_log - second_log)
    return _weighted_masked_mean(sq_diff, weights, mask)


def hinge(y_true, y_pred, weights=None, mask=None):
    '''Calculates the hinge loss, which is defined as
    `max(1 - y_true * y_pred, 0)`
    '''
    hloss = K.maximum(1. - y_true * y_pred, 0.)
    return _weighted_masked_mean(hloss, weights, mask)


def squared_hinge(y_true, y_pred, weights=None, mask=None):
    '''Calculates the squared value of the hinge loss
    '''
    sq_hinge = K.square(K.maximum(1. - y_true * y_pred, 0.))
    return _weighted_masked_mean(sq_hinge, weights, mask)


def categorical_crossentropy(y_true, y_pred, weights=None, mask=None):
    '''Calculates the cross-entropy value for multiclass classification
    problems. Note: Expects a binary class matrix instead of a vector
    of scalar classes.
    '''
    cross_entropy = K.categorical_crossentropy(y_pred, y_true)
    return _weighted_masked_mean(cross_entropy, weights, mask)


def sparse_categorical_crossentropy(y_true, y_pred, weights=None, mask=None):
    '''Calculates the cross-entropy value for multiclass classification
    problems with sparse targets. Note: Expects an array of integer
    classes. Labels shape must have the same number of dimensions as
    output shape. If you get a shape error, add a length-1 dimension
    to labels.
    '''
    cross_entropy = K.sparse_categorical_crossentropy(y_pred, y_true) 
    return _weighted_masked_mean(cross_entropy, weights, mask)


def binary_crossentropy(y_true, y_pred, weights=None, mask=None):
    '''Calculates the cross-entropy value for binary classification
    problems.
    '''
    cross_entropy = K.binary_crossentropy(y_pred, y_true)
    return _weighted_masked_mean(cross_entropy, weights, mask)


def kullback_leibler_divergence(y_true, y_pred):
    '''Calculates the Kullback-Leibler (KL) divergence between prediction
    and target values
    '''
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


def poisson(y_true, y_pred):
    '''Calculates the poisson function over prediction and target values.
    '''
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()))


def cosine_proximity(y_true, y_pred):
    '''Calculates the cosine similarity between the prediction and target
    values.
    '''
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred)


def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def fbeta_score(y_true, y_pred, beta=1):
    '''Computes the F score, the weighted harmonic mean of precision and recall.

    This is useful for multi-label classification where input samples can be
    tagged with a set of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.

    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Weight precision and recall together as a single scalar.
    beta2 = beta ** 2
    f_score = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
    return f_score


# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity


def get(identifier):
    return get_from_module(identifier, globals(), 'metric')
