import numpy as np
import os

import keras.callbacks as callbacks
from keras.callbacks import Callback


class SnapshotModelCheckpoint(Callback):
    """Callback that saves the snapshot weights of the model.

    Saves the model weights on certain epochs (which can be considered the
    snapshot of the model at that epoch).

    Should be used with the cosine annealing learning rate schedule to save
    the weight just before learning rate is sharply increased.

    # Arguments:
        nb_epochs: total number of epochs that the model will be trained for.
        nb_snapshots: number of times the weights of the model will be saved.
        fn_prefix: prefix for the filename of the weights.
    """

    def __init__(self, nb_epochs, nb_snapshots, fn_prefix='Model'):
        super(SnapshotModelCheckpoint, self).__init__()

        self.check = nb_epochs // nb_snapshots
        self.fn_prefix = fn_prefix

    def on_epoch_end(self, epoch, logs={}):
        if epoch != 0 and (epoch + 1) % self.check == 0:
            filepath = self.fn_prefix + "-%d.h5" % ((epoch + 1) // self.check)
            self.model.save_weights(filepath, overwrite=True)
            # print("Saved snapshot at weights/%s_%d.h5" % (self.fn_prefix, epoch))


def find_unit(T, M):

    """

    Args:
        T: number of epochs
        M: number of snapshots

    Returns: unit value = T/(2^0 + 2^1 + .... + 2^(M-1))

    """
    parts = 0
    for i in range(M):
        parts += 2 ** i
    unit = T // parts

    return unit


def find_step(t, unit):
    """
    Find index we are at: 2^step?

    For example: if T = 350, M = 3, then unit = T/(2^0 + 2^1 + 2^2) = 50
    and t = 55 => we are at 2^1 space : step = 1

    Args:
        t: epoch(th)
        unit: unit value = T/(2^0 + 2^1 + .... + 2^(M-1))

    Returns: step of epoch(th)

    """
    step = 0
    previous_sum = 0
    for i in range(128):  # we suppose that index < 128 because 2^128 is a big number bro!
        previous_sum += unit * (2 ** i)
        if previous_sum > t:
            step = i
            break

    return step


def find_current_epoch_value(t, step, unit):
    """
    Find current epoch value based on which step we are at

    For example:
    t = 200, unit = 50, step = 2 (we are at 2^2 space)
    Then: current epoch value = 200 - 50*(2^0 + 2^1) = 100

    Args:
        t: epoch(th)
        step: step we are at!
        unit: unit value = T/(2^0 + 2^1 + .... + 2^(M-1))

    Returns: current epoch value

    """
    previous_parts = 0
    for i in range(step):
        previous_parts += 2 ** i
    current = t - unit * previous_parts

    return current


class SnapshotCallbackBuilder:
    """Callback builder for snapshot ensemble training of a model.

    Creates a list of callbacks, which are provided when training a model
    so as to save the model weights at certain epochs, and then sharply
    increase the learning rate.
    """

    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        """
        Initialize a snapshot callback builder.

        # Arguments:
            nb_epochs: total number of epochs that the model will be trained for.
            nb_snapshots: number of times the weights of the model will be saved.
            init_lr: initial learning rate
        """
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr
        self.unit = find_unit(self.T, self.M)

    def get_callbacks(self, model_prefix='Model', is_super=True):
        """
        Creates a list of callbacks that can be used during training to create a
        snapshot ensemble of the model.

        Args:
            is_super: True if we want to use improved version of snapshot anneal schedule
            model_prefix: prefix for the filename of the weights.

        Returns: list of 3 callbacks [ModelCheckpoint, LearningRateScheduler,
                 SnapshotModelCheckpoint] which can be provided to the 'fit' function
        """
        if not os.path.exists('weights/'):
            os.makedirs('weights/')

        callback_list = [callbacks.ModelCheckpoint("weights/%s-Best.h5" % model_prefix, monitor="val_acc",
                                                   save_best_only=True, save_weights_only=True),
                         SnapshotModelCheckpoint(self.T, self.M, fn_prefix='weights/%s' % model_prefix)]

        if is_super:
            callback_list.append(callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule_super))
        else:
            callback_list.append(callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule))

        return callback_list

    def _cosine_anneal_schedule(self, t):
        """
        LearningRateScheduler with basic snapshot anneal schedule
        Args:
            t: epoch(th)

        Returns: learning rate value at epoch(th)

        """
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)

    def _cosine_anneal_schedule_super(self, t):
        """
        LearningRateScheduler with improved snapshot anneal schedule
        Args:
            t: epoch(th)

        Returns: learning rate value at epoch(th)

        TODO: clean it up a bit (>_<)
        """

        step = find_step(t, self.unit)
        space = self.unit * (2 ** step)
        current = find_current_epoch_value(t, step, self.unit)

        cos_inner = np.pi * current  # t - 1 is used when t has 1-based indexing
        cos_inner /= space
        cos_out = np.cos(cos_inner) + 1

        return float(self.alpha_zero / 2 * cos_out)
