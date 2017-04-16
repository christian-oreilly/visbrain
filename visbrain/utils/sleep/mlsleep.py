"""Group of functions for machine-learning applied to sleep data."""
import numpy as np

__all__ = ['ml_ComputeFeatures', 'ml_TrainOnDatasets']


def ml_ComputeFeatures(data, sf, hypno):
    """Compute power features that will be used for Machine-Learning."""
    # ______________ Imports ______________
    from ..filtering import ndmorlet, filt

    # ______________ Preprocessing ______________
    # [0.2, 40.]hz bandpass :
    data = filt(sf, [.2, 40.], data, axis=1)
    # Replace -1 with 5:
    hypno = hypno.astype(int)
    hypno[hypno == -1] = 5

    # ______________ Spectral properties ______________
    f = np.array([[0.5, 4.5], [4.5, 8.5], [8.5, 11.5], [11.5, 15.5],
                  [15.5, 32.5]]).mean(1)

    # ______________ Compute power features ______________
    x = ndmorlet(data, sf, f, axis=1, get='power', win=30.)

    # ______________ Build label vector ______________
    # Find splitting index :
    wins = int(np.round(30. * sf))
    sp = np.arange(wins, len(hypno), wins, dtype=int)
    # Split hypno :
    hyps = np.split(hypno, sp)
    # Find stage proportions :
    y = []
    for k in hyps:
        # Find unique elements and proportions :
        prop = np.bincount(k)
        # Get and use larger amont :
        y.append(prop.argmax())

    # ______________ Shape checking ______________
    if x.shape[-1] != len(y):
        raise ValueError("Shape problem : ", x.shape, len(y))

    return x, np.array(y)


def ml_BuildTrainingSets(data, sf, hypno, channels):
    """Build training sets."""
    # Compute the features :
    x, y = ml_ComputeFeatures(data, sf, hypno)
    # Save it :
    ml_SaveDataset(x, y, channels)


def ml_SaveDataset(x, y, channels):
    """Save the dataset using the current time."""
    from datetime import datetime
    file = str(datetime.now()).replace(':', '_') + '.npz'
    np.savez(file, **{'x': x, 'y': y, 'channels': channels})


def ml_TrainOnDatasets():
    """"""
    pass