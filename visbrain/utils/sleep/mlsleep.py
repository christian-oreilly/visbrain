"""Group of functions for machine-learning applied to sleep data."""
import numpy as np

__all__ = ['ml_ComputeFeatures', 'ml_TrainOnDatasets']


###############################################################################
# COMPUTE AND SAVE FEATURES DATASETS
###############################################################################
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


###############################################################################
# TRAINING CONSTRUCTION
###############################################################################

def ml_BuildFullTraining(datasets):
    """"""
    # ______________ Load and concat ______________
    x, y, chan = [], np.array([]), []
    for k in datasets:
        mat = np.load(k)
        x.append(mat['x'])
        chan.append(list(mat['channels']))
        y = np.concatenate((y, mat['y']), axis=0) if y.size else mat['y']

    # ______________ Features selection ______________
    # Power features :
    xpow = _ml_catPowerFeat(x, chan)

    # ______________ Build features and ratio ______________
    pass

    return xpow, y


def _ml_catPowerFeat(x, chan):
    """"""
    # Feature selection :
    powband = ['delta', 'theta', 'alpha', 'sigma', 'beta']
    usechan, usefeat = ['C3', 'C4'], slice(0, len(powband))
    # Find index of channels to use :
    idx = [[i.index(k) for k in usechan] for i in chan]
    # Select features :
    x = [x[k][usefeat, idx[k], :] for k in range(len(idx))]
    # Concatenate features :
    xcat = np.concatenate(x, axis=2)
    # Reshape :
    sh = xcat.shape
    return xcat.reshape(sh[0] * sh[1], sh[2]).T


###############################################################################
# MACHINE LEARNING
###############################################################################

def ml_DendoLike(x, y):
    """Build a dendogram like clasifiers."""
    # ______________ Imports ______________
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier

    # ______________ Define node ______________
    svm = ('svm', SVC())
    # lda = ('lda', LinearDiscriminantAnalysis())
    # rf = ('rf', RandomForestClassifier())
    clf = SVC() #VotingClassifier([svm, lda ], weights=[3, 0])
    node = {k: clf for k in range(5)}
    d = Dendogram(node)
    d.fit(x, y)
    idx = []
    for k in range(x.shape[0]):
        idx.append(d.predict(x[[k], :]))
    print(np.sum(np.array(idx) == y) / len(y))
    print(idx)


class Dendogram(object):
    """Build a Dendogram.

    Dendo description :
        - C1 -> Artefact Vs (Wake, REM, N1, N2, N3)
        - C2 -> Wake Vs (REM, N1, N2, N3)
        - C3 -> REM Vs (N1, N2, N3)
        - C4 -> N1 Vs (N2, N3)
        - C5 -> N2 Vs N3

    Args:
        node: dict
            Dictionary containing the classifier to use at each stage
            classification.
    """

    def __init__(self, node):
        self.node = node
        self.order = [5, 0, 4, 1, 2]

    def fit(self, xtrain, ytrain):
        """"""
        self.ytrain = ytrain
        for num, k in enumerate(self.order):
            # Build the exclusion label vector :
            yt = self._ybuild(k)
            # Train the classifier :
            print(np.unique(yt))
            self.node[num].fit(xtrain, yt)

    def _ybuild(self, index):
        ytrain = self.ytrain.copy()
        ytrain[self.ytrain != index] = -1
        return ytrain

    def predict(self, xtest, verbose=None):
        """"""
        # - C1 -> Artefact Vs (Wake, REM, N1, N2, N3)
        for k in range(len(self.order)):
            idx = self.node[k].predict(xtest)
            if idx == -1:
                continue
            else:
                break
        if k == len(self.order) - 1:
            return None
        else:
            return self.order[k]
        

def ml_TrainOnDatasets(x, y):
    """"""
    pass