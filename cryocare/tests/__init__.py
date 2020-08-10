
from pyworkflow.tests import DataSet

# Inputs are expected to be the even/odd reconstructed pair of tomograms. They must be
# generated and added to the Dataset.
# For now, a folder named cryocare must be created in SCIPION_HOME/data/tests and must
# contain both tomograms.

DataSet(name='cryocare', folder='cryocare',
        files={
            'tomo_even': 'Tomo110_even.rec',
            'tomo_odd': 'Tomo110_odd.rec',
            'pattern': '*.rec'
        })
