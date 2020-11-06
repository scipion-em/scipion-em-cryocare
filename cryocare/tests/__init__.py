

from pyworkflow.tests import DataSet
DataSet(name='cryocare', folder='cryocare',
        files={
            'reconsEvenOddTomoDir': 'Tomos_EvenOdd_Reconstructed',
            'tomo_even': 'Tomos_EvenOdd_Reconstructed/Tomo110_Even_bin6.mrc',
            'tomo_odd': 'Tomos_EvenOdd_Reconstructed/Tomo110_Odd_bin6.mrc'
        })
