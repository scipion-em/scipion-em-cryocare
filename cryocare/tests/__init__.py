# **************************************************************************
# *
# * Authors:     Scipion Team
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

from enum import Enum

from cryocare.constants import CRYOCARE_MODEL_TGZ
from pyworkflow.tests import DataSet

CRYOCARE = 'cryocare'


class DataSetCryoCARE(Enum):
    rec_even_odd_tomos_dir = 'Tomos_EvenOdd_Reconstructed'
    tomo_even = 'Tomos_EvenOdd_Reconstructed/Tomo110_Even_bin6.mrc'
    tomo_odd = 'Tomos_EvenOdd_Reconstructed/Tomo110_Odd_bin6.mrc'
    model_dir = 'Training_Model'
    training_data_dir = 'Training_Data'
    train_data_file = 'Training_Data/train_data.npz'
    validation_data_file = 'Training_Data/val_data.npz'
    training_data_conf_dir = 'Training_Data_Config'
    training_data_conf = 'Training_Data_Config/training_data_config'
    training_data_model = CRYOCARE_MODEL_TGZ
    sRate = 4.71
    tomoDimensions = [618, 639, 104]
    tomoSetSize = 1


DataSet(name=CRYOCARE, folder=CRYOCARE, files={el.name: el.value for el in DataSetCryoCARE})
