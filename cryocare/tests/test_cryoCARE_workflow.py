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

from os.path import exists, join
from cryocare.tests import CRYOCARE, DataSetCryoCARE
from pyworkflow.tests import setupTestProject, DataSet
from pyworkflow.utils import magentaStr
from tomo.protocols import ProtImportTomograms
from cryocare.protocols.protocol_load_model import Outputobjects as loadTrainingModelOutputs, ProtCryoCARELoadModel
from cryocare.protocols.protocol_training import Outputobjects as trainOutputs, ProtCryoCARETraining
from cryocare.protocols.protocol_predict import Outputobjects as predictOutputs, ProtCryoCAREPrediction
from cryocare.constants import TRAIN_DATA_FN, VALIDATION_DATA_FN, CRYOCARE_MODEL_TGZ
from cryocare.objects import CryocareModel
from tomo.tests.test_base_centralized_layer import TestBaseCentralizedLayer


class TestCryoCARE(TestBaseCentralizedLayer):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet(CRYOCARE)

    def _runImportTomograms(self, tomoFile, mode):
        print(magentaStr("\n==> Importing the %s tomograms:" % mode))
        sRate = DataSetCryoCARE.sRate.value
        protImport = self.newProtocol(ProtImportTomograms,
                                      filesPath=self.dataset.getFile(tomoFile),
                                      samplingRate=sRate)
        protImport.setObjLabel('Import %s tomograms' % mode)
        self.launchProtocol(protImport)
        output = getattr(protImport, 'Tomograms', None)
        self.checkTomograms(output,
                            expectedSetSize=DataSetCryoCARE.tomoSetSize.value,
                            expectedSRate=sRate,
                            expectedDimensions=DataSetCryoCARE.tomoDimensions.value)
        return output

    def _runTrainingData(self, evenTomos, oddTomos):
        print(magentaStr("\n==> Training"))
        patchSize = 40
        protTraining = self.newProtocol(ProtCryoCARETraining,
                                        evenTomos=evenTomos,
                                        oddTomos=oddTomos,
                                        patch_shape=patchSize,
                                        num_slices=400,
                                        n_normalization_samples=60,
                                        epochs=2,
                                        steps_per_epoch=10)
        self.launchProtocol(protTraining)
        cryoCareModel = getattr(protTraining, trainOutputs.model.name, None)
        # Check generated model
        self.assertEqual(type(cryoCareModel), CryocareModel)
        self.assertEqual(cryoCareModel.getPath(), protTraining._getExtraPath(CRYOCARE_MODEL_TGZ))
        # Check files and links generated
        self.assertTrue(exists(protTraining._getExtraPath('train_config.json')))
        self.assertTrue(exists(join(protTraining._getTrainDataDir(), TRAIN_DATA_FN)))
        self.assertTrue(exists(join(protTraining._getTrainDataDir(), VALIDATION_DATA_FN)))
        return protTraining

    def _runLoadTrainingModel(self):
        print(magentaStr("\n==> Loading a pre-trained model:"))
        protImportTM = self.newProtocol(
            ProtCryoCARELoadModel,
            trainDataModel=self.dataset.getFile(DataSetCryoCARE.training_data_model.name),
            trainDataDir=self.dataset.getFile(DataSetCryoCARE.training_data_dir.name))
        protImportTM = self.launchProtocol(protImportTM)
        cryoCareModel = getattr(protImportTM, loadTrainingModelOutputs.model.name, None)
        self.assertEqual(type(cryoCareModel), CryocareModel)
        self.assertEqual(cryoCareModel.getPath(), protImportTM._getExtraPath(CRYOCARE_MODEL_TGZ))
        self.assertEqual(cryoCareModel.getTrainDataDir(), protImportTM._getExtraPath())
        return protImportTM

    def _runPredict(self, evenTomos, oddTomos, model, displayText=None):
        if displayText:
            print(magentaStr(displayText))

        # Predict
        protPredict = self.newProtocol(ProtCryoCAREPrediction,
                                       evenTomos=evenTomos,
                                       oddTomos=oddTomos,
                                       model=model)

        self.launchProtocol(protPredict)
        output = getattr(protPredict, predictOutputs.tomograms.name, None)
        self.checkTomograms(output,
                            expectedSetSize=DataSetCryoCARE.tomoSetSize.value,
                            expectedSRate=DataSetCryoCARE.sRate.value,
                            expectedDimensions=DataSetCryoCARE.tomoDimensions.value)

    def testWorkflow(self):
        evenTomos = self._runImportTomograms(DataSetCryoCARE.tomo_even.name, 'even')
        oddTomos = self._runImportTomograms(DataSetCryoCARE.tomo_odd.name, 'odd')
        protTraining = self._runTrainingData(evenTomos, oddTomos)
        # Prediction from training
        displayText = "\n==> Predicting:"
        model = getattr(protTraining, trainOutputs.model.name, None)
        self._runPredict(evenTomos, oddTomos, model, displayText=displayText)
        # Load a pre-trained model and predict
        displayText = "\n==> Loading a pre-trained model and predicting:"
        protLoadPreTrainedModel = self._runLoadTrainingModel()
        model = getattr(protLoadPreTrainedModel, trainOutputs.model.name, None)
        self._runPredict(evenTomos, oddTomos, model, displayText=displayText)
