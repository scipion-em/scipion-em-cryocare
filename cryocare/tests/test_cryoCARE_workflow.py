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

from os.path import exists
from cryocare.tests import CRYOCARE, DataSetCryoCARE
from pyworkflow.tests import BaseTest, setupTestProject, DataSet
from pyworkflow.utils import magentaStr
from tomo.protocols import ProtImportTomograms
from cryocare.protocols.protocol_load_model import outputObjects as loadTrainingModelOutputs, ProtCryoCARELoadModel
from cryocare.protocols.protocol_training import outputObjects as trainOutputs, ProtCryoCARETraining
from cryocare.protocols.protocol_predict import outputObjects as predictOutputs, ProtCryoCAREPrediction
from cryocare.constants import TRAIN_DATA_FN, TRAIN_DATA_CONFIG, TRAIN_DATA_DIR, VALIDATION_DATA_FN, CRYOCARE_MODEL_TGZ
from cryocare.objects import CryocareTrainData, CryocareModel


class TestCryoCARE(BaseTest):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet(CRYOCARE)
        cls.sRate = 4.71

    def _runImportTomograms(self, tomoFile, mode):
        print(magentaStr("\n==> Importing the %s tomograms:" % mode))
        protImport = self.newProtocol(ProtImportTomograms,
                                      filesPath=self.dataset.getFile(tomoFile),
                                      samplingRate=self.sRate)
        protImport.setObjLabel('Import %s tomograms' % mode)
        self.launchProtocol(protImport)
        output = getattr(protImport, 'Tomograms', None)
        self.assertSetSize(output, size=1)
        return protImport

    def _runTrainingData(self, protImportEven, protImportOdd):
        print(magentaStr("\n==> Training"))
        patchSize = 40
        protTraining = self.newProtocol(ProtCryoCARETraining,
                                        evenTomos=protImportEven.Tomograms,
                                        oddTomos=protImportOdd.Tomograms,
                                        patch_shape=patchSize,
                                        num_slices=400,
                                        n_normalization_samples=60,
                                        epochs=2,
                                        steps_per_epoch=10)
        import os
        self.launchProtocol(protTraining)
        cryoCareModel = getattr(protTraining, trainOutputs.model.name, None)
        # Check generated model
        self.assertEqual(type(cryoCareModel), CryocareModel)
        self.assertEqual(cryoCareModel.getPath(), protTraining._getExtraPath(CRYOCARE_MODEL_TGZ))
        #self.assertEqual(ProtCryoCARETraining._getExtraPath(TRAIN_DATA_DIR))
        # Check files and links generated
        self.assertTrue(exists(protTraining._getExtraPath('train_config.json')))
        self.assertTrue(exists(os.path.join(protTraining._getTrainDataDir(), TRAIN_DATA_FN)))
        self.assertTrue(exists(os.path.join(protTraining._getTrainDataDir(), VALIDATION_DATA_FN)))
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

    def _runPredict(self, protImportEven, protImportOdd, **kwargs):
        protLoadPreTrainedModel = kwargs.get("protLoadPreTrainedModel", None)
        protTraining = kwargs.get("protTraining", None)
        if protLoadPreTrainedModel:
            print(magentaStr("\n==> Loading a pre-trained model and predicting:"))
            trainedModelProt = protLoadPreTrainedModel
        else:
            print(magentaStr("\n==> Predicting:"))
            trainedModelProt = protTraining

        # Predict
        protPredict = self.newProtocol(ProtCryoCAREPrediction,
                                       even=protImportEven.Tomograms,
                                       odd=protImportOdd.Tomograms,
                                       model=getattr(trainedModelProt, trainOutputs.model.name, None))

        self.launchProtocol(protPredict)
        output = getattr(protPredict, predictOutputs.tomograms.name, None)
        self.assertEqual(output.getDim(), (618, 639, 104))
        self.assertEqual(output.getSize(), 1)
        self.assertEqual(output.getSamplingRate(), self.sRate)
        self.assertTrue(exists(protPredict._getExtraPath('Tomo110__bin6_denoised', 'Tomo110__bin6.mrc')))

    def testWorkflow(self):
        importTomoProtEven = self._runImportTomograms(DataSetCryoCARE.tomo_even.name, 'even')
        importTomoProtOdd = self._runImportTomograms(DataSetCryoCARE.tomo_odd.name, 'odd')
        protTraining = self._runTrainingData(importTomoProtEven, importTomoProtOdd)
        # Prediction from training
        self._runPredict(importTomoProtEven, importTomoProtOdd, protTraining=protTraining)
        # Load a pre-trained model and predict
        protLoadPreTrainedModel = self._runLoadTrainingModel()
        self._runPredict(importTomoProtEven, importTomoProtOdd, protTraining=protLoadPreTrainedModel)
