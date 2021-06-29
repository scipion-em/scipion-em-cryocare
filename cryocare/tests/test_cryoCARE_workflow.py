from os.path import exists
from pyworkflow.tests import BaseTest, setupTestProject
import tomo.protocols
from pyworkflow.utils import magentaStr

from . import DataSet
from ..constants import TRAIN_DATA_FN, TRAIN_DATA_CONFIG, TRAIN_DATA_DIR, VALIDATION_DATA_FN
from ..objects import CryocareTrainData
from ..protocols import ProtCryoCAREPrediction, ProtCryoCAREPrepareTrainingData, ProtCryoCARELoadModel


class TestCryoCARE(BaseTest):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('cryocare2')
        cls.sRate = 2.355

    def _runImportTomograms(self, tomoFile):
        print(magentaStr("\n==> Importing the tomograms even and odd:"))
        protImport = self.newProtocol(
            tomo.protocols.ProtImportTomograms,
            filesPath=self.dataset.getFile(tomoFile),
            samplingRate=self.sRate)
        self.launchProtocol(protImport)
        output = getattr(protImport, 'outputTomograms', None)
        self.assertSetSize(output, size=1)
        return protImport

    def _runPrepareTrainingData(self, protImportEven, protImportOdd):
        print(magentaStr("\n==> Preparing the training data:"))
        protPrepTrainingData = self.newProtocol(ProtCryoCAREPrepareTrainingData,
                                                evenTomos=protImportEven.outputTomograms,
                                                oddTomos=protImportOdd.outputTomograms)
        self.launchProtocol(protPrepTrainingData)
        cryoCareTrainData = getattr(protPrepTrainingData, 'train_data', None)

        # Check generated object
        self.assertEqual(type(cryoCareTrainData), CryocareTrainData)
        self.assertEqual(cryoCareTrainData.getTrainDataDir(), protPrepTrainingData._getExtraPath(TRAIN_DATA_DIR))
        self.assertEqual(cryoCareTrainData.getPatchSize(), 72)
        # Check files generated
        self.assertTrue(exists(protPrepTrainingData._getExtraPath(TRAIN_DATA_DIR, TRAIN_DATA_FN)))
        self.assertTrue(exists(protPrepTrainingData._getExtraPath(TRAIN_DATA_DIR, VALIDATION_DATA_FN)))
        self.assertTrue(exists(protPrepTrainingData._getExtraPath(TRAIN_DATA_CONFIG, TRAIN_DATA_CONFIG)))

        return protPrepTrainingData

    def _runTrainingData(self, protPrepTrainingData):
        # Skipped because of it long execution time. Generated model was stored as part of the test
        # dataset and imported in the prediction test
        print(magentaStr("\n==> Skipping training:"))
        return []
    #     protTraining = self.newProtocol(ProtCryoCARETraining,
    #                                     train_data=getattr(protPrepTrainingData, 'train_data', None),
    #                                     batch_size=4)
    #
    #     self.launchProtocol(protTraining)
    #     cryoCareModel = getattr(protTraining, 'model', None)
    #     # Check generated model
    #     self.assertEqual(type(cryoCareModel), CryocareModel)
    #     self.assertEqual(cryoCareModel.getPath(), protTraining._getExtraPath())
    #     self.assertEqual(cryoCareModel.getTrainDataDir(), protPrepTrainingData._getExtraPath(TRAIN_DATA_DIR))
    #     # Check files and links generated
    #     self.assertTrue(exists(protTraining._getExtraPath('train_config.json')))
    #     self.assertTrue(exists(protTraining._getExtraPath(TRAIN_DATA_FN)))
    #     self.assertTrue(exists(protTraining._getExtraPath(VALIDATION_DATA_FN)))
    #     self.assertTrue(exists(protTraining._getExtraPath(CRYOCARE_MODEL, 'config.json')))
    #     self.assertTrue(exists(protTraining._getExtraPath(CRYOCARE_MODEL, 'weights_best.h5')))
    #     self.assertTrue(exists(protTraining._getExtraPath(CRYOCARE_MODEL, 'weights_last.h5')))
    #     return protTraining

    def _runPredict(self, protImportEven, protImportOdd, trainingProt):
        print(magentaStr("\n==> Loading a pre-trained model and predicting:"))
        # Load a trained model
        protImportTrainedModel = self.newProtocol(
            ProtCryoCARELoadModel,
            basedir=self.dataset.getFile('model_dir'),
            trainDataDir=self.dataset.getFile('training_data_dir'))
        protImportTrainedModel = self.launchProtocol(protImportTrainedModel)

        # Predict
        protPredict = self.newProtocol(ProtCryoCAREPrediction,
                                       even=protImportEven.outputTomograms,
                                       odd=protImportOdd.outputTomograms,
                                       model=getattr(protImportTrainedModel, 'model', None))

        self.launchProtocol(protPredict)
        output = getattr(protPredict, 'outputTomograms', None)
        self.assertEqual(output.getDim(), (1236, 1279, 209))
        self.assertEqual(output.getSize(), 1)
        self.assertEqual(output.getSamplingRate(), self.sRate)
        self.assertTrue(exists(protPredict._getExtraPath('Tomo110_bin6_denoised.mrc')))

    def testWorkflow(self):
        importTomoProtEven = self._runImportTomograms('tomo_even')
        importTomoProtOdd = self._runImportTomograms('tomo_odd')
        prepTrainingDataProt = self._runPrepareTrainingData(importTomoProtEven, importTomoProtOdd)
        trainingProt = self._runTrainingData(prepTrainingDataProt)
        self._runPredict(importTomoProtEven, importTomoProtOdd, trainingProt)
