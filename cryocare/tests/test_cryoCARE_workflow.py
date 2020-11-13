from os.path import exists
from pyworkflow.tests import BaseTest, setupTestProject
import tomo.protocols
from . import DataSet
from ..constants import CRYOCARE_MODEL, MEAN_STD_FN, TRAIN_DATA_FN, TRAIN_DATA_CONFIG, TRAIN_DATA_DIR
from ..objects import CryocareTrainData, CryocareModel
from ..protocols import ProtCryoCAREPrediction, ProtCryoCAREPrepareTrainingData, ProtCryoCARETraining


class TestCryoCARE(BaseTest):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('cryocare')
        cls.sRate = 2.355

    def _runImportTomograms(self, tomoFile):
        protImport = self.newProtocol(
            tomo.protocols.ProtImportTomograms,
            filesPath=self.dataset.getFile(tomoFile),
            samplingRate=self.sRate)
        self.launchProtocol(protImport)
        output = getattr(protImport, 'outputTomograms', None)
        self.assertSetSize(output, size=1)
        return protImport

    def _runPrepareTrainingData(self, protImportEven, protImportOdd):
        protPrepTrainingData = self.newProtocol(ProtCryoCAREPrepareTrainingData,
                                                evenTomos=protImportEven.outputTomograms,
                                                oddTomos=protImportOdd.outputTomograms)
        self.launchProtocol(protPrepTrainingData)
        output = getattr(protPrepTrainingData, 'train_data', None)

        # Check generated object
        self.assertEqual(type(output), CryocareTrainData)
        self.assertEqual(output.getTrainData(), protPrepTrainingData._getExtraPath(TRAIN_DATA_DIR, TRAIN_DATA_FN))
        self.assertEqual(output.getMeanStd(), protPrepTrainingData._getExtraPath(TRAIN_DATA_DIR, MEAN_STD_FN))
        self.assertEqual(output.getPatchSize(), 64)
        # Check files generated
        self.assertTrue(exists(protPrepTrainingData._getExtraPath(TRAIN_DATA_DIR, TRAIN_DATA_FN)))
        self.assertTrue(exists(protPrepTrainingData._getExtraPath(TRAIN_DATA_DIR, MEAN_STD_FN)))
        self.assertTrue(exists(protPrepTrainingData._getExtraPath(TRAIN_DATA_CONFIG,'training_data_config_001.json')))

        return protPrepTrainingData

    def _runTrainingData(self, protPrepTrainingData):
        protTraining = self.newProtocol(ProtCryoCARETraining,
                                        train_data=getattr(protPrepTrainingData, 'train_data', None))

        self.launchProtocol(protTraining)
        output = getattr(protTraining, 'model', None)
        # Check generated model
        self.assertEqual(type(output), CryocareModel)
        self.assertEqual(output.getPath(), protTraining._getExtraPath())
        self.assertEqual(output.getMeanStd(), protPrepTrainingData.train_data.getMeanStd())
        # Check files generated
        self.assertTrue(exists(protTraining._getExtraPath('train_config.json')))
        self.assertTrue(exists(protTraining._getExtraPath(CRYOCARE_MODEL, 'config.json')))
        self.assertTrue(exists(protTraining._getExtraPath(CRYOCARE_MODEL, 'weights_best.h5')))
        self.assertTrue(exists(protTraining._getExtraPath(CRYOCARE_MODEL, 'weights_last.h5')))
        return protTraining

    def _runPredict(self, protImportEven, protImportOdd, protPredict):
        protPredict = self.newProtocol(ProtCryoCAREPrediction,
                                       even=protImportEven.outputTomograms,
                                       odd=protImportOdd.outputTomograms,
                                       model=getattr(protPredict, 'model', None))

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
