from os.path import exists

from pyworkflow.object import Pointer
from pyworkflow.tests import BaseTest, setupTestProject
import tomo.protocols
from . import DataSet
from ..objects import CryocareTrainData, CryocareModel
from ..protocols import ProtCryoCAREPrediction, ProtCryoCAREPrepareTrainingData, ProtCryoCARETraining


class TestCryoCARE(BaseTest):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('cryocare')
        cls.sRate = 2.355 * 6  # Tomograms were binned to 6

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
        self.assertEqual(type(output), CryocareTrainData)
        # self.assertTrue(exists(protPrepTrainingData._getExtraPath('train_data_config.json')))
        self.assertTrue(exists(protPrepTrainingData._getExtraPath('train_data', 'mean_std.npz')))
        self.assertTrue(exists(protPrepTrainingData._getExtraPath('train_data', 'train_data.npz')))
        return protPrepTrainingData

    def _runTrainingData(self, protPrepTrainingData):
        protTraining = self.newProtocol(ProtCryoCARETraining,
                                        train_data=getattr(protPrepTrainingData, 'train_data', None))

        self.launchProtocol(protTraining)
        output = getattr(protTraining, 'model', None)
        self.assertEqual(type(output), CryocareModel)
        self.assertTrue(exists(protTraining._getExtraPath('train_config.json')))
        self.assertTrue(exists(protTraining._getExtraPath('model', 'config.json')))
        self.assertTrue(exists(protTraining._getExtraPath('model', 'weights_best.h5')))
        self.assertTrue(exists(protTraining._getExtraPath('model', 'weights_last.h5')))
        return protTraining

    def _runPredict(self, protImportEven, protImportOdd, protPrepTrainingData, protPredict):
        outputName = 'denoised_test.mrc'
        protPredict = self.newProtocol(ProtCryoCAREPrediction,
                                       meanStdPath=protPrepTrainingData._getExtraPath('train_data'),
                                       model=getattr(protPredict, 'model', None),
                                       output_name=outputName)

        protPredict.even = Pointer(protImportEven, extended='outputTomograms.1')
        protPredict.odd = Pointer(protImportOdd, extended='outputTomograms.1')

        self.launchProtocol(protPredict)
        output = getattr(protPredict, 'denoised_tomo', None)
        self.assertEqual(output.getDim(), (1236, 1279, 209))
        self.assertEqual(output.getSamplingRate(), self.sRate)
        self.assertTrue(exists(protPredict._getExtraPath('train_config.json')))
        self.assertTrue(exists(protPredict._getExtraPath('prediction_config.json')))
        self.assertTrue(exists(protPredict._getExtraPath(outputName)))

    def testWorkflow(self):
        importTomoProtEven = self._runImportTomograms('tomo_even')
        importTomoProtOdd = self._runImportTomograms('tomo_odd')
        prepTrainingDataProt = self._runPrepareTrainingData(importTomoProtEven, importTomoProtOdd)
        # trainingProt = self._runTrainingData(prepTrainingDataProt)
        # self._runPredict(importTomoProtEven, importTomoProtOdd, prepTrainingDataProt, trainingProt)
