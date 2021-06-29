from os.path import exists, join

from pyworkflow.tests import BaseTest, setupTestProject
from . import DataSet
from ..constants import TRAIN_DATA_FN, VALIDATION_DATA_FN
from ..objects import CryocareTrainData, CryocareModel
from ..protocols import ProtCryoCARELoadTrainData, ProtCryoCARELoadModel


class TestCryoCAREImports(BaseTest):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('cryocare')

    def testLoadTrainingData(self):
        protImportTD = self.newProtocol(
            ProtCryoCARELoadTrainData,
            trainDataDir=self.dataset.getFile('training_data_dir'),
            trainConfigFile=self.dataset.getFile('training_data_conf'))
        self.launchProtocol(protImportTD)
        cryoCareTrainData = getattr(protImportTD, 'train_data', None)
        testTrainDataDir = cryoCareTrainData.getTrainDataDir()
        self.assertEqual(type(cryoCareTrainData), CryocareTrainData)
        self.assertEqual(testTrainDataDir, self.dataset.getFile('training_data_dir'))
        self.assertTrue(exists(join(testTrainDataDir, TRAIN_DATA_FN)))
        self.assertTrue(exists(join(testTrainDataDir, VALIDATION_DATA_FN)))
        self.assertEqual(cryoCareTrainData.getPatchSize(), 72)

    def testLoadTrainingModel(self):
        protImportTM = self.newProtocol(
            ProtCryoCARELoadModel,
            basedir=self.dataset.getFile('model_dir'),
            trainDataDir=self.dataset.getFile('training_data_dir'))
        self.launchProtocol(protImportTM)
        cryoCareModel = getattr(protImportTM, 'model', None)
        self.assertEqual(type(cryoCareModel), CryocareModel)
        self.assertEqual(cryoCareModel.getPath(), protImportTM._getExtraPath())
        self.assertEqual(cryoCareModel.getTrainDataDir(), protImportTM._getExtraPath())


