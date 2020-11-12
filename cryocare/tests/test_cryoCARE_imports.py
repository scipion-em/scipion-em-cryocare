
from pyworkflow.tests import BaseTest, setupTestProject
from . import DataSet
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
        output = getattr(protImportTD, 'train_data', None)
        self.assertEqual(type(output), CryocareTrainData)
        self.assertEqual(output.getTrainData(), self.dataset.getFile('train_data_file'))
        self.assertEqual(output.getMeanStd(), self.dataset.getFile('mean_std_file'))
        self.assertEqual(output.getPatchSize(), 64)

    def testLoadTrainingModel(self):
        protImportTM = self.newProtocol(
            ProtCryoCARELoadModel,
            basedir=self.dataset.getFile('model_dir'),
            meanStd=self.dataset.getFile('mean_std_file'))
        self.launchProtocol(protImportTM)
        output = getattr(protImportTM, 'model', None)
        self.assertEqual(type(output), CryocareModel)
        self.assertEqual(output.getPath(), self.dataset.getFile('model_dir'))
        self.assertEqual(output.getMeanStd(), self.dataset.getFile('mean_std_file'))


