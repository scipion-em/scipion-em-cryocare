
from pyworkflow.object import Pointer
from pyworkflow.tests import BaseTest, setupTestProject
import tomo.protocols
from . import DataSet
from ..protocols import ProtCryocareTraining


class TestTraining(BaseTest):
    """This class check if the protocol to import tomograms works properly."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('cryocare')
        cls.tomogram = cls.dataset.getFile('pattern')

    def _runImportTomograms(self):
        protImport = self.newProtocol(
            tomo.protocols.ProtImportTomograms,
            filesPath=self.tomogram,
            samplingRate=2.355)
        self.launchProtocol(protImport)
        return protImport

    def testTraining(self):
        protImport = self._runImportTomograms()
        output = getattr(protImport, 'outputTomograms', None)
        self.assertSetSize(output, size=2, msg="There was a problem with Import Tomograms protocol")
        protTraining = self.newProtocol(ProtCryocareTraining)
        protTraining.evenTomo = Pointer(protImport, extended='outputTomograms.1')
        protTraining.oddTomo = Pointer(protImport, extended='outputTomograms.2')

        self.launchProtocol(protTraining)
        # Define assertions
        self.assertTrue(hasattr(protTraining, 'cryocareModel'), "Model wasn't correctly generated.")
