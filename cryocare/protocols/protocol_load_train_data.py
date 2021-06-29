import json
from os.path import join

from pwem.protocols import EMProtocol, FileParam
from pyworkflow import BETA
from pyworkflow.protocol import PathParam
from pyworkflow.utils import Message, createLink

from cryocare.constants import TRAIN_DATA_FN, VALIDATION_DATA_FN, TRAIN_DATA_DIR
from cryocare.objects import CryocareTrainData


class ProtCryoCARELoadTrainData(EMProtocol):
    """Use two data-independent reconstructed tomograms to train a 3D cryo-CARE network."""

    _label = 'CryoCARE Load Training Data'
    _devStatus = BETA

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        # You need a params to belong to a section:
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('trainDataDir', PathParam,
                      label='Training data directory',
                      important=True,
                      allowsNull=False,
                      help='Path of the training data extracted from even and odd monograms. '
                           'It must contain the following files:\n\n   - *{}*\n   - *{}*'.format(
                            TRAIN_DATA_FN, VALIDATION_DATA_FN))
        form.addParam('trainConfigFile', FileParam,
                      label='Training config file',
                      important=True,
                      allowsNull=False,
                      help='Config file generated in the corresponding training data preparation. '
                           'Used to get the patch size.')

    def _insertAllSteps(self):
        self._initialize()
        self._insertFunctionStep(self.createOutputStep)

    def _initialize(self):
        createLink(join('..', self.trainDataDir.get()), self._getExtraPath(TRAIN_DATA_DIR))

    def createOutputStep(self):
        train_data = CryocareTrainData(train_data_dir=self.trainDataDir.get(),
                                       patch_size=self._getPatchSize())
        self._defineOutputs(train_data=train_data)

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        summary = []

        if self.isFinished():
            summary.append("Loaded training data info:\n"
                           "train_data_dir = *{}*\n"
                           "patch_size = *{}*".format(
                            self.trainDataDir.get(),
                            self._getPatchSize()))
        return summary

    # --------------------------- UTIL functions -----------------------------------
    def _getPatchSize(self):
        with open(self.trainConfigFile.get()) as json_file:
            data = json.load(json_file)
            return data['patch_shape'][0]
