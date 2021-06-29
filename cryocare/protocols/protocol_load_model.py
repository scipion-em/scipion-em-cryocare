import glob
from os.path import exists, join

from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.protocol import PathParam, FileParam
from pyworkflow.utils import Message, createLink

from cryocare.constants import TRAIN_DATA_FN, VALIDATION_DATA_FN, CRYOCARE_MODEL
from cryocare.objects import CryocareModel
from cryocare.utils import makeDatasetSymLinks


class ProtCryoCARELoadModel(EMProtocol):
    """Use two data-independent reconstructed tomograms to train a 3D cryo-CARE network."""

    _label = 'CryoCARE Load Model'
    _devStatus = BETA

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        # You need a params to belong to a section:
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('basedir', PathParam,
                      label='Base directory of the trained cryoCARE model',
                      important=True,
                      allowsNull=False,
                      help='It must contain a model in .h5 format.')
        form.addParam('trainDataDir', FileParam,
                      label='Directory of the prepared data for training',
                      important=True,
                      allowsNull=False,
                      help='It must contain two files: train_data.npz and val_data.npz, generated when '
                           'preparing the training data.')

    def _insertAllSteps(self):
        self._initialize()
        self._insertFunctionStep(self.createOutputStep)

    def _initialize(self):
        # The prediction is expecting the training and validation datasets to be in the same place as the training
        # model, but they are located in the training data generation extra directory. Hence, a symbolic link will
        # be created
        makeDatasetSymLinks(self, self.trainDataDir.get())
        createLink(join('..', self.basedir.get()), self._getExtraPath(CRYOCARE_MODEL))

    def createOutputStep(self):
        model = CryocareModel(basedir=self._getExtraPath(), train_data_dir=self._getExtraPath())
        self._defineOutputs(model=model)

    # --------------------------- INFO functions -----------------------------------
    def _validate(self):
        errors = []
        if not exists(self.basedir.get()):
            errors.append('Training model base directory does not exists.')
        elif not glob.glob(join(self.basedir.get(), '*.h5')):
            errors.append('No model files were found in the introduced training model base directory.')

        if not exists(self.trainDataDir.get()):
            errors.append('Directory of the prepared data for training does not exists.')
        else:
            if not exists(join(self.trainDataDir.get(), TRAIN_DATA_FN)):
                errors.append('No %s file was found in the introduced training model base directory.' 
                              % TRAIN_DATA_FN)
            if not exists(join(self.trainDataDir.get(), VALIDATION_DATA_FN)):
                errors.append('No %s file was found in the introduced training model base directory.'
                              % VALIDATION_DATA_FN)
        return errors

    def _summary(self):
        summary = []

        if self.isFinished():
            summary.append("Loaded training model_dir = *%s*" % self.basedir.get())
        return summary
