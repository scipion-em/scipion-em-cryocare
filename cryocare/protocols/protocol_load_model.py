from enum import Enum
from os.path import exists, join

from cryocare.utils import makeDatasetSymLinks, getModelName
from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.protocol import PathParam, FileParam
from pyworkflow.utils import Message, createLink

from cryocare.constants import TRAIN_DATA_FN, VALIDATION_DATA_FN
from cryocare.objects import CryocareModel


class Outputobjects(Enum):
    model = CryocareModel


class ProtCryoCARELoadModel(EMProtocol):
    """Load a previously trained model."""

    _label = 'CryoCARE Load Trained Model'
    _devStatus = BETA
    _possibleOutputs = Outputobjects

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        # You need a params to belong to a section:
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('trainDataModel', PathParam,
                      label='Pre-trained cryoCARE model (.tar.gz)',
                      important=True,
                      allowsNull=False,
                      help='It is a .tar.gz file containing a folder that contains, in turn, the following files:\n\n'
                           '\t- config.json\n'
                           '\t- history.dat\n'
                           '\t- norm.json\n'
                           '\t- weights_best.h5\n'
                           '\t- weights_last.h5\n')
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
        createLink(join(self.trainDataModel.get()), getModelName(self))

    def createOutputStep(self):
        model = CryocareModel(model_file=getModelName(self), train_data_dir=self._getExtraPath())
        self._defineOutputs(**{Outputobjects.model.name: model})

    # --------------------------- INFO functions -----------------------------------
    def _validate(self):
        errors = []
        if not exists(self.trainDataModel.get()):
            errors.append('Training model introduced does not exists.')

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
            summary.append("Loaded training model_dir = *%s*" % self.trainDataModel.get())
        return summary

