import json
from os import mkdir
from os.path import join

from pwem.protocols import EMProtocol
from pyworkflow.protocol import params, IntParam, FloatParam
from pyworkflow.utils import Message

from cryocare import Plugin
from cryocare.objects import CryocareTrainData


class ProtCryocarePrepareTrainingData(EMProtocol):
    """Operate the data to make it be expressed as expected by cryoCARE net."""

    _label = 'CryoCARE Training Data Extraction'

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        # You need a params to belong to a section:
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('evenTomo', params.PointerParam,
                      pointerClass='Tomogram',
                      label='Tomogram (from even frames)',
                      important=True,
                      help='Tomogram reconstructed from the even frames of the tilt'
                           'series movies.')

        form.addParam('oddTomo', params.PointerParam,
                      pointerClass='Tomogram',
                      label='Tomogram (from odd frames)',
                      important=True,
                      help='Tomogram reconstructed from the odd frames of the tilt'
                           'series movies.')

        form.addSection(label='Config Parameters')
        form.addParam('patch_shape', IntParam,
                      label='Side length of the training volumes',
                      default=72,
                      help='Corresponding sub-volumes pairs of the provided 3D shape '
                           'are extracted from the even and odd tomograms.')

        form.addParam('num_slices', IntParam,
                      label='Number of training pairs to extract',
                      default=1200,
                      help='Number of sub-volumes to sample from the even and odd tomograms.')

        form.addParam('split', FloatParam,
                      label='Train-Validation Split',
                      default=0.9,
                      help='Training- and validation-data split value.')

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('prepareTrainingDataStep')
        self._insertFunctionStep('runDataExtraction')
        self._insertFunctionStep('createOutputStep')

    def prepareTrainingDataStep(self):
        mkdir(self._getExtraPath('train_data'))
        config = {
            'even': self.evenTomo.get().getFileName(),
            'odd': self.oddTomo.get().getFileName(),
            'patch_shape': 3 * [self.patch_shape.get()],
            'num_slices': self.num_slices.get(),
            'split': self.split.get(),
            'path': self._getExtraPath('train_data')
        }
        self.config_path = params.String(join(self._getExtraPath(), 'train_data_config.json'))
        with open(self.config_path.get(), 'w+') as f:
            json.dump(config, f, indent=2)

    def runDataExtraction(self):
        Plugin.runCryocare(self, 'cryoCARE_extract_train_data.py', '--conf {}'.format(self.config_path.get()))

    def createOutputStep(self):
        train_data = CryocareTrainData(train_data=self._getExtraPath('train_data', 'train_data.npz'),
                                       mean_std=self._getExtraPath('train_data', 'mean_std.npz'))
        self._defineOutputs(train_data=train_data)

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        summary = []

        if self.isFinished():
            summary.append("Saved config file to: {}".format(self.config_path.get()))
        return summary

    def _validate(self):
        validateMsgs = []

        if self.patch_shape.get() % 2 != 0:
            validateMsgs.append('Patch shape has to be an even number.')

        if self.num_slices.get() <= 0:
            validateMsgs.append('Number of training pairs has to be > 0.')

        if self.split.get() >= 1.0:
            validateMsgs.append('Split has to be < 1.0.')

        if self.split.get() <= 0.0:
            validateMsgs.append('Split has to be > 0.0.')

        return validateMsgs

    def _citations(self):
        return ['buchholz2019cryo', 'buchholz2019content']
