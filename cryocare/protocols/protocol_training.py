import json
from os.path import join, exists

from pwem.protocols import EMProtocol, StringParam
from pyworkflow.protocol import IntParam, PointerParam, FloatParam, params, GT, Positive
from pyworkflow.utils import Message

from cryocare import Plugin
from cryocare.objects import CryocareTrainData, CryocareModel


class ProtCryoCARETraining(EMProtocol):
    """Use two data-independent reconstructed tomograms to train a 3D cryo-CARE network."""

    _label = 'CryoCARE Training'

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        # You need a params to belong to a section:
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('train_data', PointerParam,
                      pointerClass='CryocareTrainData',
                      label='Training data',
                      important=True,
                      allowsNull=False,
                      help='Training data extracted from even and odd tomograms.')
        form.addSection(label='Training Parameters')
        form.addParam('epochs', IntParam,
                      default=200,
                      label='Training epochs',
                      validators=[GT(0)],
                      help='Number of epochs for which the network is trained.')
        form.addParam('batch_size', IntParam,
                      default=16,
                      label='Batch size',
                      validators=[GT(0)],
                      help='Size of the training batch.')
        form.addParam('learning_rate', FloatParam,
                      default=0.0004,
                      label='Learning rate',
                      validators=[GT(0)],
                      help='Training learning rate.')
        form.addSection(label='U-Net Parameters')
        form.addParam('unet_kern_size', IntParam,
                      default=3,
                      label='Convolution kernel size',
                      help='Size of the convolution kernels used in the U-Net.')
        form.addParam('unet_n_depth', IntParam,
                      default=2,
                      label='U-Net depth',
                      validators=[GT(0)],
                      help='Depth of the U-Net.')
        form.addParam('unet_n_first', IntParam,
                      default=16,
                      label='Number of initial feature channels',
                      validators=[GT(0)],
                      help='Number of initial feature channels.')
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="GPU ID, normally it is 0. "
                            "TODO: Document better GPU IDs and threads. ")

    def _insertAllSteps(self):
        self._insertFunctionStep('prepareTrainingStep')
        self._insertFunctionStep('trainingStep')
        self._insertFunctionStep('createOutputStep')

    def prepareTrainingStep(self):
        config = {
            'train_data': self.train_data.get().getTrainData(),
            'epochs': self.epochs.get(),
            'batch_size': self.batch_size.get(),
            'unet_kern_size': self.unet_kern_size.get(),
            'unet_n_depth': self.unet_n_depth.get(),
            'unet_n_first': self.unet_n_first.get(),
            'learning_rate': self.learning_rate.get(),
            'model_name': 'cryoCARE_model',
            'path': self._getExtraPath()
        }

        self._config_path = join(self._getExtraPath(), 'train_config.json')
        with open(self._config_path, 'w+') as f:
            json.dump(config, f, indent=2)

    def trainingStep(self):
        Plugin.runCryocare(self, 'cryoCARE_train.py', '--conf {}'.format(self._config_path))

    def createOutputStep(self):
        model = CryocareModel(basedir=self._getExtraPath(),
                              model_name=self.model_name.get(),
                              mean_std=self.train_data.get().getMeanStd())
        self._defineOutputs(model=model)

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        summary = []

        if self.isFinished():
            summary.append("Model saved to: {}".format(self._getExtraPath(self.model_name.get())))
        return summary

    def _validate(self):
        validateMsgs = []

        if self.unet_kern_size.get() % 2 != 1:
            validateMsgs.append('Convolution kernel size has to be an odd number.')

        return validateMsgs
