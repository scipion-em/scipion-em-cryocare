import json
from os.path import join, exists

from pwem.protocols import EMProtocol, StringParam
from pyworkflow.protocol import IntParam, PointerParam, FloatParam, params
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
                      help = 'Training data extracted from even and odd tomograms.')
        form.addSection(label='Training Parameters')
        form.addParam('epochs', IntParam,
                      default=200,
                      label='Training epochs',
                      help='Number of epochs for which the network is trained.')
        form.addParam('batch_size', IntParam,
                      default=16,
                      label='Batch size',
                      help='Size of the training batch.')
        form.addParam('learning_rate', FloatParam,
                      default=0.0004,
                      label='Learning rate',
                      help='Training learning rate.')
        form.addParam('model_name', StringParam,
                      default='model',
                      label='Model name',
                      help='Name of the model.')
        form.addSection(label='U-Net Parameters')
        form.addParam('unet_kern_size', IntParam,
                      default=3,
                      label='Convolution kernel size',
                      help='Size of the convolution kernels used in the U-Net.')
        form.addParam('unet_n_depth', IntParam,
                      default=2,
                      label='U-Net depth',
                      help='Depth of the U-Net.')
        form.addParam('unet_n_first', IntParam,
                      default=16,
                      label='Number of initial feature channels',
                      help='Number of initial feature channels.')


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
            'model_name': self.model_name.get(),
            'path': self._getExtraPath()
        }

        self.config_path = params.String(join(self._getExtraPath(), 'train_config.json'))
        with open(self.config_path.get(), 'w+') as f:
            json.dump(config, f, indent=2)

    def trainingStep(self):
        Plugin.runCryocare(self, 'cryoCARE_train.py', '--conf {}'.format(self.config_path.get()))

    def createOutputStep(self):
        model = CryocareModel(basedir=self._getExtraPath(), model_name=self.model_name.get())
        self._defineOutputs(model=model)


    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        summary = []

        if self.isFinished():
            summary.append("Model saved to: {}".format(self._getExtraPath(self.model_name.get())))
        return summary

    def _validate(self):
        validateMsgs = []

        if not self.train_data.get():
            validateMsgs.append('Please select some training data.')

        if self.epochs.get() <= 0:
            validateMsgs.append('Epochs has to be > 0.')

        if self.batch_size.get() <= 0:
            validateMsgs.append('Batch size has to be > 0.')

        if self.learning_rate.get() <= 0:
            validateMsgs.append('The learning rate has to be > 0.')

        if self.unet_kern_size.get() % 2 != 1:
            validateMsgs.append('Convolution kernel size has to be an odd number.')

        if self.unet_n_depth.get() <= 0:
            validateMsgs.append('U-Net depth has to be > 0.')

        if self.unet_n_first.get() <= 0:
            validateMsgs.append('Number of initial feature channels has to be > 0.')

        return validateMsgs
