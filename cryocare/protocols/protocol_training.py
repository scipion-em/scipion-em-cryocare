import json
import operator
from os.path import join

from pwem.protocols import EMProtocol
from pyworkflow.protocol import IntParam, PointerParam, FloatParam, params, GT, LEVEL_ADVANCED, GE
from pyworkflow.utils import Message
from scipion.constants import PYTHON

from cryocare import Plugin
from cryocare.constants import CRYOCARE_MODEL
from cryocare.objects import CryocareModel


class ProtCryoCARETraining(EMProtocol):
    """Use two data-independent reconstructed tomograms to train a 3D cryo-CARE network."""

    _label = 'CryoCARE Training'
    _configPath = None

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
                      help='Number of epochs for which the network is trained. '
                           'An epoch refers to one cycle through the full training dataset. '
                           'It gives the network a chance to see the previous data to readjust '
                           'the model parameters so that the model is not biased towards the '
                           'last few data points during training.')
        form.addParam('batch_size', IntParam,
                      default=16,
                      label='Batch size',
                      validators=[GT(0)],
                      help='Size of the training batch. '
                           'An entire big dataset cannot be passed into the neural net at once, '
                           'so it is divided into batches. The batch size is the total number of '
                           'training examples present in a single batch.')
        form.addParam('learning_rate', FloatParam,
                      default=0.0004,
                      label='Learning rate',
                      validators=[GT(0)],
                      expertLevel=LEVEL_ADVANCED,
                      help='Training learning rate. '
                           'In machine learning and statistics, the learning rate is a tuning '
                           'parameter in an optimization algorithm that determines the step size '
                           'at each iteration while moving toward a minimum of a loss function. '
                           'Large learning rates result in unstable training and tiny rates '
                           'result in a failure to train.')
        form.addSection(label='U-Net Parameters')
        form.addParam('unet_kern_size', IntParam,
                      default=3,
                      label='Convolution kernel size',
                      help='Size of the convolution kernels used in the U-Net. '
                           'Convolutional neural networks are basically a stack of layers '
                           'which are defined by the action of a number of filters on the input. '
                           'Those filters are usually called kernels. They can be conceptually '
                           'interpreted as feature extractors.')
        form.addParam('unet_n_depth', IntParam,
                      default=0,
                      label='U-Net depth',
                      validators=[GE(0)],
                      help='Depth of the U-Net.')
        form.addParam('unet_n_first', IntParam,
                      default=16,
                      label='Number of initial feature channels',
                      validators=[GT(0)],
                      expertLevel=LEVEL_ADVANCED,
                      help='Number of initial feature channels.')
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="GPU ID, normally it is 0.")

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
            'unet_n_depth': self._getUNetDepth(),
            'unet_n_first': self.unet_n_first.get(),
            'learning_rate': self.learning_rate.get(),
            'model_name': CRYOCARE_MODEL,
            'path': self._getExtraPath()
        }

        self._configPath = self._getExtraPath('train_config.json')
        with open(self._configPath, 'w+') as f:
            json.dump(config, f, indent=2)

    def trainingStep(self):
        Plugin.runCryocare(self, PYTHON, '$(which cryoCARE_train.py) --conf {}'.format(self._configPath))

    def createOutputStep(self):
        model = CryocareModel(basedir=self._getExtraPath(),
                              mean_std=self.train_data.get().getMeanStd())
        self._defineOutputs(model=model)

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        summary = []

        if self.isFinished():
            summary.append("Generated training model info:\n"
                           "model_dir = *{}*\n"
                           "normalization_file = *{}*".format
                           (self._getExtraPath(CRYOCARE_MODEL),
                            self.train_data.get().getMeanStd()))
        return summary

    def _validate(self):
        validateMsgs = []

        if self.unet_kern_size.get() % 2 != 1:
            validateMsgs.append('Convolution kernel size has to be an odd number.')

        return validateMsgs

# --------------------------- UTIL functions -----------------------------------
    def _getUNetDepth(self):
        # Estimate the best net depth value according to the patch size if the user left this field empty
        if self.unet_n_depth.get() == 0:
            refValues = [72, 96, 128]  # Corresponds to a net depth of 2, 3 and 4, respectively
            netDepth = [2, 3, 4]
            diff = [abs(i - self.train_data.get().getPatchSize()) for i in refValues]
            ind, _ = min(enumerate(diff), key=operator.itemgetter(0))
            return netDepth[ind]
        else:
            return self.unet_n_depth.get()

