import glob
import json
import operator
from enum import Enum
from os.path import join
import numpy as np

from cryocare.protocols.protocol_base import ProtCryoCAREBase
from cryocare.utils import checkInputTomoSetsSize, getModelName
from pyworkflow import BETA
from pyworkflow.protocol import params, IntParam, FloatParam, Positive, LT, GT, GE, LEVEL_ADVANCED, EnumParam
from pyworkflow.utils import makePath, moveFile
from scipion.constants import PYTHON

from cryocare import Plugin
from cryocare.constants import TRAIN_DATA_DIR, TRAIN_DATA_FN, TRAIN_DATA_CONFIG, VALIDATION_DATA_FN, CRYOCARE_MODEL
from cryocare.objects import CryocareModel


# Tilt axis values
X_AXIS = 0
Y_AXIS = 1
Z_AXIS = 2
X_AXIS_LABEL = 'X'
Y_AXIS_LABEL = 'Y'
Z_AXIS_LABEL = 'Z'


class Outputobjects(Enum):
    model = CryocareModel


class ProtCryoCARETraining(ProtCryoCAREBase):
    """Operate the data to make it be expressed as expected by cryoCARE net."""

    _label = 'CryoCARE Training'
    _devStatus = BETA
    _possibleOutputs = Outputobjects

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._configFile = None
        self._configPath = None

    # -------------------------- DEFINE param functions ----------------------

    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        super()._defineParams(form)
        form.addSection(label='Config Parameters')
        form.addParam('tilt_axis', EnumParam,
                      label='Tilt axis of the tomograms',
                      expertLevel=params.LEVEL_ADVANCED,
                      choices=[X_AXIS_LABEL, Y_AXIS_LABEL, Z_AXIS_LABEL],
                      default=Y_AXIS,
                      allowsNull=False,
                      display=EnumParam.DISPLAY_HLIST,
                      help='Tomograms are split along this axis to extract train and validation data separately.')

        form.addParam('patch_shape', IntParam,
                      label='Side length of the training volumes',
                      default=72,
                      help='Corresponding sub-volumes pairs of the provided 3D shape '
                           'are extracted from the even and odd tomograms. The higher it is,'
                           'the higher net depth is required for training and the longer it '
                           'takes. Its value also depends on the resolution of the input tomograms, '
                           'being a higher patch size required for higher resolution.')

        form.addParam('num_slices', IntParam,
                      label='Number of training pairs to extract',
                      default=1200,
                      validators=[Positive],
                      help='Number of sub-volumes to sample from the even and odd tomograms.')

        form.addParam('n_normalization_samples', IntParam,
                      label='No. of subvolumes extracted per tomogram',
                      default=120,
                      expertLevel=LEVEL_ADVANCED,
                      validators=[Positive],
                      help='Number of training pairs which will be used to compute mean and standard deviation '
                           'for normalization. By default it is the 10% of the number of training pairs.')

        form.addParam('split', FloatParam,
                      label='Train-Validation Split',
                      default=0.9,
                      validators=[GT(0), LT(1)],
                      expertLevel=LEVEL_ADVANCED,
                      help='Training and validation data split value.')

        form.addSection(label='Training Parameters')
        form.addParam('epochs', IntParam,
                      default=100,
                      label='Training epochs',
                      validators=[Positive],
                      help='Number of epochs for which the network is trained. '
                           'An epoch refers to one cycle through the full training dataset. '
                           'It gives the network a chance to see the previous data to readjust '
                           'the model parameters so that the model is not biased towards the '
                           'last few data points during training.')
        form.addParam('steps_per_epoch', IntParam,
                      label='Steps per epoch',
                      default=200,
                      validators=[Positive],
                      help='Number of gradient steps performed per epoch.')
        form.addParam('batch_size', IntParam,
                      default=16,
                      label='Batch size',
                      validators=[Positive],
                      help='Size of the training batch. '
                           'An entire big dataset cannot be passed into the neural net at once, '
                           'so it is divided into batches. The batch size is the total number of '
                           'training examples present in a single batch.')
        form.addParam('learning_rate', FloatParam,
                      default=0.0004,
                      label='Learning rate',
                      validators=[Positive],
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

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self._initialize()
        self._insertFunctionStep(self.prepareTrainingDataStep)
        self._insertFunctionStep(self.runDataExtraction)
        self._insertFunctionStep(self.prepareTrainingStep)
        self._insertFunctionStep(self.trainingStep)
        self._insertFunctionStep(self.createOutputStep)

    def _initialize(self):
        makePath(self._getTrainDataConfDir())
        self._configFile = join(self._getTrainDataConfDir(), TRAIN_DATA_CONFIG)
        self._configPath = self._getExtraPath('train_config.json')

    def prepareTrainingDataStep(self):

        if self.areEvenOddLinked.get():
            fnOdd, fnEven = self.getOddEvenLists()
        else:
            self._getListOfTomoNames(self.evenTomos.get() )
            fnOdd = self._getListOfTomoNames(self.oddTomos.get())
            fnEven = self._getListOfTomoNames(self.evenTomos.get())

        config = {
            'even': fnEven,
            'odd': fnOdd,
            'patch_shape': 3 * [self.patch_shape.get()],
            'num_slices': self.num_slices.get(),
            'split': self.split.get(),
            'tilt_axis': self._decodeTiltAxisValue(self.tilt_axis.get()),
            'n_normalization_samples': self.n_normalization_samples.get(),
            'path': self._getExtraPath('train_data')
        }
        with open(self._configFile, 'w+') as f:
            json.dump(config, f, indent=2)

    def runDataExtraction(self):
        Plugin.runCryocare(self, PYTHON, '$(which cryoCARE_extract_train_data.py) --conf %s' % self._configFile)

    def prepareTrainingStep(self):
        config = {
            'train_data': self._getTrainDataDir(),
            'epochs': self.epochs.get(),
            'steps_per_epoch': self.steps_per_epoch.get(),
            'batch_size': self.batch_size.get(),
            'unet_kern_size': self.unet_kern_size.get(),
            'unet_n_depth': self._getUNetDepth(),
            'unet_n_first': self.unet_n_first.get(),
            'learning_rate': self.learning_rate.get(),
            'model_name': CRYOCARE_MODEL,
            'path': self._getExtraPath()
        }
        with open(self._configPath, 'w+') as f:
            json.dump(config, f, indent=2)

    def trainingStep(self):
        Plugin.runCryocare(self, PYTHON, '$(which cryoCARE_train.py) --conf {}'.format(self._configPath),
                           gpuId=getattr(self, params.GPU_LIST).get())

    def createOutputStep(self):
        model = CryocareModel(model_file=getModelName(self),
                              train_data_dir=self._getTrainDataDir())
        self._defineOutputs(**{Outputobjects.model.name: model})

        if self.areEvenOddLinked.get():
            self._defineSourceRelation(self.tomo.get(), model)
        else:
            self._defineSourceRelation(self.oddTomos.get(), model)
            self._defineSourceRelation(self.evenTomos.get(), model)

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        summary = []

        if self.isFinished():
            summary.append("Generated training data info:\n"
                           "train_data_file = *{}*\n"
                           "validation_data_file = *{}*\n"
                           "patch_size = *{}*".format(
                            self._getTrainDataFile(),
                            self._getValidationDataFile(),
                            self.patch_shape.get()))
        return summary

    def _validate(self):
        validateMsgs = []
        sideLength = self.patch_shape.get()

        if self.areEvenOddLinked.get():
            inputTomo = self.tomo.get()
            xt, yt, zt = inputTomo.getDimensions()
            for idim in [xt, yt, zt]:
                if idim <= 2 * sideLength:
                    validateMsgs.append('X, Y and Z dimensions of the tomograms introduced must satisfy the '
                                        'condition\n\n*dimension > 2 x SideLength*\n\n'
                                        '(X, Y, Z) = (%i, %i, %i)\n'
                                        'SideLength = %i\n\n' % (xt, yt, zt, sideLength))
                    break
        else:
            evenTomos = self.evenTomos.get()
            oddTomos = self.oddTomos.get()
            xe, ye, ze = evenTomos.getDimensions()
            xo, yo, zo = oddTomos.getDimensions()
            for idim in [xe, ye, ze, xo, yo, zo]:
                if idim <= 2 * sideLength:
                    validateMsgs.append('X, Y and Z dimensions of the tomograms introduced must satisfy the '
                                        'condition\n\n*dimension > 2 x SideLength*\n\n'
                                        '(X, Y, Z) = (%i, %i, %i)\n'
                                        'SideLength = %i\n\n' % (xe, ye, ze, sideLength))
                    msg = checkInputTomoSetsSize(evenTomos, oddTomos)
                    if msg:
                        validateMsgs.append(msg)
                    break

        # Check the patch conditions
        if sideLength % 2 != 0:
            validateMsgs.append('Patch shape has to be an even number.')

        return validateMsgs

    # --------------------------- UTIL functions -----------------------------------
    @staticmethod
    def _combineTrainDataFiles(pattern, outputFile):
        files = glob.glob(pattern)
        if len(files) == 1:
            moveFile(files[0], outputFile)
        else:
            # Create a dictionary with the data fields contained in each npz file
            dataDict = {}
            with np.load(files[0]) as data:
                for field in data.files:
                    dataDict[field] = []

            # Read and combine the data from all files
            for i, name in enumerate(files):
                with np.load(name) as data:
                    for field in data.files:
                        dataDict[field].append(data[field])

            # Save the combined data into a npz file
            np.savez(outputFile, **dataDict)

    def _getTrainDataDir(self):
        return self._getExtraPath(TRAIN_DATA_DIR)

    def _getTrainDataFile(self):
        return join(self._getTrainDataDir(), TRAIN_DATA_FN)

    def _getValidationDataFile(self):
        return join(self._getTrainDataDir(), VALIDATION_DATA_FN)

    def _getTrainDataConfDir(self):
        return self._getExtraPath(TRAIN_DATA_CONFIG)

    @staticmethod
    def _decodeTiltAxisValue(value):
        if value == X_AXIS:
            return X_AXIS_LABEL
        elif value == Y_AXIS:
            return Y_AXIS_LABEL
        else:
            return Z_AXIS_LABEL

    @staticmethod
    def _getListOfTomoNames(tomoSet):
        return [tomo.getFileName() for tomo in tomoSet]

    def getOddEvenLists(self):
        oddList = []
        evenList = []
        for t in self.tomo.get():
            odd, even = t.getHalfMaps().split(',')
            oddList.append(odd)
            evenList.append(even)
        return oddList, evenList

    def _getUNetDepth(self):
        # Estimate the best net depth value according to the patch size if the user left this field empty
        if self.unet_n_depth.get() == 0:
            refValues = [72, 96, 128]  # Corresponds to a net depth of 2, 3 and 4, respectively
            netDepth = [2, 3, 4]
            diff = [abs(i - self.patch_shape.get()) for i in refValues]
            ind, _ = min(enumerate(diff), key=operator.itemgetter(0))
            return netDepth[ind]
        else:
            return self.unet_n_depth.get()
