import glob
import json
from os.path import join
import numpy as np

from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.protocol import params, IntParam, FloatParam, Positive, LT, GT, LEVEL_ADVANCED, EnumParam
from pyworkflow.utils import Message, makePath, moveFile
from scipion.constants import PYTHON

from cryocare import Plugin
from cryocare.constants import TRAIN_DATA_DIR, TRAIN_DATA_FN, TRAIN_DATA_CONFIG, VALIDATION_DATA_FN
from cryocare.objects import CryocareTrainData
from cryocare.utils import CryocareUtils as ccutils

# Tilt axis values
X_AXIS = 0
Y_AXIS = 1
Z_AXIS = 2
X_AXIS_LABEL = 'X'
Y_AXIS_LABEL = 'Y'
Z_AXIS_LABEL = 'Z'


class ProtCryoCAREPrepareTrainingData(EMProtocol):
    """Operate the data to make it be expressed as expected by cryoCARE net."""

    _label = 'CryoCARE Training Data Extraction'
    _devStatus = BETA
    _configFile = None

    # -------------------------- DEFINE param functions ----------------------

    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        # You need a params to belong to a section:
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('evenTomos', params.PointerParam,
                      pointerClass='SetOfTomograms',
                      label='Even tomograms',
                      important=True,
                      allowsNull=False,
                      help='Set of tomograms reconstructed from the even frames of the tilt'
                           'series movies.')
        form.addParam('oddTomos', params.PointerParam,
                      pointerClass='SetOfTomograms',
                      label='Odd tomograms',
                      important=True,
                      allowsNull=False,
                      help='Set of tomogram reconstructed from the odd frames of the tilt'
                           'series movies.')

        form.addSection(label='Config Parameters')
        form.addParam('tilt_axis', EnumParam,
                      label='Tilt axis of the tomograms',
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

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self._initialize()
        self._insertFunctionStep(self.prepareTrainingDataStep)
        self._insertFunctionStep(self.runDataExtraction)
        self._insertFunctionStep(self.createOutputStep)

    def _initialize(self):
        makePath(self._getTrainDataDir())
        makePath(self._getTrainDataConfDir())
        self._configFile = join(self._getTrainDataConfDir(), TRAIN_DATA_CONFIG)

    def prepareTrainingDataStep(self):
        config = {
            'even': self._getListOfTomoNames(self.evenTomos.get()),
            'odd': self._getListOfTomoNames(self.oddTomos.get()),
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

    def createOutputStep(self):
        # Generate a train data object containing the resulting data
        train_data = CryocareTrainData(train_data_dir=self._getTrainDataDir(),
                                       patch_size=self.patch_shape.get())
        self._defineOutputs(train_data=train_data)

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

        msg = ccutils.checkInputTomoSetsSize(self.evenTomos.get(), self.oddTomos.get())
        if msg:
            validateMsgs.append()

        if self.patch_shape.get() % 2 != 0:
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

