import glob
import json
from os import mkdir, rename
from os.path import join, exists
import numpy as np

from pwem.protocols import EMProtocol
from pyworkflow.protocol import params, IntParam, FloatParam
from pyworkflow.utils import Message

from cryocare import Plugin
from cryocare.constants import TRAIN_DATA_DIR, TRAIN_DATA_FN, MEAN_STD_FN
from cryocare.objects import CryocareTrainData


class ProtCryoCAREPrepareTrainingData(EMProtocol):
    """Operate the data to make it be expressed as expected by cryoCARE net."""

    _label = 'CryoCARE Training Data Extraction'
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
                      label='Tomogram (from even frames)',
                      important=True,
                      help='Set of tomograms reconstructed from the even frames of the tilt'
                           'series movies.')

        form.addParam('oddTomos', params.PointerParam,
                      pointerClass='SetOfTomograms',
                      label='Tomogram (from odd frames)',
                      important=True,
                      help='Set of tomogram reconstructed from the odd frames of the tilt'
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
        numTomo = 1
        trainDataDir = self._getTrainDataDir()
        if not exists(trainDataDir):
            mkdir(trainDataDir)

        for evenTomo, oddTomo in zip(self.evenTomos.get(), self.oddTomos.get()):
            self._insertFunctionStep('prepareTrainingDataStep', evenTomo, oddTomo, numTomo)
            self._insertFunctionStep('runDataExtraction', numTomo)
            numTomo += 1

        self._insertFunctionStep('createOutputStep')

    def prepareTrainingDataStep(self, evenTomo, oddTomo, numTomo):
        config = {
            'even': evenTomo.getFileName(),
            'odd': oddTomo.getFileName(),
            'patch_shape': 3 * [self.patch_shape.get()],
            'num_slices': self.num_slices.get(),
            'split': self.split.get(),
            'path': self._getExtraPath('train_data')
        }
        self._configFile = self._getTmpPath('train_data_config_{:03d}.json'.format(numTomo))
        with open(self._configFile, 'w+') as f:
            json.dump(config, f, indent=2)

    def runDataExtraction(self, numTomo):
        Plugin.runCryocare(self, 'cryoCARE_extract_train_data.py', '--conf {}'.format(self._configFile))
        # Rename the generated files to preserve them so they can be merged in createOutputStep
        rename(self._getTrainDataFile(), self._getTmpPath('{:03d}_{}'.format(numTomo, TRAIN_DATA_FN)))
        rename(self._getMeanStdFile(), self._getTmpPath('{:03d}_{}'.format(numTomo, MEAN_STD_FN)))

    def createOutputStep(self):
        trainDataFile = self._getTrainDataFile()
        meanStdFile = self._getMeanStdFile()
        # Combine all train_data and mean_std files into one
        self._combineTrainDataFiles(self._getTmpPath('*' + TRAIN_DATA_FN), trainDataFile)  # train_data files
        self._combineTrainDataFiles(self._getTmpPath('*' + MEAN_STD_FN), meanStdFile)  # mea_std files
        # Generate a train data object containing the resulting data
        train_data = CryocareTrainData(train_data=trainDataFile,
                                       mean_std=meanStdFile)
        self._defineOutputs(train_data=train_data)

    # --------------------------- INFO functions -----------------------------------
    # def _summary(self):
    #     summary = []
    #
    #     if self.isFinished():
    #         summary.append("Saved config file to: {}".format(self.config_path.get()))
    #     return summary

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

    # --------------------------- UTIL functions -----------------------------------
    @staticmethod
    def _combineTrainDataFiles(pattern, outputFile):
        files = glob.glob(pattern)
        if len(files) == 1:
            rename(files[0], outputFile)
        else:
            trainingFiles = [np.load(file) for file in files]
            np.save(outputFile, np.concatenate(trainingFiles))

    def _getTrainDataDir(self):
        return self._getExtraPath(TRAIN_DATA_DIR)

    def _getTrainDataFile(self):
        return join(self._getTrainDataDir(), TRAIN_DATA_FN)

    def _getMeanStdFile(self):
        return join(self._getTrainDataDir(), MEAN_STD_FN)

