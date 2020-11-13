import json
from os import remove
from os.path import abspath, join

from pwem.protocols import EMProtocol
from pyworkflow.protocol import params
from pyworkflow.utils import Message, getParentFolder, removeBaseExt, makePath, copyFile
from scipion.constants import PYTHON

from cryocare import Plugin
from tomo.objects import Tomogram
from tomo.protocols import ProtTomoBase

from cryocare.constants import PREDICT_CONFIG, CRYOCARE_MODEL, MEAN_STD_FN
from cryocare.utils import CryocareUtils as ccutils


class ProtCryoCAREPrediction(EMProtocol, ProtTomoBase):
    """Generate the final restored tomogram by applying the cryoCARE trained network to both
tomograms followed by per-pixel averaging."""

    _label = 'CryoCARE Prediction'
    _configPath = None
    _outputFiles = []

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        # You need a params to belong to a section:
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('even', params.PointerParam,
                      pointerClass='SetOfTomograms',
                      label='Even tomograms',
                      important=True,
                      allowsNull=False,
                      help='Set of tomogram reconstructed from the even frames of the tilt'
                           'series movies.')

        form.addParam('odd', params.PointerParam,
                      pointerClass='SetOfTomograms',
                      label='Odd tomograms',
                      important=True,
                      allowsNull=False,
                      help='Set of tomograms reconstructed from the odd frames of the tilt'
                           'series movies.')

        form.addParam('model', params.PointerParam,
                      pointerClass='CryocareModel',
                      label="cryoCARE Model",
                      important=True,
                      allowsNull=False,
                      help='Select a trained cryoCARE model.')

        form.addSection(label='Memory Management')
        form.addParam('mrc_slice_shape', params.IntParam,
                      default=1200,
                      label='Side length of sub-volumes',
                      help='Denoising is performed in chunks to reduce memory consumption. '
                           'Sub-volumes with this side length are loaded into memory.')

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        numTomo = 1
        makePath(self._getPredictConfDir())
        # Insert processing steps
        for evenTomo, oddTomo in zip(self.even.get(), self.odd.get()):
            self._insertFunctionStep('preparePredictStep', evenTomo, oddTomo, numTomo)
            self._insertFunctionStep('predictStep')
            numTomo += 1

        self._insertFunctionStep('createOutputStep')

    def preparePredictStep(self, evenTomo, oddTomo, numTomo):
        outputName = self._getOutputName(evenTomo)
        self._outputFiles.append(outputName)
        config = {
            'model_name': CRYOCARE_MODEL,
            'path': self.model.get().getPath(),
            'even': evenTomo.getFileName(),
            'odd': oddTomo.getFileName(),
            'output_name': outputName,
            'mrc_slice_shape': 3 * [self.mrc_slice_shape.get()]
        }
        self._configPath = join(self._getPredictConfDir(), '{}_{:03d}.json'.format(PREDICT_CONFIG, numTomo))
        with open(self._configPath, 'w+') as f:
            json.dump(config, f, indent=2)

    def predictStep(self):
        # cryoCARE_predict.py expects the mean_std.npz file to be in the same directory as the model
        expectedMeanStdFile = join(self.model.get().getPath(), MEAN_STD_FN)
        copyFile(self.model.get().getMeanStd(), expectedMeanStdFile)
        # Run cryoCARE
        Plugin.runCryocare(self, PYTHON, '$(which cryoCARE_predict.py) --conf {}'.format(abspath(self._configPath)))
        # Remove copied file
        remove(expectedMeanStdFile)

    def createOutputStep(self):
        outputSetOfTomo = self._createSetOfTomograms(suffix='_denoised')
        outputSetOfTomo.copyInfo(self.even.get())

        for i, inTomo in enumerate(self.even.get()):
            tomo = Tomogram()
            tomo.setLocation(self._outputFiles[i])
            tomo.setSamplingRate(inTomo.getSamplingRate())
            outputSetOfTomo.append(tomo)

        self._defineOutputs(outputTomograms=outputSetOfTomo)

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        """ Summarize what the protocol has done"""
        summary = []

        if self.isFinished():
            summary.append(
                "Tomogram denoising finished.")
        return summary

    def _validate(self):
        validateMsgs = []

        msg = ccutils.checkInputTomoSetsSize(self.even.get(), self.odd.get())
        if msg:
            validateMsgs.append()
        return validateMsgs

    # --------------------------- UTIL functions -----------------------------------
    def _getOutputName(self, inTomo):
        outputName = removeBaseExt(inTomo.getFileName()) + '_denoised.mrc'
        return abspath(self._getExtraPath(outputName.replace('_Even', '').replace('_Odd', '')))

    def _getPredictConfDir(self):
        return self._getExtraPath(PREDICT_CONFIG)
