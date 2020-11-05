import json
from os.path import join, abspath

from pwem.protocols import EMProtocol
from pyworkflow.protocol import params
from pyworkflow.utils import Message, getParentFolder, removeBaseExt

from cryocare import Plugin
from tomo.objects import Tomogram
from tomo.protocols import ProtTomoBase
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
                      label='Tomogram (from even frames)',
                      important=True,
                      help='Set of tomogram reconstructed from the even frames of the tilt'
                           'series movies.')

        form.addParam('odd', params.PointerParam,
                      pointerClass='SetOfTomograms',
                      label='Tomogram (from odd frames)',
                      important=True,
                      help='Set of tomograms reconstructed from the odd frames of the tilt'
                           'series movies.')

        form.addParam('model', params.PointerParam,
                      pointerClass='CryocareModel',
                      label="cryoCARE Model",
                      important=True,
                      help='Select a trained cryoCARE model.')

        form.addSection(label='Memory Management')
        form.addParam('mrc_slice_shape', params.IntParam,
                      default=1200,
                      label='Side length of sub-volumes',
                      help='Denoising is performed in chunks to reduce memory consumption. '
                           'Sub-volumes with this side length are loaded into memory.')

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        # Insert processing steps
        for evenTomo, oddTomo in zip(self.even.get(), self.odd.get()):
            self._insertFunctionStep('preparePredictStep', evenTomo, oddTomo)
            self._insertFunctionStep('predictStep')
        self._insertFunctionStep('createOutputStep')

    def preparePredictStep(self, evenTomo, oddTomo):
        outputName = self._getOutputName(evenTomo)
        self._outputFiles.append(outputName)
        config = {
            'model_name': self.model.get()._model_name.get(),
            'path': self.model.get()._basedir.get(),
            'meanStdPath': getParentFolder(self.model.get().getMeanStd()),
            'even': evenTomo.getFileName(),
            'odd': oddTomo.getFileName(),
            'output_name': outputName,
            'mrc_slice_shape': 3 * [self.mrc_slice_shape.get()]
        }
        self._configPath = self._getTmpPath('prediction_config.json')
        with open(self._configPath, 'w+') as f:
            json.dump(config, f, indent=2)

    def predictStep(self):
        Plugin.runCryocare(self, 'cryoCARE_predict.py', '--conf {}'.format(self._configPath))

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
                "Tomogram denoising finished: {}".format(join(self.model._basedir.get(), self.ouput_name.get())))
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
