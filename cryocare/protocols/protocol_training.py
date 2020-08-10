from pwem.protocols import EMProtocol
from pyworkflow.protocol import IntParam, PointerParam
from pyworkflow.utils import Message


class ProtCryocareTraining(EMProtocol):
    """Use two data-independent reconstructed tomograms to train a 3D cryo-CARE network."""

    _label = 'cryocare training'

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        # You need a params to belong to a section:
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('protPrepareTrainingData', PointerParam,
                      pointerClass='ProtCryocarePrepareTrainingData',
                      label='cryoCARE prepare training data run',
                      help = 'Select the previous cryoCARE training data run.')
        form.addSection(label='Training')
        form.addParam('epochs', IntParam,
                      default=16,
                      label='Training epochs',
                      help='Epochs')

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        # Insert processing steps
        self._insertFunctionStep('trainStep')
        self._insertFunctionStep('createOutputStep')

    def trainStep(self):
        pass

    def createOutputStep(self):
        pass

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        """ Summarize what the protocol has done"""
        summary = []

        if self.isFinished():
            summary.append("Some message to summarize.")
        return summary

    def _methods(self):
        return []
