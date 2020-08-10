from pwem.protocols import EMProtocol
from pyworkflow.protocol import params, Integer
from pyworkflow.utils import Message


class ProtCryocarePrepareTrainingData(EMProtocol):
    """Operate the data to make it be expressed as expected by cryoCARE net."""

    _label = 'cryocare prepare training data'

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

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        # Insert processing steps
        self._insertFunctionStep('prepareTrainingDataStep')

    def prepareTrainingDataStep(self):
        pass

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        """ Summarize what the protocol has done"""
        summary = []

        if self.isFinished():
            summary.append("Some message to summarize.")
        return summary

    def _methods(self):
        methods = []
        return methods
