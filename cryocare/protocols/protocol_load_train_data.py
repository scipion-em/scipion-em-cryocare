from pwem.protocols import EMProtocol, StringParam, FileParam
from pyworkflow.utils import Message

from cryocare.objects import CryocareTrainData


class ProtCryoCARELoadTrainData(EMProtocol):
    """Use two data-independent reconstructed tomograms to train a 3D cryo-CARE network."""

    _label = 'CryoCARE Load Training Data'

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        # You need a params to belong to a section:
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('train_data', FileParam,
                      label='Training data',
                      important=True,
                      help='Training data extracted from even and odd tomograms.')
        form.addParam('mean_std', FileParam,
                      label='Normalization Parameters',
                      important=True,
                      help='Path to mean and standard deviation used for normalization.')

    def _insertAllSteps(self):
        self._insertFunctionStep('createOutputStep')

    def createOutputStep(self):
        train_data = CryocareTrainData(train_data=self.train_data.get(), mean_std=self.mean_std.get())
        self._defineOutputs(train_data=train_data)

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        summary = []

        if self.isFinished():
            summary.append("Loaded training data {} and normalization parameters {}.".format(self.train_data.get(),
                                                                                             self.mean_std.get()))
        return summary
