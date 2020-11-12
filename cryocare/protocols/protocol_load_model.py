
from pwem.protocols import EMProtocol
from pyworkflow.protocol import PathParam, FileParam
from pyworkflow.utils import Message
from cryocare.objects import CryocareModel


class ProtCryoCARELoadModel(EMProtocol):
    """Use two data-independent reconstructed tomograms to train a 3D cryo-CARE network."""

    _label = 'CryoCARE Load Model'

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        # You need a params to belong to a section:
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('basedir', PathParam,
                      label='Basedir',
                      important=True,
                      allowsNull=False,
                      help='Base directory of the trained cryoCARE model.')
        form.addParam('meanStd', FileParam,
                      label='Normalization file (mean_std)',
                      important=True,
                      allowsNull=False,
                      help='mean_std.npz file generated when preparing the training data.')

    def _insertAllSteps(self):
        self._insertFunctionStep('createOutputStep')

    def createOutputStep(self):
        model = CryocareModel(basedir=self.basedir.get(), mean_std=self.meanStd.get())
        self._defineOutputs(model=model)

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        summary = []

        if self.isFinished():
            summary.append("Loaded training model info:\n"
                           "model_dir = *{}*\n"
                           "normalization_file = *{}*".format
                           (self.basedir.get(), self.meanStd.get()))
        return summary
