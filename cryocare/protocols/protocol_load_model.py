import json
from os.path import join, exists

from pwem.protocols import EMProtocol, StringParam, FileParam
from pyworkflow.protocol import IntParam, PointerParam, FloatParam, params
from pyworkflow.utils import Message

from cryocare import Plugin
from cryocare.objects import CryocareTrainData, CryocareModel


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
        form.addParam('basedir', FileParam,
                      label='Basedir',
                      important=True,
                      help='Basedirectory of the trained cryoCARE model.')
        form.addParam('model_name', StringParam,
                      label='Model name',
                      important=True,
                      help='Name of the cryoCARE model.')

    def _insertAllSteps(self):
        self._insertFunctionStep('createOutputStep')

    def createOutputStep(self):
        model = CryocareModel(basedir=self.basedir.get(), model_name=self.model_name.get())
        self._defineOutputs(model=model)


    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        summary = []

        if self.isFinished():
            summary.append("Loaded model {} from {}.".format(self.model_name.get(), self.basedir.get()))
        return summary
