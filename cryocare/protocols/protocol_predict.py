import json
from os.path import join

from pwem.protocols import EMProtocol
from pyworkflow.protocol import params
from pyworkflow.utils import Message

from cryocare import Plugin
from tomo.objects import Tomogram


class ProtCryoCAREPrediction(EMProtocol):
    """Generate the final restored tomogram by applying the cryoCARE trained network to both
tomograms followed by per-pixel averaging."""

    _label = 'CryoCARE Prediction'

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        # You need a params to belong to a section:
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('even', params.PointerParam,
                      pointerClass='Tomogram',
                      label='Tomogram (from even frames)',
                      important=True,
                      help='Tomogram reconstructed from the even frames of the tilt'
                           'series movies.')

        form.addParam('odd', params.PointerParam,
                      pointerClass='Tomogram',
                      label='Tomogram (from odd frames)',
                      important=True,
                      help='Tomogram reconstructed from the odd frames of the tilt'
                           'series movies.')

        form.addParam('model', params.PointerParam,
                      pointerClass='CryocareModel',
                      label="cryoCARE Model",
                      important=True,
                      help='Select a trained cryoCARE model.')

        form.addParam('output_name', params.StringParam,
                      default='denoised.mrc',
                      label='Output File Name',
                      help='Name of the denoised tomogram.')

        form.addSection(label='Memory Management')
        form.addParam('mrc_slice_shape', params.IntParam,
                      default=1200,
                      label='Side length of sub-volumes',
                      help='Denoising is performed in chunks to reduce memory consumption. '
                           'Sub-volumes with this side length are loaded into memory.')

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        # Insert processing steps
        self._insertFunctionStep('preparePredictStep')
        self._insertFunctionStep('predictStep')
        self._insertFunctionStep('createOutputStep')

    def preparePredictStep(self):
        config = {
            'model_name': self.model._model_name.get(),
            'path': self.model._basedir.get(),
            'even': self.even.get().getFileName(),
            'odd': self.odd.get().getFileName(),
            'output_name': self.output_name.get(),
            'mrc_slice_shape': 3 * [self.mrc_slice_shape.get()]
        }
        self.config_path = params.String(join(self._getExtraPath(), 'prediction_config.json'))
        with open(self.config_path.get(), 'w+') as f:
            json.dump(config, f, indent=2)

    def predictStep(self):
        Plugin.runCryocare(self, 'cryoCARE_predict.py', '--conf {}'.format(self.config_path.get()))

    def createOutputStep(self):
        denoised_tomo = Tomogram(location=join(self.model._basedir.get(), self.ouput_name.get()))
        self._defineOutputs(denoised_tomo=denoised_tomo)

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        """ Summarize what the protocol has done"""
        summary = []

        if self.isFinished():
            summary.append(
                "Tomogram denoising finished: {}".format(join(self.model._basedir.get(), self.ouput_name.get())))
        return summary
