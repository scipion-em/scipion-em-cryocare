# **************************************************************************
# *
# * Authors:     Scipion Team
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
import glob
import json
import re
import shutil
from enum import Enum
from os.path import join
from cryocare.utils import checkInputTomoSetsSize
from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.protocol import params, StringParam
from pyworkflow.utils import Message, makePath
from scipion.constants import PYTHON
from cryocare import Plugin
from tomo.objects import Tomogram, SetOfTomograms
from cryocare.constants import PREDICT_CONFIG


DENOISED_SUFFIX = 'denoised'
EVEN = 'even'


class outputObjects(Enum):
    tomograms = SetOfTomograms


class ProtCryoCAREPrediction(EMProtocol):
    """Generate the final restored tomogram by applying the cryoCARE trained network to both
tomograms followed by per-pixel averaging."""

    _label = 'CryoCARE Prediction'
    _devStatus = BETA
    _possibleOutputs = outputObjects

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._configPath = {}
        self.sRate = None

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        # You need a params to belong to a section:
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('useIndependentOddEven', params.BooleanParam,
                      default=True,
                      label="Are odd-even associated to the Tomograms?",
                      help=" .")
        form.addParam('even', params.PointerParam,
                      pointerClass='SetOfTomograms',
                      condition='useIndependentOddEven',
                      label='Even tomograms',
                      important=True,
                      allowsNull=False,
                      help='Set of tomograms reconstructed from the even frames of the tilt'
                           'series movies.')
        form.addParam('odd', params.PointerParam,
                      pointerClass='SetOfTomograms',
                      condition='useIndependentOddEven',
                      label='Odd tomograms',
                      important=True,
                      allowsNull=False,
                      help='Set of tomogram reconstructed from the odd frames of the tilt'
                           'series movies.')
        form.addParam('tomo', params.PointerParam,
                      pointerClass='SetOfTomograms',
                      condition='not useIndependentOddEven',
                      label='Tomograms',
                      important=True,
                      help='Set of tomograms reconstructed from the even frames of the tilt'
                           'series movies.')

        form.addParam('model', params.PointerParam,
                      pointerClass='CryocareModel',
                      label="cryoCARE Model",
                      important=True,
                      allowsNull=False,
                      help='Select a trained cryoCARE model.')

        form.addParam('n_tiles', StringParam,
                      label="Number of tiles",
                      default='1 1 1',
                      important=True,
                      allowsNull=False,
                      help='Normally the gpu cannot handle the whole size of the tomograms, so it can be split into '
                           'n tiles per axis to process smaller volumes instead of one big at once.')

        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="GPU ID, normally it is 0.")

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self._initialize()

        if self.useIndependentOddEven.get():
            # Insert processing steps
            for evenTomo, oddTomo in zip(self.even.get(), self.odd.get()):
                tsId = evenTomo.getTsId()
                self._insertFunctionStep(self.preparePredictStep, tsId, evenTomo.getFileName(), oddTomo.getFileName())
                self._insertFunctionStep(self.predictStep, tsId)
                self._insertFunctionStep(self.createOutputStep, tsId)
        else:
            for t in self.tomo.get():
                tsId = t.getTsId()
                self._insertFunctionStep(self.preparePredictFullTomoStep, tsId, t)
                self._insertFunctionStep(self.predictStep, tsId)
                self._insertFunctionStep(self.createOutputStep, tsId)


    def _initialize(self):
        makePath(self._getPredictConfDir())
        if self.useIndependentOddEven.get():
            self.sRate = self.even.get().getSamplingRate()
        else:
            self.sRate = self.tomo.get().getSamplingRate()


    def preparePredictFullTomoStep(self, tsId, inputTomo):
        odd, even = inputTomo.getHalfMaps().split(',')
        self.preparePredictStep(tsId, odd, even)

    def preparePredictStep(self, tsId, evenTomo, oddTomo):
        config = {
            'path': self.model.get().getPath(),
            'even': evenTomo,
            'odd': oddTomo,
            'n_tiles': [int(i) for i in self.n_tiles.get().split()],
            'output': self._getOutputPath(tsId),
            'overwrite': False
        }
        self._configPath[tsId] = join(self._getPredictConfDir(), '%s_%s.json' % (PREDICT_CONFIG, tsId))
        with open(self._configPath[tsId], 'w+') as f:
            json.dump(config, f, indent=2)

    def predictStep(self, tsId):
        # Run cryoCARE
        Plugin.runCryocare(self, PYTHON, '$(which cryoCARE_predict.py) --conf %s' % self._configPath[tsId],
                           gpuId=getattr(self, params.GPU_LIST).get())
        # Remove even/odd words from the output name to avoid confusion
        origName = self._getOutputFile(tsId)
        finalNameRe = re.compile(re.escape(EVEN), re.IGNORECASE)  # Used to do a case-insensitive replacement
        shutil.move(origName, finalNameRe.sub('', origName))

    def createOutputStep(self, tsId):
        outputSetOfTomo = getattr(self, outputObjects.tomograms.name, None)
        if not outputSetOfTomo:
            outputSetOfTomo = SetOfTomograms.create(self._getPath(), template='tomograms%s.sqlite', suffix=DENOISED_SUFFIX)
            if self.useIndependentOddEven.get():
                outputSetOfTomo.copyInfo(self.even.get())
            else:
                outputSetOfTomo.copyInfo(self.tomo.get())
        tomo = self._genOutputTomogram(tsId)
        outputSetOfTomo.append(tomo)

        self._defineOutputs(**{outputObjects.tomograms.name: outputSetOfTomo})
        if self.useIndependentOddEven.get():
            self._defineSourceRelation(self.even.get(), outputSetOfTomo)
            self._defineSourceRelation(self.odd.get(), outputSetOfTomo)
        else:
            self._defineSourceRelation(self.tomo.get(), outputSetOfTomo)
        self._defineSourceRelation(self.model.get(), outputSetOfTomo)

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        """ Summarize what the protocol has done"""
        summary = []

        if self.isFinished():
            summary.append("Tomogram denoising finished.")
        return summary

    def _validate(self):
        validateMsgs = []
        # Check the sampling rate
        if self.useIndependentOddEven.get():
            sRateEven = self.even.get().getSamplingRate()
            sRateOdd = self.odd.get().getSamplingRate()
            if sRateEven != sRateOdd:
                validateMsgs.append('The sampling rate of the introduced sets of tomograms is different:\n'
                                    'Even SR %.2f != Odd SR %.2f\n\n' % (sRateEven, sRateOdd))
            msg = checkInputTomoSetsSize(self.even.get(), self.odd.get())
            if msg:
                validateMsgs.append(msg)

        return validateMsgs

    # --------------------------- UTIL functions -----------------------------------
    def _getPredictConfDir(self):
        return self._getExtraPath(PREDICT_CONFIG)

    def _getOutputPath(self, tsId):
        """cryoCARE will generate a new folder for each tomogram denoised. Apart from that, if the
        tomograms were imported, the 'Even_' word can be included in the tsId, as in that case it will be
        the filename. To avoid confusion, it's removed from the generated folder name."""
        outPath = self._getExtraPath(tsId + '_' + DENOISED_SUFFIX)
        outPathRe = re.compile(re.escape(EVEN), re.IGNORECASE)  # Used to carry out a case-insensitive replacement
        return outPathRe.sub('', outPath)

    def _getOutputFile(self, tsId):
        return glob.glob(join(self._getOutputPath(tsId), '*.mrc'))[0] # Only one file is contained in each dir

    def _genOutputTomogram(self, tsId):
        tomo = Tomogram()
        tomo.setLocation(self._getOutputFile(tsId))
        tomo.setSamplingRate(self.sRate)
        return tomo
