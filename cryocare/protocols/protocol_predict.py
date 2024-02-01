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

from cryocare.protocols.protocol_base import ProtCryoCAREBase
from cryocare.utils import checkInputTomoSetsSize
from pyworkflow import BETA
from pyworkflow.protocol import params, StringParam
from pyworkflow.utils import makePath
from scipion.constants import PYTHON
from cryocare import Plugin
from tomo.objects import Tomogram, SetOfTomograms
from cryocare.constants import PREDICT_CONFIG

DENOISED_SUFFIX = 'denoised'
EVEN = 'even'


class Outputobjects(Enum):
    tomograms = SetOfTomograms


class ProtCryoCAREPrediction(ProtCryoCAREBase):
    """Generate the final restored tomogram by applying the cryoCARE trained network to both
tomograms followed by per-pixel averaging."""

    _label = 'CryoCARE Prediction'
    _devStatus = BETA
    _possibleOutputs = Outputobjects

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sRate = None

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        super()._defineParams(form)
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
        tomoListEven, tomoListOdd = self._initialize()
        for evenTomo, oddTomo in zip(tomoListEven, tomoListOdd):
            tsId = evenTomo.getTsId()
            self._insertFunctionStep(self.preparePredictStep, tsId, evenTomo.getFileName(), oddTomo.getFileName())
            self._insertFunctionStep(self.predictStep, tsId)
            self._insertFunctionStep(self.createOutputStep, evenTomo)

    def _initialize(self):
        makePath(self._getPredictConfDir())
        tomoSet = self.tomo.get() if self.areEvenOddLinked.get() else self.evenTomos.get()
        self.sRate = tomoSet.getSamplingRate()
        if self.areEvenOddLinked.get():
            tomoList = [tomo.clone() for tomo in self.tomo.get()]
            tomoListEven = []
            tomoListOdd = []
            for tomo in tomoList:
                odd, even = tomo.getHalfMaps().split(',')
                tomoListEven.append(even)
                tomoListOdd.append(odd)
        else:
            tomoListEven = [tomo.clone() for tomo in self.evenTomos.get()]
            tomoListOdd = [tomo.clone() for tomo in self.oddTomos.get()]
        return tomoListEven, tomoListOdd

    def preparePredictStep(self, tsId, evenTomo, oddTomo):
        config = {
            'path': self.model.get().getPath(),
            'even': evenTomo,
            'odd': oddTomo,
            'n_tiles': [int(i) for i in self.n_tiles.get().split()],
            'output': self._getOutputPath(tsId),
            'overwrite': False
        }
        with open(self.getConfigPath(tsId), 'w+') as f:
            json.dump(config, f, indent=2)

    def predictStep(self, tsId):
        # Run cryoCARE
        Plugin.runCryocare(self, PYTHON, '$(which cryoCARE_predict.py) --conf %s' % self.getConfigPath(tsId),
                           gpuId=getattr(self, params.GPU_LIST).get())
        # Remove even/odd words from the output name to avoid confusion
        origName = self._getOutputFile(tsId)
        finalNameRe = re.compile(re.escape(EVEN), re.IGNORECASE)  # Used to do a case-insensitive replacement
        shutil.move(origName, finalNameRe.sub('', origName))

    def createOutputStep(self, tomo):
        outputSetOfTomo = getattr(self, Outputobjects.tomograms.name, None)
        if not outputSetOfTomo:
            outputSetOfTomo = SetOfTomograms.create(self._getPath(),
                                                    template='tomograms%s.sqlite',
                                                    suffix=DENOISED_SUFFIX)
            if self.areEvenOddLinked.get():
                outputSetOfTomo.copyInfo(self.tomo.get())
            else:
                outputSetOfTomo.copyInfo(self.evenTomos.get())
        outTomo = self._genOutputTomogram(tomo)
        outputSetOfTomo.append(outTomo)

        self._defineOutputs(**{Outputobjects.tomograms.name: outputSetOfTomo})
        if self.areEvenOddLinked.get():
            self._defineSourceRelation(self.tomo.get(), outputSetOfTomo)
        else:
            self._defineSourceRelation(self.evenTomos.get(), outputSetOfTomo)
            self._defineSourceRelation(self.oddTomos.get(), outputSetOfTomo)
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
        if not self.areEvenOddLinked.get():
            sRateEven = self.evenTomos.get().getSamplingRate()
            sRateOdd = self.oddTomos.get().getSamplingRate()
            if sRateEven != sRateOdd:
                validateMsgs.append('The sampling rate of the introduced sets of tomograms is different:\n'
                                    'Even SR %.2f != Odd SR %.2f\n\n' % (sRateEven, sRateOdd))
            msg = checkInputTomoSetsSize(self.evenTomos.get(), self.oddTomos.get())
            if msg:
                validateMsgs.append(msg)

        return validateMsgs

    # --------------------------- UTIL functions -----------------------------------
    def _getPredictConfDir(self):
        return self._getExtraPath(PREDICT_CONFIG)

    def getConfigPath(self, tsId):
        return join(self._getPredictConfDir(), '%s_%s.json' % (PREDICT_CONFIG, tsId))

    def _getOutputPath(self, tsId):
        """cryoCARE will generate a new folder for each tomogram denoised. Apart from that, if the
        tomograms were imported, the 'Even_' word can be included in the tsId, as in that case it will be
        the filename. To avoid confusion, it's removed from the generated folder name."""
        outPath = self._getExtraPath(tsId + '_' + DENOISED_SUFFIX)
        outPathRe = re.compile(re.escape(EVEN), re.IGNORECASE)  # Used to carry out a case-insensitive replacement
        return outPathRe.sub('', outPath)

    def _getOutputFile(self, tsId):
        return glob.glob(join(self._getOutputPath(tsId), '*.mrc'))[0]  # Only one file is contained in each dir

    def _genOutputTomogram(self, inTomo):
        tomo = Tomogram()
        tomo.copyInfo(inTomo)
        tomo.setLocation(self._getOutputFile(inTomo.getTsId()))
        return tomo
