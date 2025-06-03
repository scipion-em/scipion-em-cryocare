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
from pyworkflow.object import Set
from pyworkflow.protocol import params, StringParam, STEPS_PARALLEL
from pyworkflow.utils import makePath
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
    stepsExecutionMode = STEPS_PARALLEL

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sRate = None
        self.tomoDictEven = {}
        self.tomoDictOdd = {}

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

        form.addParallelSection(threads=1, mpi=0)
        form.addHidden(params.GPU_LIST, params.StringParam,
                       default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="GPU ID, normally it is 0.")

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self._initialize()
        closeSetStepDeps = []
        for tsId in self.tomoDictEven.keys():
            predId = self._insertFunctionStep(self.predictStep, tsId,
                                     prerequisites=[],
                                     needsGPU=True)
            cOutId = self._insertFunctionStep(self.createOutputStep, tsId,
                                     prerequisites=predId,
                                     needsGPU=False)
            closeSetStepDeps.append(cOutId)
        self._insertFunctionStep(self._closeOutputSet,
                                 prerequisites=closeSetStepDeps,
                                 needsGPU=False)

    def _initialize(self):
        makePath(self._getPredictConfDir())
        tomoSet = self.tomos.get() if self.areEvenOddLinked.get() else self.evenTomos.get()
        self.sRate = tomoSet.getSamplingRate()
        if self.areEvenOddLinked.get():
            for tomo in self.tomos.get():
                tsId = tomo.getTsId()
                odd, even = tomo.getHalfMaps().split(',')

                oddTomo = Tomogram()
                oddTomo.copyInfo(tomo)
                oddTomo.setLocation(odd)

                evenTomo = Tomogram()
                evenTomo.copyInfo(tomo)
                evenTomo.setLocation(even)

                self.tomoDictEven[tsId] = evenTomo
                self.tomoDictOdd[tsId] = oddTomo

        else:
            for tomoEven, tomoOdd in zip(self.evenTomos.get(), self.oddTomos.get()):
                tsId = tomoEven.getTsId()  # Use the same tsId (it may be different for both sets) for both dicts
                self.tomoDictEven[tsId] = tomoEven.clone()
                self.tomoDictOdd[tsId] = tomoOdd.clone()

    def predictStep(self, tsId):
        # Generate the config file: it is in this step instead of in a convertInputStep because of the
        # GPU parallelization from Scipion and the need of declaring that convertInputStep with the
        # attribute needsGpu = True only to be able to access the gpuId assigned, which may be problematic
        # in some cases
        self._genConfigFile(tsId)

        # Run cryoCARE
        Plugin.runCryocare(self, 'cryoCARE_predict.py','--conf %s' % self.getConfigPath(tsId))
        # Remove even/odd words from the output name to avoid confusion
        origName = self._getOutputFile(tsId)
        finalNameRe = re.compile(re.escape(EVEN), re.IGNORECASE)  # Used to do a case-insensitive replacement
        shutil.move(origName, finalNameRe.sub('', origName))

    def createOutputStep(self, tsId: str):
        with self._lock:
            outTomos = self._getOutputSetOfTomograms()
            inTomo = self.tomoDictEven[tsId]
            outTomo = self._genOutputTomogram(inTomo)
            outTomos.append(outTomo)
            outTomos.update(outTomo)
            outTomos.write()
            self._store(outTomos)

    # --------------------------- INFO functions -----------------------------------
    def _summary(self) -> list:
        """ Summarize what the protocol has done"""
        summary = []

        if self.isFinished():
            summary.append("Tomogram denoising finished.")
        return summary

    def _validate(self) -> list:
        validateMsgs = []
        super()._validate()
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
    def _genConfigFile(self, tsId: str) -> None:
        evenTomo = self.tomoDictEven[tsId]
        oddTomo = self.tomoDictOdd[tsId]
        # We do this to accept both GPU specified as '0' 1 2 3' or '0,1,2,3':
        gpuId = self._stepsExecutor.getGpuList()
        gpuId = gpuId[0]
        config = {
            'path': self.model.get().getPath(),
            'even': evenTomo.getFileName(),
            'odd': oddTomo.getFileName(),
            'n_tiles': [int(i) for i in self.n_tiles.get().split()],
            'output': self._getOutputPath(tsId),
            'overwrite': False,
            'gpu_id': gpuId
        }
        with open(self.getConfigPath(tsId), 'w+') as f:
            json.dump(config, f, indent=2)

    def _getPredictConfDir(self) -> str:
        return self._getExtraPath(PREDICT_CONFIG)

    def getConfigPath(self, tsId) -> str:
        return join(self._getPredictConfDir(), '%s_%s.json' % (PREDICT_CONFIG, tsId))

    def _getOutputPath(self, tsId) -> str:
        """cryoCARE will generate a new folder for each tomogram denoised. Apart from that, if the
        tomograms were imported, the 'Even_' word can be included in the tsId, as in that case it will be
        the filename. To avoid confusion, it's removed from the generated folder name."""
        outPath = self._getExtraPath(tsId + '_' + DENOISED_SUFFIX)
        outPathRe = re.compile(re.escape(EVEN), re.IGNORECASE)  # Used to carry out a case-insensitive replacement
        return outPathRe.sub('', outPath)

    def _getOutputFile(self, tsId) -> str:
        return glob.glob(join(self._getOutputPath(tsId), '*'))[0]  # Only one file is contained in each dir

    def _genOutputTomogram(self, inTomo: Tomogram) -> Tomogram:
        tomo = Tomogram()
        tomo.copyInfo(inTomo)
        tomo.setLocation(self._getOutputFile(inTomo.getTsId()))
        return tomo

    def _getOutputSetOfTomograms(self) -> SetOfTomograms:
        outTomograms = getattr(self, self._possibleOutputs.tomograms.name, None)
        if outTomograms:
            outTomograms.enableAppend()
        else:
            even = None if self.areEvenOddLinked.get() else True
            inSetPointer = self.getInTomos(asPointer=True, even=even)
            outTomograms = SetOfTomograms.create(self._getPath(), template='tomograms%s.sqlite')
            outTomograms.copyInfo(inSetPointer.get())
            outTomograms.setStreamState(Set.STREAM_OPEN)
            self._defineOutputs(**{self._possibleOutputs.tomograms.name: outTomograms})
            if self.areEvenOddLinked.get():
                self._defineSourceRelation(inSetPointer, outTomograms)
            else:
                self._defineSourceRelation(self.getInTomos(even=True, asPointer=True), outTomograms)
                self._defineSourceRelation(self.getInTomos(even=False, asPointer=True), outTomograms)
            self._defineSourceRelation(self.model, outTomograms)
        return outTomograms
