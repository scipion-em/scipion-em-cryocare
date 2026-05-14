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
import logging
import re
import shutil
from enum import Enum
from os.path import join
from cryocare.protocols.protocol_base import ProtCryoCAREBase
from cryocare.utils import checkInputTomoSetsSize
from pyworkflow import BETA
from pyworkflow.object import Set
from pyworkflow.protocol import params, StringParam, STEPS_PARALLEL
from pyworkflow.utils import makePath, cyanStr, redStr
from cryocare import Plugin
from tomo.objects import Tomogram, SetOfTomograms
from cryocare.constants import PREDICT_CONFIG

logger = logging.getLogger(__name__)

DENOISED_SUFFIX = 'denoised'
EVEN = 'even'


class Outputobjects(Enum):
    tomograms = SetOfTomograms


class ProtCryoCAREPrediction(ProtCryoCAREBase):
    """
    Generates restored tomograms by applying a previously trained cryoCARE
    denoising model to paired tomographic datasets. The protocol combines
    information from complementary tomograms to improve signal quality,
    reduce noise, and produce cleaner reconstructions suitable for
    downstream cryo-electron tomography analysis.

    AI Generated:

    CryoCARE Prediction (ProtCryoCAREPrediction) — User Manual
        Overview

        The CryoCARE Prediction protocol applies a trained cryoCARE neural
        network to tomographic datasets in order to generate denoised and
        biologically more interpretable tomograms. Its main purpose is to
        suppress acquisition noise while preserving structural details that
        are essential for downstream visualization, segmentation, particle
        picking, subtomogram averaging, and structural interpretation.

        In cryo-electron tomography workflows, noise reduction is often one
        of the most critical preprocessing stages because tomograms are
        acquired under extremely low-dose conditions. Although these imaging
        conditions preserve the biological specimen, they also produce very
        noisy reconstructions. This protocol addresses that limitation by
        using a deep learning model trained to distinguish reproducible
        structural information from stochastic noise.

        Inputs and General Workflow

        The protocol requires a previously trained cryoCARE model together
        with paired tomographic datasets. These paired tomograms typically
        correspond to even and odd reconstructions generated independently
        from the same acquisition. Because both tomograms contain the same
        underlying biological signal but different noise realizations, the
        neural network can restore structural information while avoiding
        overfitting to noise.

        The workflow processes each tomogram pair independently. The trained
        model predicts restored versions of the input tomograms and combines
        their information into a final denoised reconstruction. This design
        allows the protocol to operate efficiently on large tomography
        datasets while preserving consistency across multiple samples.

        Biological Interpretation of Denoising

        From a biological perspective, denoising should improve visibility
        without altering meaningful structural features. Properly restored
        tomograms typically reveal membranes, macromolecular complexes,
        cytoskeletal elements, and organellar boundaries more clearly than
        raw reconstructions. This often facilitates interpretation and
        increases the reliability of subsequent analysis steps.

        However, users should remember that denoising does not generate new
        information. The protocol enhances reproducible signal already
        present in the data. Careful biological interpretation remains
        necessary, especially when evaluating weak densities, flexible
        regions, or low-abundance structures.

        Compatibility of Input Data

        The quality of prediction strongly depends on the consistency between
        the training data and the tomograms being restored. Ideally, the
        prediction datasets should have been acquired under imaging
        conditions similar to those used during training, including voxel
        size, acquisition strategy, reconstruction parameters, and overall
        contrast characteristics.

        The protocol supports workflows in which even and odd tomograms are
        already linked together, as well as workflows where both datasets
        are provided independently. Maintaining correct correspondence
        between paired tomograms is essential because mismatched pairs can
        lead to biologically unreliable restorations.

        Tiling and GPU Memory Management

        Tomographic datasets are often too large to process entirely in GPU
        memory. To address this limitation, the protocol allows tomograms to
        be divided into smaller three-dimensional tiles during prediction.
        This strategy enables processing of very large cellular tomograms
        while remaining compatible with a wide range of GPU hardware.

        Increasing the number of tiles reduces memory requirements but may
        increase execution time. In most practical situations, users should
        begin with the default configuration and only increase tiling when
        memory limitations occur. Extremely aggressive tiling may slightly
        reduce continuity between neighboring regions, particularly in very
        large tomograms.

        Parallel Prediction Workflow

        The protocol is designed to support parallel execution across
        multiple tomograms. This is particularly useful in large cryo-ET
        projects involving many tilt series or cellular datasets. Parallel
        processing significantly reduces total runtime and makes the
        protocol suitable for facility-scale or high-throughput workflows.

        Each tomogram is processed independently, allowing failed datasets
        to be identified without interrupting the prediction of the
        remaining samples. This behavior is especially valuable in large
        experiments where occasional corrupted inputs or reconstruction
        inconsistencies may occur.

        Outputs and Their Interpretation

        The protocol produces a new set of denoised tomograms that preserve
        the geometry and metadata of the original reconstructions while
        providing improved contrast and reduced noise. These outputs are
        intended for downstream cryo-electron tomography analysis and can
        be used directly in visualization software or subsequent Scipion
        protocols.

        In many biological applications, denoised tomograms improve the
        detectability of macromolecular assemblies and facilitate manual or
        automated annotation. Subtomogram averaging workflows may also
        benefit indirectly because cleaner tomograms often improve particle
        picking and alignment stability.

        Nevertheless, users should visually inspect restored tomograms to
        ensure that biologically relevant features are preserved and that no
        unexpected artifacts have been introduced. Comparison against the
        original tomograms is strongly recommended during validation.

        Practical Recommendations

        For most users, the best results are obtained when the prediction
        model was trained using tomograms closely matched to the target
        dataset. Differences in acquisition conditions or reconstruction
        strategies may reduce denoising quality and should be minimized
        whenever possible.

        Users working with large tomograms should monitor GPU memory usage
        and adjust tiling only when necessary. When multiple GPUs are
        available, distributing prediction tasks across hardware resources
        can substantially accelerate processing.

        It is also advisable to inspect a subset of denoised tomograms
        before processing an entire dataset. Early validation helps confirm
        that the restored contrast and structural appearance remain
        biologically meaningful.

        Final Perspective

        CryoCARE Prediction represents a powerful deep learning approach for
        improving the interpretability of cryo-electron tomography data.
        When applied carefully and validated appropriately, it can reveal
        structural features that are difficult to observe in raw
        reconstructions while preserving the integrity of the biological
        information contained in the tomograms.
    """

    _label = 'CryoCARE Prediction'
    _devStatus = BETA
    _possibleOutputs = Outputobjects
    stepsExecutionMode = STEPS_PARALLEL

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sRate = None
        self.tomoDictEven = {}
        self.tomoDictOdd = {}
        self.failedTsIds = []

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
        self._insertFunctionStep(self.closeOutputSetStep,
                                 prerequisites=closeSetStepDeps,
                                 needsGPU=False)

    def _initialize(self):
        makePath(self._getPredictConfDir())
        tomoSet = self.tomos.get() if self.areEvenOddLinked.get() else self.evenTomos.get()
        self.sRate = tomoSet.getSamplingRate()
        if self.areEvenOddLinked.get():
            for tomo in self.tomos.get():
                tsId = tomo.getTsId()
                even, odd = sorted(tomo.getHalfMaps(asList=True))

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
        logger.info(cyanStr(f'tsId = {tsId} - predicting...'))
        # Generate the config file: it is in this step instead of in a convertInputStep because of the
        # GPU parallelization from Scipion and the need of declaring that convertInputStep with the
        # attribute needsGpu = True only to be able to access the gpuId assigned, which may be problematic
        # in some cases
        try:
            self._genConfigFile(tsId)

            # Run cryoCARE
            Plugin.runCryocare(self, 'cryoCARE_predict.py','--conf %s' % self.getConfigPath(tsId))
            # Remove even/odd words from the output name to avoid confusion
            origName = self._getOutputFile(tsId)
            finalNameRe = re.compile(re.escape(EVEN), re.IGNORECASE)  # Used to do a case-insensitive replacement
            shutil.move(origName, finalNameRe.sub('', origName))
        except Exception as e:
            self.failedTsIds.append(tsId)
            logger.error(redStr(f'tsId = {tsId} - failed with exception {e}'))

    def createOutputStep(self, tsId: str):
        if tsId not in self.failedTsIds:
            logger.info(cyanStr(f'tsId = {tsId} - registering the output...'))
            with self._lock:
                outTomos = self._getOutputSetOfTomograms()
                inTomo = self.tomoDictEven[tsId]
                outTomo = self._genOutputTomogram(inTomo)
                outTomos.append(outTomo)
                outTomos.update(outTomo)
                outTomos.write()
                self._store(outTomos)

    def closeOutputSetStep(self):
        outTomos = getattr(self, self._possibleOutputs.tomograms.name, None)
        if not outTomos:
            raise Exception('No tomogram was predicted. Please '
                            'check the Output Log > run.stdout and run.stderr')
        self._closeOutputSet()

    # --------------------------- INFO functions -----------------------------------
    def _summary(self) -> list:
        """ Summarize what the protocol has done"""
        summary = []

        if self.isFinished():
            summary.append("Tomogram denoising finished.")
            inTomoSet = self.tomos.get() if self.areEvenOddLinked.get() else self.evenTomos.get()
            outTomoSet = getattr(self, self._possibleOutputs.tomograms.name, None)
            if len(outTomoSet) != len(inTomoSet):
                inTomosTsIds = inTomoSet.getTSIds()
                outTomoTsIds = outTomoSet.getTSIds()
                nonMatchingTsIds = set(inTomosTsIds) ^ set(outTomoTsIds)
                summary.append(f'*Some tomograms failed: {nonMatchingTsIds}*')
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
