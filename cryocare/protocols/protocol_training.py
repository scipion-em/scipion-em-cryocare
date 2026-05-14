import glob
import json
import operator
from enum import Enum
from os.path import join
import numpy as np

from cryocare.protocols.protocol_base import ProtCryoCAREBase
from cryocare.utils import checkInputTomoSetsSize, getModelName
from pyworkflow import BETA
from pyworkflow.protocol import params, IntParam, FloatParam, Positive, LT, GT, GE, LEVEL_ADVANCED, EnumParam
from pyworkflow.utils import makePath, moveFile

from cryocare import Plugin
from cryocare.constants import TRAIN_DATA_DIR, TRAIN_DATA_FN, TRAIN_DATA_CONFIG, VALIDATION_DATA_FN, CRYOCARE_MODEL
from cryocare.objects import CryocareModel

# Tilt axis values
X_AXIS = 0
Y_AXIS = 1
Z_AXIS = 2
X_AXIS_LABEL = 'X'
Y_AXIS_LABEL = 'Y'
Z_AXIS_LABEL = 'Z'


class Outputobjects(Enum):
    model = CryocareModel


class ProtCryoCARETraining(ProtCryoCAREBase):
    """
    Prepares and trains a cryoCARE denoising model for cryo-electron tomography by using paired even and odd tomograms.
    The protocol generates training and validation datasets, configures the neural network training process, and produces
    a trained model that can later be applied to restore tomographic data with reduced noise and improved interpretability.

    AI Generated:

    CryoCARE Training (ProtCryoCARETraining) — User Manual
        Overview

        The CryoCARE Training protocol is designed to generate and train a deep learning model specialized in denoising
        cryo-electron tomography data. Its purpose is to learn the statistical relationship between paired tomograms
        reconstructed from independent subsets of the same acquisition, commonly referred to as even and odd tomograms.
        By learning from these paired datasets, the protocol produces a neural network model capable of reducing noise
        while preserving biologically meaningful structural information.

        In cryo-electron tomography workflows, denoising is often essential because tomograms are inherently noisy due
        to low electron dose limitations. Improved signal quality can substantially enhance visualization, segmentation,
        particle picking, subtomogram averaging, and interpretation of macromolecular organization inside cells or
        purified samples.

        Inputs and Dataset Preparation

        The protocol requires paired tomographic datasets that represent statistically independent reconstructions of
        the same specimen regions. These pairs may already be linked internally or may be provided separately as even
        and odd tomogram collections. Proper pairing is critical because the neural network learns to distinguish
        reproducible structural information from random noise.

        During preparation, the tomograms are divided into many smaller three-dimensional subvolumes used for training
        and validation. These subvolumes are extracted across the tomographic volume to expose the network to a broad
        range of structural patterns and noise conditions. The number of extracted samples strongly influences training
        quality. Larger datasets generally improve robustness but also increase computational requirements and execution
        time.

        The protocol also computes normalization statistics from representative subvolumes. This normalization step is
        biologically important because it standardizes signal intensity distributions across tomograms and stabilizes
        neural network optimization. Poor normalization may reduce convergence quality or generate inconsistent denoising
        performance.

        Patch Size and Biological Interpretation

        One of the most important parameters is the training patch size, which determines the dimensions of the
        extracted subvolumes used during learning. Small patches are computationally efficient and suitable for local
        structural features, while larger patches allow the network to capture broader contextual information such as
        membrane continuity, organelle organization, or large macromolecular assemblies.

        In practice, the optimal patch size depends on voxel size, tomogram dimensions, and the biological scale of the
        structures of interest. High-resolution datasets or large cellular features often benefit from larger patches,
        although these require more memory and deeper neural network architectures. The tomograms must always be
        sufficiently larger than the chosen patch size to ensure meaningful extraction of training regions.

        Train and Validation Splitting

        The protocol separates the extracted data into training and validation subsets. The training subset is used for
        optimization of the neural network parameters, while the validation subset monitors generalization performance
        and helps detect overfitting.

        For most biological datasets, allocating the majority of samples to training while reserving a smaller fraction
        for validation provides a good balance. If the validation fraction is too small, model evaluation may become
        unreliable. Conversely, allocating too much data to validation may unnecessarily reduce training diversity.

        Tilt Axis Considerations

        The protocol allows the user to define the tomographic tilt axis used during extraction. This choice affects how
        training subvolumes are sampled and may influence the representation of anisotropic noise patterns introduced by
        tomographic reconstruction. Correct specification of the tilt axis is particularly important in datasets with
        pronounced missing wedge artifacts or directional reconstruction distortions.

        Training Parameters and Neural Network Optimization

        Training proceeds iteratively through multiple epochs. Each epoch corresponds to one full pass through the
        training dataset, allowing the network to progressively refine its denoising behavior. Increasing the number of
        epochs generally improves learning until convergence is reached, although excessive training may lead to
        overfitting.

        The number of optimization steps per epoch controls how many parameter updates are performed during each cycle.
        Larger values increase training intensity but also extend runtime. Batch size determines how many subvolumes are
        processed simultaneously during optimization. Large batch sizes may stabilize learning but require greater GPU
        memory resources.

        The learning rate is one of the most sensitive parameters in the protocol. Excessively large values may cause
        unstable optimization and poor convergence, whereas overly small values can dramatically slow training or
        prevent meaningful learning. For most cryo-electron tomography datasets, moderate default values provide a good
        starting point.

        U-Net Architecture and Feature Representation

        The denoising model is based on a U-Net architecture, which is widely used in biomedical image analysis because
        of its ability to combine local feature extraction with multiscale contextual understanding. The protocol allows
        adjustment of the convolution kernel size, network depth, and number of initial feature channels.

        Increasing network depth enables the model to capture larger contextual relationships and more complex
        structural patterns. This may improve denoising performance for large cellular datasets or highly heterogeneous
        specimens, although deeper networks require additional computational resources and larger training datasets.

        The number of feature channels controls the representational capacity of the network. Larger values may improve
        model expressiveness but also increase memory usage and training time. The protocol can automatically estimate a
        suitable network depth from the selected patch size, helping users obtain stable training configurations without
        extensive manual optimization.

        GPU Usage and Computational Requirements

        Training is computationally intensive and typically requires GPU acceleration for practical runtimes. The
        protocol supports execution on one or multiple GPUs, enabling efficient processing of large cryo-electron
        tomography datasets. Multi-GPU execution is especially beneficial when using large patch sizes, deep networks,
        or extensive training datasets.

        Biological users should be aware that larger and more complex models may improve denoising quality but also
        substantially increase hardware requirements and execution time.

        Outputs and Their Interpretation

        The primary output of the protocol is a trained cryoCARE denoising model. This model encapsulates the learned
        relationship between noisy tomograms and their underlying reproducible structural information. The model can be
        applied later to denoise additional tomograms acquired under similar imaging and reconstruction conditions.

        The protocol also generates organized training and validation datasets that document the extracted subvolumes
        used during optimization. These outputs are useful for reproducibility, quality control, and advanced workflow
        customization.

        From a biological perspective, denoised tomograms often provide improved visibility of membranes,
        macromolecular complexes, cytoskeletal structures, and intracellular organization. However, denoised data
        should always be interpreted carefully and ideally compared against the original tomograms to avoid
        overinterpretation of subtle features.

        Practical Recommendations

        For most workflows, it is advisable to begin with moderate patch sizes and default network settings. If the
        denoising quality is insufficient, increasing patch size or network depth may improve contextual understanding,
        especially for complex cellular environments.

        Careful inspection of even and odd tomogram pairing is essential before training. Incorrect pairing or
        mismatched datasets can severely degrade model quality. Users should also verify that tomogram dimensions are
        large enough relative to the chosen patch size.

        When computational resources are limited, reducing batch size or training depth may improve stability. For
        high-resolution cellular tomography projects, larger training datasets and longer optimization schedules often
        provide superior denoising performance.

        Final Perspective

        CryoCARE training represents a powerful strategy for improving cryo-electron tomography data quality through
        self-supervised deep learning. By leveraging statistically independent tomographic reconstructions, the protocol
        enables biologically meaningful noise reduction while preserving structural detail. Careful selection of
        training parameters, patch size, and network complexity is essential for achieving reliable denoising results
        that support downstream structural interpretation and analysis.
    """

    _label = 'CryoCARE Training'
    _devStatus = BETA
    _possibleOutputs = Outputobjects

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._configFile = None
        self._configPath = None

    # -------------------------- DEFINE param functions ----------------------

    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        super()._defineParams(form)
        form.addSection(label='Config Parameters')
        form.addParam('tilt_axis', EnumParam,
                      label='Tilt axis of the tomograms',
                      expertLevel=params.LEVEL_ADVANCED,
                      choices=[X_AXIS_LABEL, Y_AXIS_LABEL, Z_AXIS_LABEL],
                      default=Y_AXIS,
                      allowsNull=False,
                      display=EnumParam.DISPLAY_HLIST,
                      help='Tomograms are split along this axis to extract train and validation data separately.')

        form.addParam('patch_shape', IntParam,
                      label='Side length of the training volumes',
                      default=72,
                      help='Corresponding sub-volumes pairs of the provided 3D shape '
                           'are extracted from the even and odd tomograms. The higher it is,'
                           'the higher net depth is required for training and the longer it '
                           'takes. Its value also depends on the resolution of the input tomograms, '
                           'being a higher patch size required for higher resolution.')

        form.addParam('num_slices', IntParam,
                      label='Number of training pairs to extract per tomogram',
                      default=1200,
                      validators=[Positive],
                      help='Number of sub-volumes to sample from each pair of even and odd tomograms.')

        form.addParam('n_normalization_samples', IntParam,
                      label='No. of subvolumes used for normalization per tomogram',
                      default=120,
                      expertLevel=LEVEL_ADVANCED,
                      validators=[Positive],
                      help='Number of training pairs which will be used to compute mean and standard deviation '
                           'for normalization. By default this is 10% of the number of training pairs.')

        form.addParam('split', FloatParam,
                      label='Train-Validation Split',
                      default=0.9,
                      validators=[GT(0), LT(1)],
                      expertLevel=LEVEL_ADVANCED,
                      help='Training and validation data split value.')

        form.addSection(label='Training Parameters')
        form.addParam('epochs', IntParam,
                      default=100,
                      label='Training epochs',
                      validators=[Positive],
                      help='Number of epochs for which the network is trained. '
                           'An epoch refers to one cycle through the full training dataset. '
                           'It gives the network a chance to see the previous data to readjust '
                           'the model parameters so that the model is not biased towards the '
                           'last few data points during training.')
        form.addParam('steps_per_epoch', IntParam,
                      label='Steps per epoch',
                      default=200,
                      validators=[Positive],
                      help='Number of gradient steps performed per epoch.')
        form.addParam('batch_size', IntParam,
                      default=16,
                      label='Batch size',
                      validators=[Positive],
                      help='Size of the training batch. '
                           'An entire big dataset cannot be passed into the neural net at once, '
                           'so it is divided into batches. The batch size is the total number of '
                           'training examples present in a single batch.')
        form.addParam('learning_rate', FloatParam,
                      default=0.0004,
                      label='Learning rate',
                      validators=[Positive],
                      expertLevel=LEVEL_ADVANCED,
                      help='Training learning rate. '
                           'In machine learning and statistics, the learning rate is a tuning '
                           'parameter in an optimization algorithm that determines the step size '
                           'at each iteration while moving toward a minimum of a loss function. '
                           'Large learning rates result in unstable training and tiny rates '
                           'result in a failure to train.')
        form.addSection(label='U-Net Parameters')
        form.addParam('unet_kern_size', IntParam,
                      default=3,
                      label='Convolution kernel size',
                      help='Size of the convolution kernels used in the U-Net. '
                           'Convolutional neural networks are basically a stack of layers '
                           'which are defined by the action of a number of filters on the input. '
                           'Those filters are usually called kernels. They can be conceptually '
                           'interpreted as feature extractors.')
        form.addParam('unet_n_depth', IntParam,
                      default=0,
                      label='U-Net depth',
                      validators=[GE(0)],
                      help='Depth of the U-Net.')
        form.addParam('unet_n_first', IntParam,
                      default=16,
                      label='Number of initial feature channels',
                      validators=[GT(0)],
                      expertLevel=LEVEL_ADVANCED,
                      help='Number of initial feature channels.')

        form.addHidden(params.GPU_LIST, params.StringParam,
                       default='0',
                       label="Choose GPU IDs",
                       help="GPU IDs. The training supports parallelization over multiple GPUs "
                            "since cryoCARE version 0.3.0.")

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self._initialize()
        self._insertFunctionStep(self.prepareTrainingDataStep, needsGPU=False)
        self._insertFunctionStep(self.runDataExtraction, needsGPU=False)
        self._insertFunctionStep(self.prepareTrainingStep, needsGPU=False)
        self._insertFunctionStep(self.trainingStep, needsGPU=True)
        self._insertFunctionStep(self.createOutputStep, needsGPU=False)

    def _initialize(self):
        makePath(self._getTrainDataConfDir())
        self._configFile = join(self._getTrainDataConfDir(), TRAIN_DATA_CONFIG)
        self._configPath = self._getExtraPath('train_config.json')

    def prepareTrainingDataStep(self):
        if self.areEvenOddLinked.get():
            fnOdd, fnEven = self.getOddEvenLists()
        else:
            self._getListOfTomoNames(self.evenTomos.get())
            fnOdd = self._getListOfTomoNames(self.oddTomos.get())
            fnEven = self._getListOfTomoNames(self.evenTomos.get())

        config = {
            'even': fnEven,
            'odd': fnOdd,
            'patch_shape': 3 * [self.patch_shape.get()],
            'num_slices': self.num_slices.get(),
            'split': self.split.get(),
            'tilt_axis': self._decodeTiltAxisValue(self.tilt_axis.get()),
            'n_normalization_samples': self.n_normalization_samples.get(),
            'path': self._getExtraPath('train_data')
        }
        with open(self._configFile, 'w+') as f:
            json.dump(config, f, indent=2)

    def runDataExtraction(self):
        Plugin.runCryocare(self, 'cryoCARE_extract_train_data.py', '--conf %s' % self._configFile)

    def prepareTrainingStep(self):
        # We do this to accept both GPU specified as '0' 1 2 3' or '0,1,2,3':
        gpuId = getattr(self, params.GPU_LIST).getListFromValues()
        gpuId = gpuId[0] if len(gpuId) == 1 else gpuId
        config = {
            'train_data': self._getTrainDataDir(),
            'epochs': self.epochs.get(),
            'steps_per_epoch': self.steps_per_epoch.get(),
            'batch_size': self.batch_size.get(),
            'unet_kern_size': self.unet_kern_size.get(),
            'unet_n_depth': self._getUNetDepth(),
            'unet_n_first': self.unet_n_first.get(),
            'learning_rate': self.learning_rate.get(),
            'model_name': CRYOCARE_MODEL,
            'path': self._getExtraPath(),
            'gpu_id': gpuId
        }
        with open(self._configPath, 'w+') as f:
            json.dump(config, f, indent=2)

    def trainingStep(self):
        Plugin.runCryocare(self, 'cryoCARE_train.py', '--conf {}'.format(self._configPath))

    def createOutputStep(self):
        model = CryocareModel(model_file=getModelName(self),
                              train_data_dir=self._getTrainDataDir())
        self._defineOutputs(**{Outputobjects.model.name: model})

        if self.areEvenOddLinked.get():
            self._defineSourceRelation(self.tomos, model)
        else:
            self._defineSourceRelation(self.oddTomos, model)
            self._defineSourceRelation(self.evenTomos, model)

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        summary = []

        if self.isFinished():
            summary.append("Generated training data info:\n"
                           "train_data_file = *{}*\n"
                           "validation_data_file = *{}*\n"
                           "patch_size = *{}*".format(
                self._getTrainDataFile(),
                self._getValidationDataFile(),
                self.patch_shape.get()))
        return summary

    def _validate(self):
        sideLength = self.patch_shape.get()
        validateMsgs = super()._validate()
        if self.areEvenOddLinked.get():
            inputTomo = self.tomos.get()
            if self.tomos.get():
                try:
                    fnOdd, fnEven = self.getOddEvenLists()
                except Exception as e:
                    validateMsgs.append('Even/Odd tomograms seem no to be linked to the introduced tomograms '
                                        'at metadata level.')

                xt, yt, zt = inputTomo.getDimensions()
                for idim in [xt, yt, zt]:
                    if idim <= 2 * sideLength:
                        validateMsgs.append('X, Y and Z dimensions of the tomograms introduced must satisfy the '
                                            'condition\n\n*dimension > 2 x SideLength*\n\n'
                                            '(X, Y, Z) = (%i, %i, %i)\n'
                                            'SideLength = %i\n\n' % (xt, yt, zt, sideLength))
                        break
        else:
            evenTomos = self.evenTomos.get()
            oddTomos = self.oddTomos.get()
            xe, ye, ze = evenTomos.getDimensions()
            xo, yo, zo = oddTomos.getDimensions()
            for idim in [xe, ye, ze, xo, yo, zo]:
                if idim <= 2 * sideLength:
                    validateMsgs.append('X, Y and Z dimensions of the tomograms introduced must satisfy the '
                                        'condition\n\n*dimension > 2 x SideLength*\n\n'
                                        '(X, Y, Z) = (%i, %i, %i)\n'
                                        'SideLength = %i\n\n' % (xe, ye, ze, sideLength))
                    msg = checkInputTomoSetsSize(evenTomos, oddTomos)
                    if msg:
                        validateMsgs.append(msg)
                    break

        # Check the patch conditions
        if sideLength % 2 != 0:
            validateMsgs.append('Patch shape has to be an even number.')

        return validateMsgs

    # --------------------------- UTIL functions -----------------------------------
    @staticmethod
    def _combineTrainDataFiles(pattern, outputFile):
        files = glob.glob(pattern)
        if len(files) == 1:
            moveFile(files[0], outputFile)
        else:
            # Create a dictionary with the data fields contained in each npz file
            dataDict = {}
            with np.load(files[0]) as data:
                for field in data.files:
                    dataDict[field] = []

            # Read and combine the data from all files
            for i, name in enumerate(files):
                with np.load(name) as data:
                    for field in data.files:
                        dataDict[field].append(data[field])

            # Save the combined data into a npz file
            np.savez(outputFile, **dataDict)

    def _getTrainDataDir(self):
        return self._getExtraPath(TRAIN_DATA_DIR)

    def _getTrainDataFile(self):
        return join(self._getTrainDataDir(), TRAIN_DATA_FN)

    def _getValidationDataFile(self):
        return join(self._getTrainDataDir(), VALIDATION_DATA_FN)

    def _getTrainDataConfDir(self):
        return self._getExtraPath(TRAIN_DATA_CONFIG)

    @staticmethod
    def _decodeTiltAxisValue(value):
        if value == X_AXIS:
            return X_AXIS_LABEL
        elif value == Y_AXIS:
            return Y_AXIS_LABEL
        else:
            return Z_AXIS_LABEL

    @staticmethod
    def _getListOfTomoNames(tomoSet):
        return [tomo.getFileName() for tomo in tomoSet]

    def getOddEvenLists(self):
        oddList = []
        evenList = []
        for t in self.tomos.get():
            even, odd = sorted(t.getHalfMaps(asList=True))
            oddList.append(odd)
            evenList.append(even)
        return oddList, evenList

    def _getUNetDepth(self):
        # Estimate the best net depth value according to the patch size if the user left this field empty
        if self.unet_n_depth.get() == 0:
            refValues = [72, 96, 128]  # Corresponds to a net depth of 2, 3 and 4, respectively
            netDepth = [2, 3, 4]
            diff = [abs(i - self.patch_shape.get()) for i in refValues]
            ind, _ = min(enumerate(diff), key=operator.itemgetter(0))
            return netDepth[ind]
        else:
            return self.unet_n_depth.get()
