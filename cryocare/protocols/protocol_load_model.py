from enum import Enum
from os.path import exists, join

from cryocare.utils import makeDatasetSymLinks, getModelName
from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.protocol import PathParam, FileParam
from pyworkflow.utils import Message, createLink

from cryocare.constants import TRAIN_DATA_FN, VALIDATION_DATA_FN
from cryocare.objects import CryocareModel


class Outputobjects(Enum):
    model = CryocareModel


class ProtCryoCARELoadModel(EMProtocol):
    """
    Loads a previously trained cryoCARE neural network model together
    with its associated training data so that it can be reused for
    tomogram restoration and denoising workflows.

    AI Generated:

    CryoCARE Load Trained Model (ProtCryoCARELoadModel) — User Manual
        Overview

        The CryoCARE Load Trained Model protocol imports an existing
        cryoCARE model into the processing environment so it can be used
        for tomogram denoising and restoration tasks. Its primary purpose
        is to make previously trained neural networks reusable across
        projects, datasets, or collaborative workflows without requiring
        retraining from scratch.

        In cryo-electron tomography, training a deep learning model can
        require substantial computational resources and carefully prepared
        datasets. Once a high-quality model has been generated, researchers
        often wish to apply it repeatedly to similar tomographic data. This
        protocol provides a structured mechanism for registering and
        validating those trained models for later prediction workflows.

        Inputs and General Workflow

        The protocol requires two main inputs: a trained cryoCARE model and
        the associated training data prepared during the original learning
        process. The model contains the neural network parameters and
        optimization results, while the training data provides the
        normalization and dataset context expected by downstream cryoCARE
        prediction workflows.

        The imported model is expected to contain all files necessary for
        inference and restoration. These typically include configuration
        parameters, normalization information, training history, and neural
        network weight files. Maintaining the integrity of these components
        is important because incomplete or inconsistent model packages may
        prevent successful tomogram restoration.

        Biological and Practical Context

        From a biological perspective, reusing a trained model is most
        effective when the target tomograms resemble the data used during
        training. Similar acquisition conditions, voxel sizes, contrast
        characteristics, reconstruction methods, and specimen types
        generally improve denoising quality and preserve biologically
        meaningful structures.

        Applying a model trained on substantially different datasets may
        reduce restoration quality or introduce artifacts that complicate
        interpretation. For this reason, users should evaluate restored
        tomograms carefully whenever a model is transferred between
        unrelated experimental conditions.

        Validation and Data Consistency

        The protocol performs consistency checks to ensure that the
        introduced model and associated training datasets are complete and
        compatible. Proper validation is particularly important because
        cryoCARE prediction workflows depend not only on the trained neural
        network but also on the normalization and dataset information
        generated during training preparation.

        Ensuring that all required training and validation datasets are
        present helps maintain reproducibility and minimizes the risk of
        incompatible prediction results. This is especially valuable in
        collaborative environments where trained models may be exchanged
        between users or institutions.

        Outputs and Their Interpretation

        After execution, the protocol generates a reusable cryoCARE model
        object that can be directly connected to prediction and denoising
        protocols. The resulting model serves as a portable representation
        of the trained neural network together with its associated metadata
        and training context.

        Biologically, the loaded model does not alter any tomographic data
        by itself. Instead, it prepares the trained network for subsequent
        restoration workflows in which noisy tomograms can be processed and
        denoised.

        Practical Recommendations

        Users should ensure that imported models originate from reliable
        training workflows and that all required files remain intact.
        Storing models together with their associated training datasets is
        strongly recommended for long-term reproducibility and portability.

        Before applying a loaded model to large experimental datasets, it
        is advisable to validate its behavior on a small subset of
        tomograms. This allows users to confirm that the restored contrast,
        structural continuity, and denoising characteristics remain
        biologically meaningful.

        Final Perspective

        The CryoCARE Load Trained Model protocol provides a practical and
        reproducible mechanism for reusing previously trained denoising
        networks in cryo-electron tomography workflows. By enabling trained
        models to be shared and reapplied efficiently, it supports scalable
        and consistent tomogram restoration across multiple experiments and
        research projects.
    """

    _label = 'CryoCARE Load Trained Model'
    _devStatus = BETA
    _possibleOutputs = Outputobjects

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        # You need a params to belong to a section:
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('trainDataModel', PathParam,
                      label='Pre-trained cryoCARE model (.tar.gz)',
                      important=True,
                      allowsNull=False,
                      help='It is a .tar.gz file containing a folder that contains, in turn, the following files:\n\n'
                           '\t- config.json\n'
                           '\t- history.dat\n'
                           '\t- norm.json\n'
                           '\t- weights_best.h5\n'
                           '\t- weights_last.h5\n')
        form.addParam('trainDataDir', FileParam,
                      label='Directory of the prepared data for training',
                      important=True,
                      allowsNull=False,
                      help='It must contain two files: train_data.npz and val_data.npz, generated when '
                           'preparing the training data.')

    def _insertAllSteps(self):
        self._initialize()
        self._insertFunctionStep(self.createOutputStep, needsGPU=False)

    def _initialize(self):
        # The prediction is expecting the training and validation datasets to be in the same place as the training
        # model, but they are located in the training data generation extra directory. Hence, a symbolic link will
        # be created
        makeDatasetSymLinks(self, self.trainDataDir.get())
        createLink(join(self.trainDataModel.get()), getModelName(self))

    def createOutputStep(self):
        model = CryocareModel(model_file=getModelName(self), train_data_dir=self._getExtraPath())
        self._defineOutputs(**{Outputobjects.model.name: model})

    # --------------------------- INFO functions -----------------------------------
    def _validate(self):
        errors = []
        if not exists(self.trainDataModel.get()):
            errors.append('Training model introduced does not exists.')

        if not exists(self.trainDataDir.get()):
            errors.append('Directory of the prepared data for training does not exists.')
        else:
            if not exists(join(self.trainDataDir.get(), TRAIN_DATA_FN)):
                errors.append('No %s file was found in the introduced training model base directory.' 
                              % TRAIN_DATA_FN)
            if not exists(join(self.trainDataDir.get(), VALIDATION_DATA_FN)):
                errors.append('No %s file was found in the introduced training model base directory.'
                              % VALIDATION_DATA_FN)
        return errors

    def _summary(self):
        summary = []

        if self.isFinished():
            summary.append("Loaded training model_dir = *%s*" % self.trainDataModel.get())
        return summary

