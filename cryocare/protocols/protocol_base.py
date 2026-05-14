from typing import Union

from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.object import Pointer
from pyworkflow.protocol import params
from pyworkflow.utils import Message
from tomo.objects import SetOfTomograms

# Inputs
IN_TOMOS = 'tomos'
IN_EVEN_TOMOS = 'evenTomos'
IN_ODD_TOMOS = 'oddTomos'

class ProtCryoCAREBase(EMProtocol):
    """
    Provides the common infrastructure required for cryoCARE-based tomogram
    restoration workflows. The protocol establishes how paired even and odd
    tomographic reconstructions are introduced into the processing pipeline so
    they can later be used for neural network training or denoising tasks.

    AI Generated:

    CryoCARE Base Protocol (ProtCryoCAREBase) — User Manual
        Overview

        The CryoCARE Base protocol defines the fundamental organization of
        tomographic data required for cryoCARE workflows within Scipion. Its
        primary purpose is to standardize how even and odd tomographic
        reconstructions are provided so that downstream protocols can perform
        self-supervised denoising and restoration in a biologically meaningful
        manner.

        In cryo-electron tomography, splitting the original projection data
        into even and odd subsets allows the generation of two independent
        reconstructions containing similar structural information but different
        noise realizations. This separation forms the basis of cryoCARE
        training and prediction strategies, where the neural network learns to
        distinguish reproducible signal from stochastic noise without requiring
        external ground truth data.

        Inputs and Data Organization

        The protocol supports two biologically relevant ways of organizing
        tomographic inputs. In the first scenario, the even and odd
        reconstructions are already associated internally with each tomogram.
        This is the most convenient and recommended organization because the
        relationship between paired reconstructions is preserved automatically.

        In the second scenario, the even and odd reconstructions are provided
        as two independent sets of tomograms. This alternative is useful when
        the datasets were generated externally or imported from different
        processing environments. In such cases, users must ensure that both
        sets correspond exactly to the same biological specimens and acquisition
        conditions.

        Biological Importance of Even and Odd Reconstructions

        The distinction between even and odd tomograms is central to cryoCARE
        methodology. Since both reconstructions originate from complementary
        subsets of the same experimental data, they contain correlated
        structural signal while preserving statistically independent noise.
        This property enables neural networks to learn denoising patterns
        without artificially smoothing biologically relevant features.

        For biological interpretation, it is essential that both tomograms
        represent the same specimen geometry, voxel sampling, and reconstruction
        conditions. Misaligned or mismatched inputs may lead to unstable
        denoising behavior or loss of structural fidelity.

        Validation and Dataset Consistency

        The protocol emphasizes validation of the introduced datasets before
        downstream processing begins. Correct association between even and odd
        tomograms is critical because cryoCARE assumes that corresponding
        structures occupy the same spatial frame. If unrelated datasets are
        combined, the resulting denoising process may amplify artifacts or
        suppress meaningful biological information.

        Users should therefore verify that tomograms originate from the same
        acquisition session and reconstruction workflow. Consistency in voxel
        size, image dimensions, and coordinate orientation is particularly
        important when the even and odd datasets are introduced separately.

        Workflow Integration

        This protocol is intended to act as the shared foundation for cryoCARE
        training and prediction procedures. By standardizing how tomograms are
        accessed and interpreted, it enables downstream protocols to focus on
        neural network optimization and restoration while maintaining a
        consistent biological interpretation of the data.

        In practical cryo-electron tomography workflows, this design simplifies
        interoperability between training, prediction, and model management
        tasks. It also reduces the likelihood of introducing inconsistencies
        between datasets generated in different software environments.

        Practical Recommendations

        For most users, the preferred strategy is to work with tomograms that
        already contain linked even and odd reconstructions. This minimizes
        organizational errors and simplifies workflow management. When separate
        datasets must be used, careful verification of correspondence between
        the two sets is strongly recommended before continuing with denoising
        procedures.

        Users should also ensure that preprocessing steps such as binning,
        alignment, and reconstruction were performed consistently for both
        halves of the dataset. Even subtle differences may affect neural
        network performance and compromise the interpretability of restored
        tomograms.

        Final Perspective

        The CryoCARE Base protocol provides the structural foundation required
        for reliable cryoCARE denoising workflows. Although it does not perform
        restoration directly, it defines the biological and organizational
        consistency necessary for self-supervised learning approaches to work
        correctly. Careful preparation and validation of even and odd
        tomographic datasets remain essential for producing trustworthy
        denoised reconstructions suitable for downstream structural analysis.
    """
    _devStatus = BETA

    # -------------------------- DEFINE param functions ----------------------

    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        # You need a params to belong to a section:
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('areEvenOddLinked', params.BooleanParam,
                      default=True,
                      label="Are odd-even associated to the Tomograms?")
        form.addParam(IN_EVEN_TOMOS, params.PointerParam,
                      pointerClass='SetOfTomograms',
                      condition='not areEvenOddLinked',
                      label='Even tomograms',
                      allowsNull=True,
                      important=True,
                      help='Set of tomograms reconstructed from the even frames of the tilt'
                           'series movies.')
        form.addParam(IN_ODD_TOMOS, params.PointerParam,
                      pointerClass='SetOfTomograms',
                      condition='not areEvenOddLinked',
                      label='Odd tomograms',
                      allowsNull=True,
                      important=True,
                      help='Set of tomogram reconstructed from the odd frames of the tilt'
                           'series movies.')
        form.addParam(IN_TOMOS, params.PointerParam,
                      pointerClass='SetOfTomograms',
                      condition='areEvenOddLinked',
                      label='Tomograms',
                      allowsNull=True,
                      important=True)

    def _validate(self):
        # As the input tomograms parameter change based on a condition, all of them must allow empty values at the
        # form level. Thus, the tomograms introduced needs to be validated here
        errorMsg = []
        if self.areEvenOddLinked.get():
            if not self.tomos.get():
                errorMsg.append('If the parameter "Are odd-even associated to the Tomograms?" was set to Yes, a set '
                                'of tomograms with the even/odd sets associated to its metadata must be introduced.')

        else:
            if not self.evenTomos.get() or not self.oddTomos.get():
                errorMsg.append('If the parameter "Are odd-even associated to the Tomograms?" was set to No, a set '
                                'of even tomograms and a set of odd tomograms must be introduced.')
        return errorMsg

    # --------------------------- UTIL functions -----------------------------------
    def getInTomos(self,
                   even: Union[None, bool] = None,
                   asPointer: bool = True) -> Union[Pointer, SetOfTomograms]:
        if even is None:
            attribName = IN_TOMOS
        else:
            if even:
                attribName = IN_EVEN_TOMOS
            else:
                attribName = IN_ODD_TOMOS
        resPointer = getattr(self, attribName)
        return resPointer if asPointer else resPointer.get()