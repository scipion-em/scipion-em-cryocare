from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.protocol import params
from pyworkflow.utils import Message


class ProtCryoCAREBase(EMProtocol):
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
                      default=False,
                      label="Are odd-even associated to the Tomograms?")
        form.addParam('evenTomos', params.PointerParam,
                      pointerClass='SetOfTomograms',
                      condition='not areEvenOddLinked',
                      label='Even tomograms',
                      allowsNull=True,
                      important=True,
                      help='Set of tomograms reconstructed from the even frames of the tilt'
                           'series movies.')
        form.addParam('oddTomos', params.PointerParam,
                      pointerClass='SetOfTomograms',
                      condition='not areEvenOddLinked',
                      label='Odd tomograms',
                      allowsNull=True,
                      important=True,
                      help='Set of tomogram reconstructed from the odd frames of the tilt'
                           'series movies.')
        form.addParam('tomo', params.PointerParam,
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
            if not self.tomo.get():
                errorMsg.append('If the parameter "Are odd-even associated to the Tomograms?" was set to Yes, a set '
                                'of tomograms with the even/odd sets associated to its metadata must be introduced.')

        else:
            if not self.evenTomos.get() or not self.oddTomos.get():
                errorMsg.append('If the parameter "Are odd-even associated to the Tomograms?" was set to No, a set '
                                'of even tomograms and a set of odd tomograms must be introduced.')
        return errorMsg
