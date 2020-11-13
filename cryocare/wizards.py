from pyworkflow.gui import ListTreeProviderString, dialog
from pyworkflow.object import Integer
from pyworkflow.wizard import Wizard

from cryocare.protocols import ProtCryoCAREPrepareTrainingData

PATCH_SHAPE = 'patch_shape'


class cryoCAREPatchSizeWizard(Wizard):
    # Dictionary to target protocol parameters
    _targets = [(ProtCryoCAREPrepareTrainingData, [PATCH_SHAPE])]

    def show(self, form, *params):
        patchValues = [Integer(i) for i in range(32, 130, 8)]

        # Get a data provider from the patchValues to be used in the tree (dialog)
        provider = ListTreeProviderString(patchValues)

        dlg = dialog.ListDialog(form.root, "Paych shape values", provider,
                                "Select one of the size values)")

        # Set the chosen value back to the form
        form.setVar(PATCH_SHAPE, dlg.values[0].get())
