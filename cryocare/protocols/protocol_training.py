# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     you (you@yourinstitution.email)
# *
# * your institution
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
# *  e-mail address 'you@yourinstitution.email'
# *
# **************************************************************************


"""
Describe your python module here:
This module will provide the traditional Hello world example
"""
from pwem.protocols import EMProtocol
from pyworkflow.protocol import params, Integer
from pyworkflow.utils import Message


class ProtCryocareTraining(EMProtocol):
    """Use two data-independent reconstructed tomograms to train a 3D cryo-CARE network."""

    _label = 'cryocare training'

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        # You need a params to belong to a section:
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('evenTomo', params.PointerParam,
                      pointerClass='Tomogram',
                      label='Tomogram (from even frames)',
                      important=True,
                      help='Tomogram reconstructed from the even frames of the tilt'
                           'series movies.')

        form.addParam('oddTomo', params.PointerParam,
                      pointerClass='Tomogram',
                      label='Tomogram (from odd frames)',
                      important=True,
                      help='Tomogram reconstructed from the odd frames of the tilt'
                           'series movies.')

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        # Insert processing steps
        self._insertFunctionStep('genTrainDataStep')
        self._insertFunctionStep('trainStep')
        self._insertFunctionStep('createOutputStep')

    def genTrainDataStep(self):
        pass

    def trainStep(self):
        pass

    def createOutputStep(self):
        pass

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        """ Summarize what the protocol has done"""
        summary = []

        if self.isFinished():
            summary.append("Some message to summarize.")
        return summary

    def _methods(self):
        return []
