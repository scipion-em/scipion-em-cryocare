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
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
from pwem.emlib.image import ImageHandler


class CryocareUtils:
    @staticmethod
    def checkInputTomoSetsSize(evenTomoSet, oddTomoSet):
        message = ''
        xe, ye, ze = evenTomoSet.getDimensions()
        xo, yo, zo = oddTomoSet.getDimensions()
        ne = evenTomoSet.getSize()
        no = oddTomoSet.getSize()
        if (xe, ye, ze, ne) != (xo, yo, zo, no):
            message = ('Size of even and odd set of tomograms must be the same:\n'
                       'Even --> (x, y, z, n) = ({xe}, {ye}, {ze}, {ne})\n'
                       'Odd  --> (x, y, z, n) = ({xo}, {yo}, {zo}, {no})'.format(
                        xe=xe, ye=ye, ze=ze, ne=ne, xo=xo, yo=yo, zo=zo, no=no))

        return message
