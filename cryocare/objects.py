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
from os.path import join

import pyworkflow.object as pwobj
from pwem import EMObject


class CryocareTrainData(EMObject):
    def __init__(self, train_data=None, mean_std=None, **kwargs):
        EMObject.__init__(self, **kwargs)
        self._train_data = pwobj.String(train_data)
        self._mean_std = pwobj.String(mean_std)

    def getTrainData(self):
        return self._train_data.get()

    def getMeanStd(self):
        return self._mean_std.get()

    def getPath(self):
        return '{}\n{}'.format(self.getTrainData(), self.getMeanStd())

    def __str__(self):
        return "CryoCARE Train Data (path=%s)" % self.getPath()


class CryocareModel(EMObject):
    def __init__(self, basedir=None, model_name=None, **kwargs):
        EMObject.__init__(self, **kwargs)
        self._basedir = pwobj.String(basedir)
        self._model_name = pwobj.String(model_name)

    def getPath(self):
        return join(self._basedir.get(), self._model_name.get())

    def __str__(self):
        return "CryoCARE Model (path=%s)" % self.getPath()