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

import pwem
import os
from pyworkflow.utils import Environ
from cryocare.constants import CRYOCARE_ENV_ACTIVATION, DEFAULT_ACTIVATION_CMD, CRYOCARE_ENV_NAME, \
    CRYOCARE_DEFAULT_VERSION, CRYOCARE_HOME, CRYOCARE_CUDA_LIB, CRYOCARE

_logo = "icon.png"
_references = ['buchholz2019cryo']
__version__ = "4.0.0"


class Plugin(pwem.Plugin):
    _homeVar = CRYOCARE_HOME
    _url = 'https://github.com/scipion-em/scipion-em-cryocare'

    @classmethod
    def _defineVariables(cls):
        # cryoCARE does NOT need EmVar because it uses a conda environment.
        cls._defineVar(CRYOCARE_ENV_ACTIVATION, DEFAULT_ACTIVATION_CMD)
        cls._defineVar(CRYOCARE_CUDA_LIB, pwem.Config.CUDA_LIB)

    @classmethod
    def getCryocareEnvActivation(cls):
        return cls.getVar(CRYOCARE_ENV_ACTIVATION)

    @classmethod
    def getEnviron(cls, gpuId='0'):
        """ Setup the environment variables needed to launch cryocare. """
        environ = Environ(os.environ)
        if 'PYTHONPATH' in environ:
            # this is required for python virtual env to work
            del environ['PYTHONPATH']

        environ.update({'CUDA_VISIBLE_DEVICES': gpuId})

        cudaLib = environ.get(CRYOCARE_CUDA_LIB, pwem.Config.CUDA_LIB)
        environ.addLibrary(cudaLib)
        return environ

    @classmethod
    def defineBinaries(cls, env):
        CRYOCARE_INSTALLED = '%s_%s_installed' % (CRYOCARE, CRYOCARE_DEFAULT_VERSION)
        installationCmd = cls.getCondaActivationCmd()
        # try to get CONDA activation command
        # Create the environment
        installationCmd += 'conda create -y -n %s python=3.8 cudatoolkit=11.2 cudnn=8.1 -c conda-forge && ' % CRYOCARE_ENV_NAME
        # 'keras-gpu=2.3.1 ' \

        # Activate new the environment
        installationCmd += 'conda activate %s && ' % CRYOCARE_ENV_NAME

        # Install the rest of dependencies

        installationCmd += 'pip install tensorflow==2.5.0 && '
        installationCmd += 'pip install matplotlib==3.6.3 && '

        # Install cryoCARE
        installationCmd += 'pip install %s==%s && ' % (CRYOCARE, CRYOCARE_DEFAULT_VERSION)
        
        # Flag installation finished
        installationCmd += 'touch %s' % CRYOCARE_INSTALLED

        cryocare_commands = [(installationCmd, CRYOCARE_INSTALLED)]

        envPath = os.environ.get('PATH', "")  # keep path since conda likely in there
        installEnvVars = {'PATH': envPath} if envPath else None

        env.addPackage(CRYOCARE,
                       version=CRYOCARE_DEFAULT_VERSION,
                       tar='void.tgz',
                       commands=cryocare_commands,
                       neededProgs=cls.getDependencies(),
                       vars=installEnvVars,
                       default=bool(cls.getCondaActivationCmd()))

    @classmethod
    def getDependencies(cls):
        # try to get CONDA activation command
        condaActivationCmd = cls.getCondaActivationCmd()
        neededProgs = []
        if not condaActivationCmd:
            neededProgs.append('conda')

        return neededProgs

    @classmethod
    def runCryocare(cls, protocol, program, args, cwd=None, gpuId='0'):
        """ Run cryoCARE command from a given protocol. """
        fullProgram = '%s %s && %s' % (cls.getCondaActivationCmd(),
                                       cls.getCryocareEnvActivation(),
                                       program)
        protocol.runJob(fullProgram, args, env=cls.getEnviron(gpuId=gpuId), cwd=cwd, numberOfMpi=1)
