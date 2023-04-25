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

CRYOCARE_HOME = 'CRYOCARE_HOME'
V0_2_2 = '0.2.2'
CRYOCARE_DEFAULT_VERSION = V0_2_2
CRYOCARE = 'cryoCARE'
CRYOCARE_ENV_NAME = '%s-%s' % (CRYOCARE, CRYOCARE_DEFAULT_VERSION)
CRYOCARE_ENV_ACTIVATION = 'CRYOCARE_ENV_ACTIVATION'
DEFAULT_ACTIVATION_CMD = 'conda activate %s' % CRYOCARE_ENV_NAME
CRYOCARE_CUDA_LIB = 'CRYOCARE_CUDA_LIB'

TRAIN_DATA_DIR = 'train_data'
TRAIN_DATA_FN = 'train_data.npz'
VALIDATION_DATA_FN = 'val_data.npz'
MEAN_STD_FN = 'mean_std.npz'
TRAIN_DATA_CONFIG = 'training_data_config'
CRYOCARE_MODEL = 'cryoCARE_model'
CRYOCARE_MODEL_TGZ = CRYOCARE_MODEL + '.tar.gz'
PREDICT_CONFIG = 'predict_config'
