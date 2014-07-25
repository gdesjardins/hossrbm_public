# Copyright (c) 2013, Guillaume Desjardins.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the <organization> nor the
#     names of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#!/opt/lisa/os/epd-7.1.2/bin/python
import sys
import copy
import numpy
import pylab as pl
import pickle
import os
from optparse import OptionParser

from theano import function
import theano.tensor as T
import theano

from pylearn2.gui.patch_viewer import make_viewer
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial

parser = OptionParser()
parser.add_option('-m', '--model', action='store', type='string', dest='path')
parser.add_option('--width',  action='store', type='int', dest='width')
parser.add_option('--height', action='store', type='int', dest='height')
parser.add_option('--channels',  action='store', type='int', dest='chans')
parser.add_option('--color', action='store_true',  dest='color', default=False)
parser.add_option('--top', action='store', type='int', dest='top', default=5)
parser.add_option('--wv_only', action='store_true', dest='wv_only', default=False)
(opts, args) = parser.parse_args()

nplots = opts.chans
if opts.color:
    assert opts.chans == 3
    nplots = 1

def get_dims(nf):
    num_rows = numpy.floor(numpy.sqrt(nf))
    return (int(num_rows), int(numpy.ceil(nf / num_rows)))

topo_shape = [opts.height, opts.width, opts.chans]
viewconv = DefaultViewConverter(topo_shape)
viewdims = slice(0, None) if opts.color else 0

# load model and retrieve parameters
model = serial.load(opts.path)
wv = model.Wv.get_value().T

wv_viewer = PatchViewer(get_dims(len(wv)), (opts.height, opts.width),
                        is_color = opts.color, pad=(2,2))
for i in xrange(len(wv)):
    topo_wvi = viewconv.design_mat_to_topo_view(wv[i:i+1])
    wv_viewer.add_patch(topo_wvi[0])
if opts.wv_only:
    wv_viewer.show()
    os.sys.exit()

wg = model.Wg.get_value()
wh = model.Wh.get_value()
wg_viewer2 = PatchViewer((opts.top, opts.top), (opts.height, opts.width),
                         is_color = opts.color, pad=(2,2))
wg_viewer1 = PatchViewer(get_dims(len(wg)/opts.top),
                         (wg_viewer2.image.shape[0], wg_viewer2.image.shape[1]),
                         is_color = opts.color, pad=(2,2))
for i in xrange(0, len(wg), opts.top):
    for j in xrange(i, i + opts.top):
        idx = numpy.argsort(wg[j])[-opts.top:][::-1]
        for idx_j in idx:
            topo_wgi = viewconv.design_mat_to_topo_view(wv[idx_j:idx_j+1])
            wg_viewer2.add_patch(topo_wgi[0])
    wg_viewer1.add_patch(wg_viewer2.image[:,:,viewdims])

wh_viewer2 = PatchViewer((opts.top, opts.top), (opts.height, opts.width),
                         is_color = opts.color, pad=(2,2))
wh_viewer1 = PatchViewer(get_dims(len(wh)/opts.top),
                         (wh_viewer2.image.shape[0], wh_viewer2.image.shape[1]),
                         is_color = opts.color, pad=(2,2))

for i in xrange(0, len(wh), opts.top):
    for j in xrange(i, i + opts.top):
        idx = numpy.argsort(wh[j])[-opts.top:][::-1]
        for idx_j in idx:
            topo_whi = viewconv.design_mat_to_topo_view(wv[idx_j:idx_j+1])
            wh_viewer2.add_patch(topo_whi[0])
    wh_viewer1.add_patch(wh_viewer2.image[:,:,viewdims])

pl.subplot(1,3,1); pl.imshow(wv_viewer.image[:,:,viewdims]); pl.gray(); pl.axis('off')
pl.subplot(1,3,2); pl.imshow(wg_viewer1.image[:,:,viewdims]); pl.gray(); pl.axis('off')
pl.subplot(1,3,3); pl.imshow(wh_viewer1.image[:,:,viewdims]); pl.gray(); pl.axis('off')
pl.show()
