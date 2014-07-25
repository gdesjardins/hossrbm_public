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

import numpy
from pylearn2.datasets import dense_design_matrix

kernels = numpy.asarray(
        [[[1,1,1], [1,0,1], [1,1,1]],
         [[1,0,1], [0,1,0], [1,0,1]],
         [[0,1,0], [1,1,1], [0,1,0]],
         [[1,1,1], [0,0,0], [1,1,1]],
         [[1,0,1], [1,0,1], [1,0,1]]],
        dtype='float32')

class NoisyColorToyData(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, center=True, noise_std=0.1, seed=10984721):

        self.rng = numpy.random.RandomState(seed)

        n_cols = 3 # colors
        m_pos = 5 # positions
        width  = len(kernels[0][0])
        height = len(kernels[0])
        cols   = (width + 1)*m_pos
        rows   = height

        X = numpy.zeros(((2**m_pos)*(2**n_cols), n_cols*rows*cols),dtype='float32')
        noisyX = numpy.zeros((1000*X.shape[0], X.shape[1]), dtype='float32')

        idx = 0
        # generate a configuration of (g,h)
        for h in numpy.ndindex(*([2]*m_pos)):
            for g in numpy.ndindex(*([2]*n_cols)):
                example = numpy.zeros((n_cols,rows,cols),dtype='float32')
                for i in xrange(n_cols):
                    for j in xrange(m_pos):
                        c = j*(width+1)
                        example[i, :, c:c+width] += g[i] * h[j] * kernels[j]

                scale = self.rng.normal(1.0, 0.1)
                sign = self.rng.randint(0,2.)*2 - 1.
                X[idx,:] = sign * scale * example.reshape(rows * cols * n_cols)

                idx += 1

        for j in xrange(0, len(noisyX), len(X)):
            noise = noise_std * self.rng.randn(X.shape[0], X.shape[1])
            noisyX[j:j+len(X)] = X + noise

        view_converter = dense_design_matrix.DefaultViewConverter((rows,cols,1))

        if center:
            noisyX -= numpy.mean(noisyX, axis=0)

        super(NoisyColorToyData,self).__init__(X = noisyX, view_converter = view_converter)

        assert not numpy.any(numpy.isnan(self.X))

kernels2 = numpy.asarray(
            [[[0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0]],
             [[1,0,0,0,0,0,1],
              [0,1,0,0,0,1,0],
              [0,0,1,0,1,0,0],
              [0,0,0,1,0,0,0],
              [0,0,1,0,1,0,0],
              [0,1,0,0,0,1,0],
              [1,0,0,0,0,0,1]],
             [[1,1,1,1,1,1,1],
              [1,0,0,0,0,0,1],
              [1,0,0,0,0,0,1],
              [1,0,0,0,0,0,1],
              [1,0,0,0,0,0,1],
              [1,0,0,0,0,0,1],
              [1,1,1,1,1,1,1]],
             [[0,0,0,1,0,0,0],
              [0,0,0,1,0,0,0],
              [0,0,0,1,0,0,0],
              [1,1,1,1,1,1,1],
              [0,0,0,1,0,0,0],
              [0,0,0,1,0,0,0],
              [0,0,0,1,0,0,0]]])

class SuperImposedShapes(dense_design_matrix.DenseDesignMatrix):

    def __init__(self):

        n = 3 # colors
        m = len(kernels2)
        (h, w) = kernels2[0].shape
        X = numpy.zeros(( (2**m)*(2**n), w*h*3),dtype='float32')
        idx = 0
        for i in numpy.ndindex(*([2]*m)):
            for j in numpy.ndindex(*([2]*n)):

                example = numpy.zeros((n,h,w), dtype='float32')
                for k in xrange(m):
                    if i[k]:
                        example[0,:,:] += j[0] * kernels2[k]
                        example[1,:,:] += j[1] * kernels2[k]
                        example[2,:,:] += j[2] * kernels2[k]

                X[idx,:] = example.reshape(h * w * n)
                idx += 1

        view_converter = dense_design_matrix.DefaultViewConverter((h,w,1))

        super(SuperImposedShapes,self).__init__(X = X, view_converter = view_converter)

        assert not numpy.any(numpy.isnan(self.X))


