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
import theano

def sparsity_mask(type, **kwargs):
    assert hasattr(SparsityMask, type)
    const = getattr(SparsityMask, type)
    return const(**kwargs)

class SparsityMask(object):

    def __init__(self, mask, **kwargs):
        self.mask = numpy.asarray(mask, dtype=theano.config.floatX)
        for (k,v) in kwargs.iteritems():
            setattr(self,k,v)

    @classmethod
    def unfactored_g(cls, n_g, n_h, bw_g, bw_h):
        """
        Creates a sparsity mask for g-units, equivalent to an unfactored model.
        :param n_g: number of g-units
        :param n_h: number of h-units
        :param bw_g: block width in g
        :param bw_h: block width in h
        """
        assert (n_g % bw_g) == 0 and (n_h % bw_h) == 0 and (n_g / bw_g == n_h / bw_h)
        n_s = (n_g / bw_g) * (bw_g * bw_h)

        # init Wg
        si = 0
        mask = numpy.zeros((n_s, n_g))
        for gi in xrange(n_g):
            mask[si:si+bw_h, gi] = 1.
            si += bw_h

        return SparsityMask(mask.T, n_g=n_g, n_h=n_h, bw_g=bw_g, bw_h=bw_h)

    @classmethod
    def unfactored_h(cls, n_g, n_h, bw_g, bw_h):
        """
        Creates a sparsity mask for h-units, equivalent to an unfactored model.
        :param n_g: number of g-units
        :param n_h: number of h-units
        :param bw_g: block width in g
        :param bw_h: block width in h
        """
        assert (n_g % bw_g) == 0 and (n_h % bw_h) == 0 and (n_g / bw_g == n_h / bw_h)
        n_s = (n_g / bw_g) * (bw_g * bw_h)

        # init Wh
        si = 0
        ds = bw_g * bw_h
        mask = numpy.zeros((n_s, n_h))

        for hi in xrange(n_h):
            bi = hi / bw_h
            mask[bi*ds + hi%bw_h:(bi+1)*ds:bw_h, hi] = 1.

        return SparsityMask(mask.T, n_g=n_g, n_h=n_h, bw_g=bw_g, bw_h=bw_h)
        
    
    """
    Methods to implement a diagonal sparsity_mask
    """
        
    @classmethod
    def diagonal_g(cls, n_g, n_h, width, delta):
        """
        Creates a sparsity mask for g-units, equivalent to an unfactored model.
        :param n_g: number of g-units
        :param n_h: number of h-units
        :param width: diagonal width in the connectivity matrix
        :param delta: how much the diagonal pattern should increase in h for
                      each unit increase in g
        """
        
        # Create the gh connectivity matrix
        gh_conn = numpy.zeros((n_g, n_h))
        si = 1
        for gi in range(n_g):
            for hi in range(n_h):
                h_start = gi * delta - width / 2
                h_end = h_start + width
                if hi >= h_start and hi < h_end:
                    # This gi should be connected with hi via the unit si
                    gh_conn[gi,hi] = si
                    si += 1
        
        # Build the binary mask
        n_s = si - 1
        mask = numpy.zeros((n_s, n_g))
        
        for gi in range(n_g):
            for hi in range(n_h):
                if gh_conn[gi,hi] >= 1:
                    mask[gh_conn[gi,hi]-1,gi] = 1.

        return SparsityMask(mask.T, n_g=n_g, n_h=n_h, width=width, delta=delta)
        
        

    @classmethod
    def diagonal_h(cls, n_g, n_h, width, delta):
        """
        Creates a sparsity mask for g-units, equivalent to an unfactored model.
        :param n_g: number of g-units
        :param n_h: number of h-units
        :param width: diagonal width in the connectivity matrix
        :param delta: how much the diagonal pattern should increase in h for
                      each unit increase in g
        """
        
        # Create the gh connectivity matrix
        gh_conn = numpy.zeros((n_g, n_h))
        si = 1
        for gi in range(n_g):
            for hi in range(n_h):
                h_start = gi * delta - width / 2
                h_end = h_start + width
                if hi >= h_start and hi < h_end:
                    # This gi should be connected with hi via the unit si
                    gh_conn[gi,hi] = si
                    si += 1       
        
        # Build the binary mask
        n_s = si - 1
        mask = numpy.zeros((n_s, n_g))
        
        for gi in range(n_g):
            for hi in range(n_h):
                if gh_conn[gi,hi] >= 1:
                    mask[gh_conn[gi,hi]-1,hi] = 1.

        return SparsityMask(mask.T, n_g=n_g, n_h=n_h, width=width, delta=delta)
