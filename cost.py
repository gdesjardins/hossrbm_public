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
from collections import OrderedDict

import theano.tensor as T
from utils import sharedX, floatX
from theano.printing import Print

class Cost():

    def __init__(self, cost, params, constants=None):
        self.cost = cost
        self.grads = OrderedDict()
        self.computed_cost = False

        self.params = OrderedDict()
        for p in params:
            self.params[p] = True

        self.constants = OrderedDict()
        constants = [] if constants is None else constants
        for c in constants:
            self.constants[c] = True

    def compute_gradients(self, lr, multipliers=None):
        multipliers = OrderedDict() if multipliers is None else multipliers
        grads =  T.grad(self.cost, self.params.keys(), 
                        consider_constant=self.constants.keys(),
                        disconnected_inputs='ignore')
        for param, gparam in zip(self.params.keys(), grads):
            param_lr = multipliers.get(param.name, 1.0) * lr
            self.grads[param] = param_lr * gparam
        self.computed_cost = True

    def update_gradient(self, param, new_grad):
        assert self.computed_cost
        assert self.grads.has_key(param)
        self.grads[param] = new_grad


from theano.printing import Print
def compute_gradients(lr, multipliers=None, *costs):
    """
    :param lr: base, scalar value for learning rate.
    :param multipliers: dictionary contains for each
     param (key) a learning rate multiplier (value).
    :param costs: (variable) list of Cost objects.
    """
    rval = OrderedDict()
    for cost in costs:
        if not cost.computed_cost:
            cost.compute_gradients(lr, multipliers)

        for (p,g) in cost.grads.iteritems():
            rval[p] = rval.get(p, 0.) + g

    #for (k,v) in rval.iteritems():
        #rval[k] = Print('%s_grad'%k.name)(v)
    return rval


def get_updates(grads, momentum_lambda=None):
    """
    Returns an updates dictionary corresponding to a single step of SGD. The learning rate
    for each parameter is computed as lr * multipliers[param]
    :param lr: base learning rate (common to all parameters)
    :param multipliers: dictionary of learning rate multipliers, each being a shared var
                        e.g. {'hbias': sharedX(0.1), 'Wf': sharedX(0.01)}
    """

    updates = OrderedDict()
    momentum = OrderedDict()

    for (param, gparam) in grads.iteritems():

        if momentum_lambda:
            # create storage for momentum term
            momentum[param] = sharedX(numpy.zeros_like(param.get_value()), name=param.name + '_old')
            new_grad = (1.-momentum_lambda) * gparam + momentum_lambda * momentum[param]
            updates[param] = param - new_grad
            updates[momentum[param]] = new_grad
        else:
            updates[param] = param - gparam

    return updates
