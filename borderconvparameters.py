# Implements convolutional 2D parameters for RBMs, with border padding.
#
# Copyright (c) 2016 Gijs van Tulder / Erasmus MC, the Netherlands
# This code is licensed under the MIT license. See LICENSE for details.
from morb.base import Parameters

import theano
import theano.tensor as T
from theano.tensor.nnet import conv

# from morb.misc import tensordot # better tensordot implementation that can be GPU accelerated
tensordot = T.tensordot # use theano implementation
class Convolutional2DParameters(Parameters):
    def __init__(self, rbm, units_list, W, shape_info=None, name=None, energy_multiplier=1, var_fixed_border=None, shared_hidden_dims=0, divide_by_number_of_hiddens=False, alternative_gradient=False):
        # use the shape_info parameter to provide a dict with keys:
        # hidden_maps, visible_maps, filter_height, filter_width, visible_height, visible_width, mb_size
        
        super(Convolutional2DParameters, self).__init__(rbm, units_list, name=name, energy_multiplier = energy_multiplier)
        assert len(units_list) == 2
        self.var = W # (hidden_maps, visible_maps, filter_height, filter_width)
        self.variables = [self.var]
        self.vu = units_list[0] # (mb_size, visible_maps, visible_height, visible_width)
        self.hu = units_list[1] # (mb_size, hidden_maps, hidden_height, hidden_width)
        self.shape_info = shape_info
        self.alternative_gradient = alternative_gradient

        if var_fixed_border:
            self.borders_zero = False
            self.margin = shape_info['filter_height'] - 1
            self.var_fixed_border = var_fixed_border # (mb_size, visible_maps, visible_height + 2 * margin, visible_width + 2 * margin)
        else:
            self.borders_zero = True

        if not shared_hidden_dims in (0,1):
            raise "shared_hidden_dims is not 0 or 1"

        # conv input is (output_maps, input_maps, filter height [numrows], filter width [numcolumns])
        # conv input is (mb_size, input_maps, input height [numrows], input width [numcolumns])
        # conv output is (mb_size, output_maps, output height [numrows], output width [numcolumns])
        
        def term_vu(vmap, pmap):
            # input = hiddens, output = visibles so we need to swap dimensions
            W_shuffled = pmap[self.var].dimshuffle(1, 0, 2, 3)
            if self.filter_shape is not None:
                shuffled_filter_shape = [self.filter_shape[k] for k in (1, 0, 2, 3)]
            else:
                shuffled_filter_shape = None
            # sum over bias dimension, if necessary
            hu_for_v = vmap[self.hu]
            if shared_hidden_dims == 1:
              # from (mb_size, hidden_maps, bias_sets, hidden_height, hidden_width)
              # to   (mb_size, hidden_maps,            hidden_height, hidden_width)
              hu_for_v = T.sum(hu_for_v, axis=2)
            # (this requires a flipped convolution; conv2d does that)
            if self.borders_zero:
              # the visible units do not include margins
              return conv.conv2d(hu_for_v, W_shuffled, border_mode='full', \
                                 image_shape=self.hidden_shape, filter_shape=shuffled_filter_shape)
            else:
              # ignore the visible unit borders
              return conv.conv2d(hu_for_v, W_shuffled, border_mode='valid', \
                                 image_shape=self.hidden_shape, filter_shape=shuffled_filter_shape)
            
        def term_hu(vmap, pmap):
            # input = visibles, output = hiddens, flip filters
            # (flip because conv2d flips the kernel a second time)
            W_flipped = pmap[self.var][:, :, ::-1, ::-1]
            if self.borders_zero:
              c = conv.conv2d(vmap[self.vu], W_flipped, border_mode='valid', \
                              image_shape=self.visible_shape, filter_shape=self.filter_shape)
            else:
              v_with_borders = self.add_fixed_borders(vmap[self.vu], vmap)
              c = conv.conv2d(v_with_borders, W_flipped, border_mode='valid', \
                              image_shape=self.visible_shape_with_border, filter_shape=self.filter_shape)
            if shared_hidden_dims == 1:
              # share over biases
              # (mb_size, hidden_maps, biases, hidden_height, hidden_width)
              c = c.dimshuffle(0, 1, 'x', 2, 3)
            return c
        
        self.terms[self.vu] = term_vu
        self.terms[self.hu] = term_hu
        
        def gradient(vmap, pmap):
            raise NotImplementedError # TODO
        
        def gradient_sum(vmap, pmap):
            if self.visible_shape is not None:
                if self.alternative_gradient or self.borders_zero:
                    i_shape = [self.visible_shape[k] for k in [1, 0, 2, 3]]
                else:
                    i_shape = [self.visible_shape_with_border[k] for k in [1, 0, 2, 3]]
            else:
                i_shape = None
        
            if self.hidden_shape is not None:
                f_shape = [self.hidden_shape[k] for k in [1, 0, 2, 3]]
            else:
                f_shape = None
            
            if self.alternative_gradient or self.borders_zero:
                v_shuffled = vmap[self.vu].dimshuffle(1, 0, 2, 3)
            else:
                v_shuffled = self.add_fixed_borders(vmap[self.vu], vmap).dimshuffle(1, 0, 2, 3)

            # sum over bias dimension, if necessary
            hu_for_v = vmap[self.hu]
            if shared_hidden_dims == 1:
              # from (mb_size, hidden_maps, bias_sets, hidden_height, hidden_width)
              # to   (mb_size, hidden_maps,            hidden_height, hidden_width)
              hu_for_v = T.sum(hu_for_v, axis=2)
            h_shuffled = hu_for_v.dimshuffle(1, 0, 2, 3)

            # (flip because conv2d flips the kernel a second time)
            if self.alternative_gradient:
                v_shuffled = v_shuffled[:, :, ::-1, ::-1]
                c = conv.conv2d(h_shuffled, v_shuffled, border_mode='valid', image_shape=f_shape, filter_shape=i_shape)
                c = c[:, :, ::-1, ::-1]
            else:
                h_shuffled = h_shuffled[:, :, ::-1, ::-1]
                c = conv.conv2d(v_shuffled, h_shuffled, border_mode='valid', image_shape=i_shape, filter_shape=f_shape)
                c = c.dimshuffle(1, 0, 2, 3)

            # must use the mean over all hidden nodes
            # ( = the size of the feature maps )
            # (see, e.g., Lee et al., 2012:
            #  "Unsupervised Learning of Hierarchical Representations
            #   with Convolutional Deep Belief Networks")
            #
            # (2013.08.02: I now think this is not correct.)
#           number_of_hiddens = 1 # self.hidden_shape[2] * self.hidden_shape[3] * self.visible_shape[1]
#           return c.dimshuffle(1, 0, 2, 3) / number_of_hiddens

            if divide_by_number_of_hiddens:
              number_of_hiddens = self.hidden_shape[2] * self.hidden_shape[3]
#             print "Number of hiddens: %d", number_of_hiddens
              return c / number_of_hiddens

            return self.energy_multiplier * c
            return theano.printing.Print("BorderConvGradientSum")(self.energy_multiplier)
            
        self.energy_gradients[self.var] = gradient
        self.energy_gradient_sums[self.var] = gradient_sum
    
    @property    
    def filter_shape(self):
        keys = ['hidden_maps', 'visible_maps', 'filter_height', 'filter_width']
        if self.shape_info is not None and all(k in self.shape_info for k in keys):
            return tuple(self.shape_info[k] for k in keys)
        else:
            return None

    @property            
    def visible_shape(self):
        keys = ['mb_size', 'visible_maps', 'visible_height', 'visible_width']                
        if self.shape_info is not None and all(k in self.shape_info for k in keys):
            return tuple(self.shape_info[k] for k in keys)
        else:
            return None

    @property            
    def visible_shape_with_border(self):
        keys = ['mb_size', 'visible_maps', 'visible_height', 'visible_width']                
        if self.shape_info is not None and all(k in self.shape_info for k in keys):
            s = [self.shape_info[k] for k in keys]
            if not self.borders_zero:
                s[2] += 2 * self.margin
                s[3] += 2 * self.margin
            return tuple(s)
        else:
            return None

    @property            
    def hidden_shape(self):
        keys = ['mb_size', 'hidden_maps', 'visible_height', 'visible_width']
        if self.shape_info is not None and all(k in self.shape_info for k in keys):
            if not self.borders_zero:
                hidden_height = self.shape_info['visible_height'] + (2 * self.margin) - self.shape_info['filter_height'] + 1
                hidden_width = self.shape_info['visible_width'] + (2 * self.margin) - self.shape_info['filter_width'] + 1
            else:
                hidden_height = self.shape_info['visible_height'] - self.shape_info['filter_height'] + 1
                hidden_width = self.shape_info['visible_width'] - self.shape_info['filter_width'] + 1
            return (self.shape_info['mb_size'], self.shape_info['hidden_maps'], hidden_height, hidden_width)
        else:
            return None
        
    def energy_term(self, vmap, pmap):
        q = self.terms[self.hu](vmap, pmap) * vmap[self.hu]
        return - self.energy_multiplier * T.sum(q, axis=range(1, q.ndim))
        # sum over all but the minibatch axis

    def add_fixed_borders(self, s_var, vmap):
        # load fixed borders
        s = vmap[self.var_fixed_border]
        return T.set_subtensor(s[:, :, self.margin:(self.margin + self.shape_info['visible_height']), self.margin:(self.margin + self.shape_info['visible_width'])], s_var)

    def remove_fixed_borders(self, s_var):
        return s_var[:, :, self.margin:(self.margin + self.shape_info['visible_height']), self.margin:(self.margin + self.shape_info['visible_width'])]

