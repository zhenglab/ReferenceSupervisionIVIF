
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']      # one input: L

    if model == 'plain':
        from models.model_plain import ModelPlain as M

    elif model == 'plain2':  # two inputs: L, C
        from models.model_plain2 import ModelPlain as M

    elif model == 'plain3':  # two inputs: L, C
        from models.model_plain3 import ModelPlain as M

    elif model == 'plain5':  # four inputs: L, k, sf, sigma
        from models.model_plain5 import ModelPlain as M

    elif model == 'plain7':     # one input: L
        from models.model_plain7 import ModelPlain as M

    elif model == 'plain9':     # one input: L
        from models.model_plain9 import ModelPlain as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
