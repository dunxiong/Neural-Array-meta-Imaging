# Select deconvolution method

from processing.G.DN import DN
from processing.G.DN import DN_no

def select_G(params, args):
    if args.G_network == 'DN':
        print('net')
        return DN(params, args)
    if args.G_network == 'NoDNN':
        print('nonet')
        return DN_no(params, args)
    else:
        assert False, ("Unsupported generator network")
