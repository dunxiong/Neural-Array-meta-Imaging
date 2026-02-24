# Select deconvolution method

from networks.G.MFWDFNet import *

def select_G(params, args):
    if args.G_network == 'MFWDFNet':

        feat_extract = FeatExtractRCAB()

        # Wiener deconvolove in Feature domain
        if args.wiener_num_psf == 'single_psf':
            deconv_wnr = DeconvWNRinFPsinglePSF(params, args)
        elif args.wiener_num_psf == 'multi_psf':
            deconv_wnr = DeconvWNRinFPmultiPSF(params, args)
        else:
            raise ValueError("Invalid deconv_wnr mode")
        
        # clear Feature fusion

        fusion_net = FusionNAFnet(params)
        
        return MFWDFNet(feat_extract, deconv_wnr, fusion_net)

    elif args.G_network == 'WinerU':
        return WinerU(params,args)
    
    elif args.G_network == 'MultiWinerU':
        return MultiWinerU(params,args)

    elif args.G_network == 'MultiFlatNet':
        return MultiFlatNet(params,args)

    else:
        assert False, ("Unsupported generator network")

