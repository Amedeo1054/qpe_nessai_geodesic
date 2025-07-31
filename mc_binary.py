import numpy as np
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import argparse, datetime
import matplotlib.pyplot as plt
import binary_fitter_nessai as fitter
import os
import corner


plt.rc('text', usetex=True)
plt.rcParams['font.size'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('config')
args = parser.parse_args()

configpars = fitter.readconfig(args.config)

if args.debug:
   print('DEBUG mode, increased verbosity')

#Create directories if necessary
if True:
 if not os.path.isdir(configpars['chaindir']):
   os.makedirs(configpars['chaindir'])
 if not os.path.isdir(configpars['plotdir']):
   os.makedirs(configpars['plotdir'])

# read data to be fitted
data = np.loadtxt(configpars["infile"],unpack=True)

if configpars['dofit']:
  
    if configpars['solver']=='nessai':
        print("Using nessai")

        # try importing the nessai bits 
        try:
            from nessai.flowsampler import FlowSampler
        #TODO: add nessai to requirements. E.g. conda install conda-forge::nessai or pip install nessai>=0.13.2
        except ImportError:
            raise ImportError("Nessai not installed. Please install nessai to use this functionality.")

        nessaispecs = configpars['nessaispecs']
        # Set some sampling defaults
        defaults = {'flow_config':None, 'checkpointing':True, 'nlive':10, 'n_pool':1, 'resume': True, 'seed':42, 'output': './'+configpars['chaindir']+'/'}
        # Then use them together with nessaispecs from the cfg to create the nessai kwargs.
        nessaikwargs = {}
        for key in defaults: 
            if key in nessaispecs:
                nessaikwargs[key] = nessaispecs[key]
            else:
                nessaikwargs[key] = defaults[key]   

        for key in ['nlive','n_pool','seed']:
            nessaikwargs[key] = int(nessaikwargs[key])     

        nessaimodel = fitter.nessai_wrapper(data, configpars['components'], configpars['fixed_components'])

        sampler = FlowSampler(model=nessaimodel, 
                               importance_nested_sampler=True,
                               **nessaikwargs)
        # Now run.
        t0 = datetime.datetime.now()
        sampler.run()
        t1 = datetime.datetime.now()
        print('Execution time {}'.format(t1-t0))
