#!/usr/bin/env python3
import numpy as np
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import glob, os, shutil, warnings
import string, gc, time, copy
import argparse, datetime
import matplotlib.pyplot as plt
from mpi4py import MPI
import pymultinest
import binary_fitter_modified as fitter
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

   with fitter.als_fitter(data, configpars['components'], configpars['fixed_components'], debug= args.debug) as temp:

      out_fmt = configpars['chaindir']+'/'+configpars['chainfmt']

      if 'mnsettings' in configpars:
         configsettings = configpars['mnsettings']
         if 'nlive' in configsettings:
            nlive = int(configsettings['nlive'])
         else:
            nlive = 1000
         if 'samplingeff' in configsettings:
            samplingeff = float(configsettings['samplingeff'])
         else:
            samplingeff = 0.3
      else:
         nlive = 300
         samplingeff = 0.6#0.3

      t0 = datetime.datetime.now()
      pymultinest.run(temp.lnlhood_mn, temp._scale_cube_mn, temp.ndim, sampling_efficiency=samplingeff, resume=False, \
            outputfiles_basename=out_fmt, verbose=True, multimodal=False, \
            importance_nested_sampling=False, n_live_points=nlive, n_iter_before_update=100, \
            evidence_tolerance=0.1)

      t1 = datetime.datetime.now()
      if MPI.COMM_WORLD.rank==0:
         print('Execution time {}'.format(t1-t0))


if configpars['doplot']:

   #This can only be one on one core....
   with fitter.als_fitter(data,configpars['components'], configpars['fixed_components'], debug= args.debug) as temp:
   
      out_fmt = configpars['chaindir']+'/'+configpars['chainfmt']
      out_plot = configpars['plotdir']+'/'+configpars['chainfmt']

      lab = [r"{:}".format(configpars["labels"][a]) for a in configpars["labels"]]
      print("labels:",lab)

      a = pymultinest.Analyzer(n_params=temp.ndim, outputfiles_basename=out_fmt)
      values = a.get_equal_weighted_posterior()

      stat = a.get_stats()

      lnz, dlnz = stat['global evidence'], stat['global evidence error']
      print('  Nested Sampling Ln(z):   {0:6.3f}'.format(lnz))

      # meds  = np.percentile(values, 50, axis=0)
      # percs = np.transpose(np.percentile(values, [16,50,84], axis=0))

      best_fit = values[np.argmax(values[:,-1])]
      best_fit = best_fit[:-1]
      print("best_fit =",best_fit)

      figure = corner.corner(values[:,:-1], 
      labels=lab, \
      quantiles=[0.16, 0.5, 0.84], \
      show_titles=True, title_kwargs={"fontsize": 10})

      index = np.random.choice(len(values),size=50)
      values2 = values[index,:-1]

      figure.savefig(out_plot+".pdf",bbox_inches="tight")

      plt.show()
