import numpy as np
import glob, os, shutil, socket, warnings
import string, gc, time, copy
import datetime
import astropy.units as u
from collections import OrderedDict

from binary_integration import integration

from generate_timings import timings

import inspect

try:
    import configparser
except(ImportError):
    import ConfigParser as configparser

try:
   from nessai.model import Model
   #TODO: add nessai to requirements. E.g. conda install conda-forge::nessai or pip install nessai>=0.13.2
except ImportError:
   raise ImportError("Nessai not installed. Please install nessai.")

warnings.filterwarnings("ignore")


# TODO: here is the nessai wrapper class. Without being able to run, it is hard to say whether it works.
class nessai_wrapper(Model):
    def __init__(self, data, components, fixed_components):

        """ 
        """
        # initialize the parent class
        #self.fixed_components = fixed_components
        self.data = data
        self.ndata = len(data[0])
        # A list of names to be sampled over. This is a list of strings.
        # Below my guess
        self.names = list(components.keys())
        # A list of bounds to be sampled over. This is dictionary of two-elements lists.
        # Below my guess
        self.bounds = {}
        for name in self.names:
            self.bounds[name] = [components[name][0], components[name][1]]

        

        # print("ndata",self.als_fitter.ndata)
        print("wrapper init")
        print(self.names)
        print(self.bounds)
        # print(self.als_fitter.components)
        # members = inspect.getmembers(Model)
        # for name, member in members:
        #    print(name, type(member))
      
    def log_likelihood(self, p):
        """Compute the log likelihood of the model given the data."""
        # Call the parent class method to compute the log likelihood
        print("callinng likelihood",p)
        return self.lnlhood_mn(p)
   
    def log_prior(self, p):
        """Compute the log prior of the model parameters."""
        print("calling prior",p)
        # Check whether it is in the bounds
        log_p = np.log(self.in_bounds(p), dtype="float")
        # Call the als_fitter instance method to compute the log prior
        return log_p+self.lnprior(p)
    
    def lnprior(self,p):
        print("bounds in lnprior",self.bounds)
        print("p into lnprior", p)

        if all(b[0] <= v <= b[1] for v, b in zip(p, self.bounds)):

            pav = 0

            return pav

        return -np.inf
    
    def lnlhood_mn(self,p):
       
        t_p = self.reconstruct(p)

        if len(t_p) < self.ndata:
            return -np.inf

        ispec2 = 1./((self.data[1])**2)

        lhood = -0.5*np.nansum((ispec2*(self.data[0]-t_p[:self.ndata])**2 - np.log(ispec2) + np.log(2.*np.pi)))

        return lhood         



    def reconstruct(self,p):
        #get parameters of the model   
        print("p into recontruct = ",p)
        p_dict = {}
        for i,d in enumerate(self.names):
            p_dict[d[:-5]] = p[i]

        #p_dict.update(self.fixed_components)

        # print(10**p_dict["m1"],10**p_dict["m2"],p_dict["sma"],p_dict["ecc"],p_dict["inc"], p_dict["arg_peri"], p_dict["asc_node"], p_dict["spin"],p_dict["disc_inc"],self.fixed_components["d_obs"],p_dict["theta_obs"],self.fixed_components["phi_obs"])
        #t_p = integration(10**p_dict["m1"],
        #                10**p_dict["m2"],
        #                p_dict["sma"],
        #                p_dict["ecc"],
        #                p_dict["inc"], 
        #                p_dict["arg_peri"], 
        #                p_dict["asc_node"],
        #                p_dict["true_anomaly"], 
        #                p_dict["spin"],
        #                p_dict["disc_inc"],
        #                p_dict["rdisc_min"],
        #                p_dict["rdisc_max"],
        #                p_dict["d_obs"],
        #                p_dict["theta_obs"],
        #                p_dict["phi_obs"])

        t_p = timings(p_dict["m1"],
                          p_dict["sma"],
                          p_dict["ecc"],
                          p_dict["inc"], 
                          p_dict["spin"],
                          p_dict["disc_inc"],
                          p_dict["theta_obs"])

                          
        t_p = t_p - p_dict["toff"]

        t_p = t_p[t_p>0]

        return t_p[:self.ndata]

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

def readconfig(configfile=None, logger=None):

   """
   Build parameter dictionaries from input configfile.
   """

   booldir = {'True':True, 'False':False}

   input_params = configparser.ConfigParser()
   input_params.read(configfile)

   run_params = {}

   #Start with mandatory parameters
   if not input_params.has_option('input', 'infile'):
      raise configparser.NoOptionError("input", "infile")
   else:
      run_params['infile']  = input_params.get('input', 'infile')

   if input_params.has_option('input', 'solver'):
      run_params["solver"] = input_params.get('input', 'solver')
      # TODO: raising a warning if the solver is nessai and no options are provided 
      if run_params["solver"] == 'nessai':
         if not input_params.has_section('nessaispecs'):
            raise configparser.NoSectionError("nessaispecs")
   else:
      # TODO: still defaults to multinest
      run_params["solver"] = 'multinest'

   #Paths are desirable but not essential, default to cwd

   if not input_params.has_option('pathing', 'outdir'):
      run_params["outdir"] = './'
   else:
      run_params["outdir"] = input_params.get('pathing', 'outdir')

   if not input_params.has_option('pathing', 'chaindir'):
      run_params["chaindir"] = run_params["outdir"]+'fits/'
   else:
      run_params["chaindir"] = run_params["outdir"]+input_params.get('pathing', 'chaindir')

   if not input_params.has_option('pathing', 'plotdir'):
      run_params["plotdir"] = run_params["outdir"]+'plots/'
   else:
      run_params["plotdir"] = run_params["outdir"]+input_params.get('pathing', 'plotdir')

   if not input_params.has_option('pathing', 'chainfmt'):
      run_params["chainfmt"] = 'pc_fits_{}_{1}'
   else:
      run_params["chainfmt"] = input_params.get('pathing', 'chainfmt')


   # read parameters of the model considered

   run_params["components"] = OrderedDict()
   for d in input_params.options("components"):
      run_params["components"][d] = np.array(input_params.get("components",d).split(','), dtype=float)

   run_params["fixed_components"] = OrderedDict()
   for d in input_params.options("fixed_components"):
      run_params["fixed_components"][d] = np.array(input_params.get("fixed_components",d).split(','), dtype=float)

   run_params["labels"] = OrderedDict()
   for d in input_params.options("labels"):
      run_params["labels"][d] = input_params.get("labels",d)



   if input_params.has_option('plots', 'nmaxcols'):
      run_params["nmaxcols"] = int(input_params.get('plots', 'nmaxcols')[0])
   else:
      run_params["nmaxcols"] = 5

   if input_params.has_option('plots', 'yrange'):
      run_params["yrange"] = np.array(input_params.get('plots', 'yrange').split(','), dtype=float)
   else:
      run_params["yrange"] = np.array((-0.1,1.2))

   #Parameters driving the run
   if input_params.has_option('run', 'dofit'):
      run_params["dofit"] = booldir[input_params.get('run', 'dofit')]
   else:
      run_params["dofit"] = True

   if input_params.has_option('run', 'doplot'):
      run_params["doplot"] = booldir[input_params.get('run', 'doplot')]
   else:
      run_params["doplot"] = True


   if input_params.has_section('mnsettings'):

      settingsdict = (dict((opt, booldir[input_params.get('mnsettings',opt)])
               if input_params.get('mnsettings',opt) in ['True','False']
               else (opt, input_params.get('mnsettings', opt))
               for opt in input_params.options('mnsettings')))

      run_params['mnsettings'] = settingsdict
   
   # Check whether the sampler is nessai and options are not provided 
   # in the config file. If so, set default values.
      
   # TODO: Adding nessai properties through the cfg file
   if input_params.has_section('nessaispecs'):
      nessaispecs = (dict((opt, booldir[input_params.get('nessaispecs',opt)])
               if input_params.get('nessaispecs',opt) in ['True','False']
               else (opt, input_params.get('nessaispecs', opt))
               for opt in input_params.options('nessaispecs')))
      run_params['nessaispecs'] = nessaispecs
   return run_params
