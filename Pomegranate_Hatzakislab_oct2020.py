
import numpy as np

from pims import ImageSequence
import numpy as np
import pandas as pd
import trackpy as tp
import matplotlib  as mpl
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.feature import blob_log
from skimage import feature
from scipy.stats.stats import pearsonr 
import os
import scipy
import scipy.ndimage as ndimage
from skimage import measure
from skimage.color import rgb2gray 
from skimage import io

   
from pims import TiffStack

import scipy

from skimage import feature

    

from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.feature import blob_log
from skimage import feature
from scipy.stats.stats import pearsonr 
import os
import scipy
import scipy.ndimage as ndimage
from skimage import measure
from skimage.color import rgb2gray
import matplotlib.patches as mpatches   
import glob
from skimage import measure

from pomegranate import *
import time

import random
import itertools

import cython
import probfit

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn import neighbors
from tqdm import tqdm
import iminuit as Minuit

from probfit import BinnedLH, Chi2Regression, Extended, BinnedChi2, UnbinnedLH, gaussian
from matplotlib import patches

##################################################################################################################################
##################################################################################################################################
#################################### Edit below #################################################################
##################################################################################################################################
##################################################################################################################################





#save where for all below
save_path = "some_save_path" # must exist



#######################################################
#### specify parameters for challenge sets and simulated data #####
#################################################################

simulated_data = False # set to true to run


simul_data_path_training = "/some_path/"
simul_data_path_challenge = "/some_other_path/"

n_states  = 2 # specify number of states

# create clustering, fit lifetimes and provide tdp

fit_tdp_simul = False # set to true to run

path_to_hmm_treated_challenge = "/some_path_to_csv_file_from_above"

n_clusters = 3 # provide number of clusters( transitions) to fit


# set framerate of data
sec_per_frame = 0.1 #

#######################################################
#### specify parameters for experimental sets #####
#################################################################

# Traces are cut to max 500 frames

experimental_sets = False # set to true to run

#please provide the enclosed model path "/The model.json"
path_to_general_model ="/some_path_to_enclosed_overall_model"
exp_data_folder_path = "/some_path_to_exp_folder_containing .dat files"


# create clustering, fit lifetimes and provide tdp

fit_tdp_exp=False#put true to run  


path_to_hmm_treated_exp= "/some_path_to_csv_file_from_above"
n_clusters = 3 # provide number of clusters( transitions) to fit


# set framerate of data
sec_per_frame = 0.1 #

##################################################################################################################################
##################################################################################################################################
#################################### no more editing #################################################################
##################################################################################################################################
##################################################################################################################################







def f1(r,D):
    if D> 0:
        return (r/(2.*D*0.03))*np.exp(-((r**2)/(4.*D*0.03)))
    else:
        return 0

def f2(r,D2):
    if D2> 0:
        return (r/(2.*D2*0.03))*np.exp(-((r**2)/(4.*D2*0.03)))
    else:
        return 0
def f3(r,D3):
    if D3> 0:
        return (r/(2.*D3*0.03))*np.exp(-((r**2)/(4.*D3*0.03)))
    else:
        return 0  
def f4(r,D4):
    if D4> 0:
        return (r/(2.*D4*0.03))*np.exp(-((r**2)/(4.*D4*0.03)))
    else:
        return 0          

def fix_ax_probs(ax,x_label,y_label):
    ax.set_ylabel(y_label, size = 12)
    ax.set_xlabel(x_label, size = 12)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 10)
    return ax



def find_multiple_D_from_df(r,time_interval,distributions):
    """
    Give a single list of step lengths - can thus be for a single particle or entire population
    no need to create histogram as the function will find a pdf
    Change time if needed
    
    """
    r = np.asarray(r)
    compdf1 = probfit.functor.AddPdfNorm(f1)
    compdf2 = probfit.functor.AddPdfNorm(f1,f2)
    compdf3 = probfit.functor.AddPdfNorm(f1,f2,f3)
    compdf4 = probfit.functor.AddPdfNorm(f1,f2,f3,f4)
    compdf5 = probfit.functor.AddPdfNorm(f1,f2,f3,f4,f5)
    
    ulh1 = UnbinnedLH(compdf1, r, extended=False)
    ulh2 = UnbinnedLH(compdf2, r, extended=False)
    ulh3 = UnbinnedLH(compdf3, r, extended=False)
    ulh4 = UnbinnedLH(compdf4, r, extended=False)
    ulh5 = UnbinnedLH(compdf5, r, extended=False)
    
    import iminuit
    m1 = iminuit.Minuit(ulh1, D=1., limit_D = (0,5),pedantic= False,print_level = 0)
    m2 = iminuit.Minuit(ulh2, D=1., limit_D = (0,5), D2=.1, limit_D2 = (0,5),pedantic= False,print_level = 0)
    m3 = iminuit.Minuit(ulh2, D=1., limit_D = (0,5), D2=.1, limit_D2 = (0,5),pedantic= False,print_level = 0)
    
    m1.migrad(ncall=10000)
    m2.migrad(ncall=10000)

    return m1,m2

def fix_ax_probs(ax,x_label,y_label):
    ax.set_ylabel(y_label, size = 12)
    ax.set_xlabel(x_label, size = 12)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 10)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc = "upper right",frameon = False)
    return ax


def _bic(data_len, k_states, log_likelihood):
    return np.log(data_len) * k_states - 2 * log_likelihood

def get_model_from_dataframe(df,distributions,save_path,n_states_bic):
    print ("Generating model...")
    #df= df[df.steplength > 0.01]
    data = df['FRET_E'].tolist()
    data=np.asarray(data)
    data = data.reshape(1, -1)
    
    data_to_fit = [] 
    group_all = df.groupby('particle')
    for name, group in group_all:
        data_to_fit.append(np.asarray(group.FRET_E.tolist()))


    # learn from data
    model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=int(distributions), X=data_to_fit)
    model.fit(data_to_fit)
    
    model.bake()
    
    log= model.log_probability(data)
    bic = _bic(len(data),n_states_bic,log)
   
   
    print ("Model baked..")
    t = model.to_json()
    import json
    with open(str(save_path+'data.json'), 'w', encoding='utf-8') as f:
        json.dump(t, f)
    # use custom fitted model is gen 2
    
    return model,bic,log



def get_model_BIC(df,distributions):
    print ("Generating model...")
    df= df[df.steplength > 0.01]
    data = df['steplength'].tolist()
    data=np.asarray(data)
    data = data.reshape(1, -1)
    
    data_to_fit = [] 
    group_all = df.groupby('particle')
    for name, group in group_all:
        data_to_fit.append(np.asarray(group.steplength.tolist()))
    

    # learn from data
    model = HiddenMarkovModel.from_samples(GammaDistribution, n_components=int(distributions), X=data_to_fit)
    model.fit(data_to_fit)
    model.bake()
    print ("Model baked..")
    bic_long = model.log_probability(data)
    
    
    return bic_long,model



def load_model(path):
    import json
    with open(str(path), 'r', encoding='utf-8') as f:
        t = json.load(f)
    model = HiddenMarkovModel.from_json(t)
    return model

def find_d_single_from_df(r):
    r = np.asarray(r)
    def f1(r,D):
        if D> 0:
            return (r/(2.*D*0.030))*np.exp(-((r**2)/(4.*D*0.03)))
        else:
            return 0
    compdf1 = probfit.functor.AddPdfNorm(f1)
    
    ulh1 = UnbinnedLH(compdf1, r, extended=False)
    import iminuit
    m1 = iminuit.Minuit(ulh1, D=0.1, limit_D = (0,2),pedantic= False,print_level = 0)
    m1.migrad(ncall=30000)
    return m1.values['D']
    

def run_hmm_treat(df_raw,model,save_path):
    print ("Running dataframe and fitting traces to hmm model...")
    sub_path = str(save_path+'_traces_hmm/')
    if not os.path.exists(sub_path):                      # creates a folder for saving the stuff in if it does not exist
        os.makedirs(sub_path)
    run = 0
    df_raw = df_raw.sort_values(['particle','time'], ascending=True)
    #df_raw = df_raw[df_raw.steplength >0.01]
    df_new = pd.DataFrame() 
    df_new_full = pd.DataFrame() 
    
    
    
    group_all = df_raw.groupby('particle')
    for name, group in tqdm(group_all):
        tmp_trace = group.FRET_E.tolist()
        tmp_trace = (np.asarray(tmp_trace))
        tmp_trace_re = tmp_trace.reshape(-1, 1)
        data_len = len(tmp_trace)
        time = np.linspace(1, data_len, data_len)
        
        d_full_trace = find_d_single_from_df(tmp_trace)
        state_seq = model.predict(tmp_trace_re)
        #if sum(np.diff(abs(np.asarray(state_seq[1:])))) ==0:
        #    fig,ax = plt.subplots(figsize=(6, 3))
        #    ax.plot(tmp_trace, "gray", linewidth = 3,alpha = 0.8)
        #    ax.set_ylabel("FRET")
        #    ax.set_xlabel("t")
        #    ax.set_ylim(0,1.5)
        #    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
        #    ax.tick_params(axis = 'both', which = 'minor', labelsize = 14)
        #    ax.spines['right'].set_visible(False)
        #    ax.spines['top'].set_visible(False)
        #    ax.grid(False)
        #    fig.tight_layout()    
        #    fig.savefig(str(sub_path+str('__')+str(run)+'.pdf')  ) 
        #    plt.clf()
        #    plt.close("all")
        #    
        #    run +=1    
        #    continue
        df = pd.DataFrame({"time": time, "step": tmp_trace, "state": state_seq})
        df['track_id'] = str(name)
        df["unique_state_ID"] = df["state"].transform(lambda group: (group.diff() != 0).cumsum())
        
        df["idealized"] = df.groupby(["state"], as_index = False)["step"].transform("mean") # but does this make sense, better fit and fit a d either by fitting a straight line to msd or by fitting the very few data pointds tp a PDF
        df["after"] = np.roll(df["idealized"], -1)
#        
#        if run %1 ==0:
#            #print (str(("Plotting trace with hmm " + str(run))))
#            fig,ax = plt.subplots(figsize=(6, 3))
#           
#            ax.plot(df['step'].values, "gray", linewidth = 1,alpha = 0.8)
#            ax.plot(df['idealized'].values, "firebrick",linewidth = 1,alpha = 0.8)
#           
#            ax.set_ylabel("FRET")
#            ax.set_xlabel("t")
#            ax.set_ylim(0,1.5)
#            ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
#            ax.tick_params(axis = 'both', which = 'minor', labelsize = 14)
#            ax.spines['right'].set_visible(False)
#            ax.spines['top'].set_visible(False)
#            ax.grid(False)
#            fig.tight_layout()    
#            fig.savefig(str(sub_path+'__'+str(run)+'__.pdf')  ) 
#            plt.clf()
#            plt.close("all")
            
        
        df['d'] = df.groupby('state')['step'].transform(find_d_single_from_df) # UBLH
        df["d_after"] = np.roll(df["d"], -1) 
        
        
        df["state_jump"] = df["idealized"].transform(lambda group: (abs(group.diff()) != 0).cumsum())
        df_tmp_full = df.copy()
    
        df = df.drop_duplicates(subset = "state_jump", keep = "last")
        df['single_d'] = d_full_trace
        
        timedif = np.diff(df["time"])
        timedif = np.append(np.nan, timedif)
        df["lifetime"] = timedif
        df['track'] = name
        df = df[1:]
        df_new= df_new.append(df, ignore_index = True)
        
        df_new_full = df_new_full.append(df_tmp_full, ignore_index = True)
        run +=1    
    return df_new   ,df_new_full  


#tester_df  = pd.read_csv('/Volumes/Soeren/Lipase/treat_2/Amalie_L3/L3_product_toSoren/all_tracked_p2.csv', low_memory=False, sep = ',') 

#fig,ax =plt.subplots(2,1)
#a=ax[0].hist2d(tester_df['x'],tester_df['y'],10,weights = tester_df['steplength'])
#b=ax[1].hist2d(tester_df['x'],tester_df['y'],10)




def create_grid_like_data_labels(df,max_val,grid_size):
    """
    Put spt data into grid
    max_val is location in pixel or micron, depending on data type
    """


    max_val = 81.2
    grid_size = 10
    sections = np.linspace(0,max_val,int(grid_size))
    grid_id = np.arange(0,len(sections),1)
    x_vals = np.asarray(df['x'].tolist())
    y_vals = np.asarray(df['y'].tolist())   
    
    grid_id_list = []
    #for x,y in zip(sections):
        


    
    

 
def countour_2d(xdata, ydata, n_colors = 2, kernel = "gaussian", extend_grid = 1, bandwidth = 0.1, shade_lowest = False, gridsize = 100, bins = "auto"):
    """
    Valid kernels for sklearn are
    ['gaussian' | 'tophat' | 'epanechnikov' | 'exponential' | 'linear' | 'cosine']

    Example
    -------
    X, Y, Z, lev = countour_2d(x, y, shade_lowest = False)
    fig, ax = plt.subplots()
    c = ax.contourf(X,Y,Z, levels=lev, cmap = "inferno")
    fig.colorbar(c)

    Alternatively, unpack like
    contour = countour_2d(x, y, shade_lowest = False)
    c = ax.contourf(*contour)

    """
    if kernel == "epa":
        kernel = "epanechnikov"

    # Stretch the min/max values to make sure that the KDE goes beyond the outermost points
    meanx = np.mean(xdata)*extend_grid
    meany = np.mean(ydata)*extend_grid

    # Create a grid for KDE
    X, Y = np.mgrid[min(xdata)-meanx:max(xdata)+meanx:complex(gridsize),min(ydata)-meany:max(ydata)+meany:complex(gridsize)]

    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([xdata, ydata])

    # Define KDE with specified bandwidth
    kernel_sk = neighbors.KernelDensity(kernel = kernel, bandwidth = bandwidth).fit(list(zip(*values)))
    Z = np.exp(kernel_sk.score_samples(list(zip(*positions))))

    Z = np.reshape(Z.T, X.shape)

    if not shade_lowest:
        n_colors += 1

    locator = ticker.MaxNLocator(n_colors, min_n_ticks = n_colors)

    if len(bins) > 1:
        levels = bins
    elif bins is "auto":
        levels = locator.tick_values(Z.min(), Z.max())
    else:
        raise ValueError("Levels must be either a list of bins (e.g. np.arange) or 'auto'")

    if not shade_lowest:
        levels = levels[1:]

    return X, Y, Z, levels
def error_ellipse(xdata, ydata, n_std, ax = None, return_ax = False, **kwargs):
    """
    Parameters
    ----------
    xdata : array-like
    ydata : array-like
    n_std : scalar
        Number of sigmas (e.g. 2 for 95% confidence interval)
    ax : ax to plot on
    return_ax : bool
        Returns axis for plot
    return_inside : bool
        Returns a list of True/False for inside/outside ellipse
    **kwargs
        Passed to matplotlib.patches.Ellipse. Color, alpha, etc..

    Returns
    -------
    Ellipse with the correct orientation, given the data


    Example
    -------
    x = np.random.randn(100)
    y = 0.1 * x + np.random.randn(100)

    fig, ax = plt.subplots()

    ax, in_out = _define_eclipse(x, y, n_std = 2, ax = ax, alpha = 0.5, return_ax = True)
    ax.scatter(x, y, c = in_out)
    plt.show()

    """

    def _eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    points = np.stack([xdata, ydata], axis = 1)  # Combine points to 2-column matrix
    center = points.mean(axis = 0)  # Calculate mean for every column (x,y)

    # Calculate covariance matrix for coordinates (how correlated they are)
    cov = np.cov(points, rowvar = False)  # rowvar = False because there are 2 variables, not nrows variables

    vals, vecs = _eigsorted(cov)

    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)

    in_out = is_in_ellipse(xdata = xdata, ydata = ydata, center = center, width = width, height = height, angle = angle)
    in_out = np.array(in_out)

    if return_ax:
        ellip = patches.Ellipse(xy = center, width = width, height = height, angle = angle, **kwargs)
        if ax is None:
            ax = plt.gca()
        ax.add_artist(ellip)
    #     return ax, in_out
    # else:
    return in_out
def is_in_ellipse(xdata, ydata, center, width, height, angle):
    """
    Determines whether points are in ellipse, given the parameters of the ellipse

    Parameters
    ----------
    xdata : array-like
    ydata : array-lie
    center : array-like, tuple
        center of the ellipse as (x,y)
    width : scalar
    height : scalar
    angle : scalar
        angle in degrees

    Returns
    -------
    List of True/False, depending on points being inside/outside of the ellipse
    """

    cos_angle = np.cos(np.radians(180 - angle))
    sin_angle = np.sin(np.radians(180 - angle))

    xc = xdata - center[0]
    yc = ydata - center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle

    rad_cc = (xct ** 2 / (width / 2) ** 2) + (yct ** 2 / (height / 2) ** 2)

    in_ellipse = []
    for r in rad_cc:
        in_ellipse.append(True) if r <= 1. else in_ellipse.append(False)

    return in_ellipse
  
def create_tdp_basic(df,save_path):
    """
    Used to initial inspection of data
    """
    blanchard_cols = ["#FDF9E9", "#FDF9E9", "#FFFFFF", "#BACAE9", "#88AAD8", "#527FC0", "#2E5992", "#00AAA4", "#3AB64A", "#CADB2A", "#FFF203", "#FFD26F", "#FAA841", "#F5834F", "#ED1D25"]
    ccmap_lines =  ["lightgrey", "#FFFFFF", "#BACAE9", "#88AAD8", "#527FC0", "#2E5992", "#00AAA4", "#3AB64A", "#CADB2A", "#FFF203", "#FFD26F", "#FAA841", "#F5834F", "#ED1D25"]
    import uncertainties as un
    import uncertainties.umath as umath  
    from uncertainties import unumpy as unp
    import matplotlib
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns

    blanchard_cmap = LinearSegmentedColormap.from_list(name = "", colors = blanchard_cols)

    import matplotlib.colors as mcolors
    std_cols = sns.color_palette("Set2", 8)
    from sklearn import mixture, decomposition, cluster
    from tqdm import tqdm
    
    
    df = df[ abs(df.after-df.idealized)>0.007]
    
    fig,ax = plt.subplots(figsize = (3,3))
    ax.hist(df['idealized'],50, color = "gray",density = True, range = (0,1.1),alpha =0.7)
    ax = fix_ax_probs(ax,'Idealized steps','Density')
    fig.tight_layout()
    fig.savefig(str(save_path+'idealized_steplength_hist.pdf'))
    plt.close('all')
    

    cbins = np.arange(0, 7, 0.5)

    
    cont = countour_2d(xdata = df["lifetime"]/1000,
                            ydata = df["life_after"]/1000,
                            kernel = "linear",
                            bandwidth = 0.1,
                            shade_lowest = True,
                            gridsize = 80,
                            n_colors = len(blanchard_cols),
                            bins = cbins)
    
    fig, ax =  plt.subplots(figsize=(3, 3))
    c = ax.contourf(*cont, cmap = blanchard_cmap, extend = "both")
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)

    ax = fix_ax_probs(ax,'Idealized','Idealized after')
    fig.colorbar(c, ax = ax, extendrect = True)
    fig.tight_layout()
    fig.savefig(str(save_path+'_TDP_non_sorted_steps.pdf'))
    fig.savefig(str(save+'newtdp_from_life.pdf'))
    plt.close()
    









def histbins(binmin, binmax, binwidth = 0.03):
    return np.arange(binmin, binmax, binwidth)

def flatten_list(input_list, as_array = False):
    """
    :param input_list: a list of lists
    :return a single list of all the merged lists
    """

    flat_list = list(itertools.chain.from_iterable(input_list))

    if as_array:
        return np.array(flat_list)
    else:
        return flat_list
def lh_fit(data, f, binned_likelihood, **kwargs):
    import iminuit
    """
    Parameters
    ----------
    data: array
        unbinned data to fit
    f: function
        function to fit which returns the likelihood
    binned_likelihood: bool
        binned or unbinned likelihood
    kwargs:
        write parameters to fit like a (scalar), a_limit (range), fix_a (bool)

    Returns
    -------
    params: array
        array of estimated fit parameters
    errs: array
        array of estimated fit parameter errors
    loglh: scalar
        the minimized log likelihood
    """
    # Create an unbinned likelihood object with function and data.
    if binned_likelihood:
        minimize = probfit.BinnedLH(f, data)
    else:
        minimize = probfit.UnbinnedLH(f, data)

    # Minimizes the unbinned likelihood for the given function
    m = iminuit.Minuit(minimize,
                       **kwargs,
                       print_level = 0,
                       pedantic = False)
    m.migrad()

    params = np.array([val for val in m.values.values()])
    errs   = np.array([val for val in m.errors.values()])
    log_lh = np.sum(np.log(f(data, *params)))

    return params, errs, log_lh      



### true lifetime and tdp
def create_tdf_file(df2,name,save_path,clusters):
    df = df2
    blanchard_cols = ["#FDF9E9", "#FDF9E9", "#FFFFFF", "#BACAE9", "#88AAD8", "#527FC0", "#2E5992", "#00AAA4", "#3AB64A", "#CADB2A", "#FFF203", "#FFD26F", "#FAA841", "#F5834F", "#ED1D25"]
    ccmap_lines =  ["lightgrey", "#FFFFFF", "#BACAE9", "#88AAD8", "#527FC0", "#2E5992", "#00AAA4", "#3AB64A", "#CADB2A", "#FFF203", "#FFD26F", "#FAA841", "#F5834F", "#ED1D25"]
    import uncertainties as un
    import uncertainties.umath as umath  
    from uncertainties import unumpy as unp
    import matplotlib
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns

    blanchard_cmap = LinearSegmentedColormap.from_list(name = "", colors = blanchard_cols)

    import matplotlib.colors as mcolors
    std_cols = sns.color_palette("Set2", 14)
    std_cols=(sns.color_palette("cubehelix", 12))
    
    
    
    experiment_type = name
    from sklearn import mixture, decomposition, cluster
    from tqdm import tqdm
   
           
    df = df[ abs(df.after-df.idealized)>0.007]

    
    cbins = np.arange(0, 7, 0.5)

    
    cont = countour_2d(xdata = df["idealized"],
                            ydata = df["after"],
                            kernel = "linear",
                            bandwidth = 0.2,
                            shade_lowest = True,
                            gridsize = 80,
                            n_colors = len(blanchard_cols),
                            bins = cbins)

    uppr_diag = df2[df2["idealized"] < df2["after"]] # 0
    lwer_diag = df2[df2["idealized"] > df2["after"]] # 1
    halves = [uppr_diag, lwer_diag]
    newdiags = []
       
    for n, diag in enumerate(halves):
        #diag['d']=np.log(diag['d'])
        #diag['d_after']=np.log(diag['d_after'])
        #diag_vals = diag.drop(['lifetime','time','step','state','unique_state_ID','idealized','single_d','after','state_jump','track','track_id','experiment_type'], axis = 1).as_matrix()
        #diag_vals = diag.drop(['lifetime','time','step','state','unique_state_ID','d','single_d','track','d_after','state_jump','track_id','experiment_type'], axis = 1).as_matrix()
        
        # for all hmm
        diag_vals = diag.drop(['lifetime','time','step','state','unique_state_ID','d','d_after','state_jump','track_id','track'], axis = 1).as_matrix()
        #diag_vals = diag.drop(['lifetime','time','step','state','unique_state_ID','idealized','after','state_jump','track_id','experiment_type','track'], axis = 1).as_matrix()
        list(list(diag_vals))
        diag["half"] = n
        
        
        
        # ['Native','Native_product','L2','L3','Inactive']
        

        if n ==1:    
            params = [[(0.26),(0.01)],
                       [(0.75),(0.01)],
                       [(0.99),(0.01)],
                       [(0.75),(0.26)],
                       [(0.99),(0.26)],
                       [(0.99),(0.75)]] 
            
        else:
            params = [[(0.01),(0.26)],
                       [(0.01),(0.75)],
                       [(0.01),(0.75)],
                       [(0.26),(0.75)],
                       [(0.26),(0.99)],
                       [(0.75),(0.99)]] 
                
                    
       # if name == 'Native_product':
        #    if n ==1:    
         #       params = [[(0.4),(0.15)],[(0.13),(0.05)]]
          #      
          #  else:
          #      params = [[(0.13),(0.4)],[(0.05),(0.13)]]
        #else:
        #    if n ==1:    
        #        params = [[(0.8),(0.2)],[(0.25),(0.05)]]
        #        
        #        weights_init = [[(0.4),(0.2)],[(0.2),(0.2)]]
        #        
        #    else:
        #        params = [[(0.05),(0.2)],[(0.25),(0.8)]]
        #        weights_init = [[(0.2),(0.4)],[(0.1),(0.1)]]
        #params2 =   [[(1),(1)],[(0.11),(0.1)],[(0.11),(0.1)]] 
        #m = cluster.KMeans(n_clusters = clusters,n_init = 30,tol=1e-8).fit(diag_vals)

        #m = mixture.GaussianMixture(n_components=clusters,warm_start = False, covariance_type='full').fit(diag_vals)
        #m = mixture.GaussianMixture(n_components=clusters, means_init = params,warm_start = False, covariance_type='tied').fit(diag_vals)
        m = mixture.GaussianMixture(n_components=clusters,warm_start = False, covariance_type='diag').fit(diag_vals)

        
        diag["label"] = m.predict(diag_vals)
        #diag["label"] = m.labels_
        #centers = m.cluster_centers_
        
        if n == 1:
            diag["label"] += 2
        
        
        
        
        
        
        newdiags.append(diag)

        
    df2 = pd.concat(newdiags)        
    df2["d_pos"] = df2.groupby("label")["idealized"].transform(np.mean)
    df2["d_pos_after"] = df2.groupby("idealized")["after"].transform(np.mean)
    df2 = df2.sort_values(["half", "d_pos"])
    
  
    

    #fig, ax = plt.subplots(figsize=(8, 8))         
    #ax = sns.scatterplot(x="d", y="d_after", hue="label",data=df_tmp)

    newdiags = []
    for i, diag in df2.groupby("half"):
        if i == 0:
            diag["label"] = diag["d_pos"].transform(lambda group: (group.diff() != 0).cumsum()) - 1
            diag["label"] = 2 * diag["label"]
        else:
            diag["label"] = diag["d_pos"].transform(lambda group: (group.diff() != 0).cumsum()) - 1
            diag["label"] = 2 * diag["label"] + 1
        newdiags.append(diag)

    df2 = pd.concat(newdiags)
    df2 = df2.sort_values(["label"]).reset_index()
    df2["state_grp"] = df2["label"] - df2["half"]
    
    def fast_concat(temp_list):
        """
        A faster way to make a dataframe, assuming that it's a shallow copy of ONE original dataframe.
        Note that this function loses the original indices.
    
        :param temp: a temporary list of sub-dataframes
        :return: tdp_df: a concatenated Pandas dataframe, but faster than pd.concat()
        """
        COLUMN_NAMES = temp_list[0].columns
        df_dict = dict.fromkeys(COLUMN_NAMES, [])
    
        for col in COLUMN_NAMES:
            # Use a generator to save memory
            extracted = (temp[col] for temp in temp_list)
    
            # Flatten and save to df_dict
            df_dict[col] = flatten_list(extracted)
    
        df = pd.DataFrame.from_dict(df_dict)[COLUMN_NAMES]
        del temp_list
        return df
    
    label_edits = []
    print (set(df2['label']))
    for i, grp in df2.groupby("label"):
        print (i)
        grp["in_out"] = error_ellipse(xdata = grp["idealized"].values, ydata = grp["after"].values, n_std = 5., return_ax = False, color = std_cols[i], alpha = 0.2)
        grp["label"][grp["in_out"] == False] = -1
        label_edits.append(grp)
    df2 = fast_concat(label_edits)
    
    
    
    fig,ax = plt.subplots(figsize = (5,5))
    for i, grp in df2.groupby("label"):
        e_bf, e_af = grp["idealized"], grp["after"]
        ax.scatter(e_bf, e_af, zorder = 10, s = 12, color = "black" if i == -1 else std_cols[i])
    ax = fix_ax_probs(ax,'D','D after')
    fig.tight_layout()
    fig.savefig(str(save_path+'_TDP_points_sorted_png_.png'))
    fig.savefig(str(save_path+'_TDP_points_sorted.pdf'))
    plt.clf()
    plt.close('all')

    df2 = df2[df2.in_out == True] # removing points we dont want
    
    
    cbins = np.arange(0, 7, 0.5)
    cont = countour_2d(xdata = df2["idealized"],
                            ydata = df2["after"],
                            kernel = "linear",
                            bandwidth = 0.2,
                            shade_lowest = True,
                            gridsize = 80,
                            n_colors = len(blanchard_cols),
                            bins = cbins)
    
    fig, ax =  plt.subplots(figsize=(4, 3))
    c = ax.contourf(*cont, cmap = blanchard_cmap, extend = "both")
    ax.set_ylim(-0.1,1.7)
    ax.set_xlim(-0.1,1.7)
    ax = fix_ax_probs(ax,'D','D after')
    fig.tight_layout()
    fig.colorbar(c, ax = ax, extendrect = True)
    fig.savefig(str(save_path+'_D_tdp_true_cols_sort.pdf'), bbox_inches = "tight")
    plt.close()
    df2.to_csv(str(save_path+'__TDP_labels__.csv'), header=True, index=None, sep=',', mode='w')

    bins = histbins(2, 30, 1)
    plot_pts = np.linspace(2, 30, 300)
    n_total = len(df2)
    res = []
    n = 0
    import uncertainties as un
    import uncertainties.umath as umath
    
    def single_exp_fit(x, scale):
        from scipy import signal, stats
        return stats.expon.pdf(x, loc = 0, scale = scale)
    def fit_func(x,start_val,rate):             # function to fit
        return start_val*np.exp(-rate*x)
    
    fig, axes = plt.subplots(nrows = int(clusters), ncols = 2, figsize = (12, 12))
    ax = axes.ravel()
    
    fig2, axes2 = plt.subplots(nrows = int(clusters), ncols = 2, figsize = (12, 12))
    ax2 = axes2.ravel()
    #df2 = df2[df2.lifetime<25]
    
    fig_iden,ax_iden = plt.subplots(figsize = (3,3))
    
    
    for i, grp in tqdm(df2.groupby("label")):
        if i == -1: # skip outlier labels
            continue
        
        # Convert to values for probfit
        
        lifetime = grp["lifetime"].values
        
       # if i ==0:
       #     d_mask = grp["idealized"].values
       #     d_mask = d_mask<0.03
       #     lifetime = lifetime[d_mask]
       
            
        
        #lifetime_mask = lifetime>2
        
        
        
        #lifetime_mask = lifetime<31
        #lifetime = lifetime[lifetime_mask]

        # Calculate where transitions are
        E_bf = np.mean(grp["idealized"])
        E_bf_std = np.std(grp["idealized"])

        E_af = np.mean(grp["after"])
        E_af_std = np.std(grp["after"])
        
        ax_iden.annotate(str([i]), (E_bf,E_af), horizontalalignment='right', color = 'blue')
        

        r_bf = (un.ufloat(E_bf, E_bf_std))
        r_af = (un.ufloat(E_af, E_af_std))
        
        E_bf_label = "{:.2f} $\pm$ {:.2f}".format(E_bf, E_bf_std)
        E_af_label = "{:.2f} $\pm$ {:.2f}".format(E_af, E_af_std)
        r_bf_label = "{:.2f} $\pm$ {:.2f}".format(r_bf.n, r_bf.s)
        r_af_label = "{:.2f} $\pm$ {:.2f}".format(r_af.n, r_af.s)
        
        success = 0
        
        
        
        while success != 1: # keep removing datapoints until it works
            scale, err, *_ = lh_fit(data = lifetime,
                                         f = single_exp_fit,
                                         binned_likelihood = True,
                                         scale = 2.,
                                         limit_scale = (0.1, 50.))

            tau = un.ufloat(scale[0], err[0]) /(1/sec_per_frame)
            rate = 1/tau
            if tau.s > 0.8 * tau.n:
                lifetime_cutoff -= 1
                if lifetime_cutoff < 10:
                    success += 1
                grp = grp[grp["lifetime"] <= lifetime_cutoff]
                lifetime = grp["lifetime"].values
            else:
                success += 1

        n_datapoints = len(lifetime)
        n_percent = n_datapoints / n_total * 100
        

        hist_label = r"$\tau$ (s) = {:.2f} $\pm$ {:.2f}".format(tau.n, tau.s) + \
                     "\n" + \
                     r"k = {:.2f} $\pm$ {:.2f}".format(rate.n, rate.s) + \
                     "\n" + \
                     "{} entries ({:.0f} %)".format(n_datapoints, n_percent)
        
        fig_tmp,ax_tmp = plt.subplots(figsize = (3,3))
        ax_tmp.hist(grp["lifetime"], bins = bins, normed = True, color = 'gray', alpha = 0.5, label = hist_label)
        ax_tmp.plot(plot_pts, single_exp_fit(plot_pts, scale = scale), "--", color = "red",linewidth = 3)
        ax_tmp.set_title(str(i))
        ax_tmp.set_xlim(1,25)
        ax_tmp.set_xticks(range(2,25, 6))
        ax_tmp.set_xlabel("Dwell time (frame)")
        ax_tmp.set_ylabel("Probability Density")
        ax_tmp.legend(loc = "upper left")
        leg = ax_tmp.legend(handlelength = 0, handletextpad = 0, fancybox = False)
        [item.set_visible(False) for item in leg.legendHandles]   
        plt.tight_layout()
        fig_tmp.savefig(str(save_path +str(i)+ '_Single_Lifetimes_LH.pdf'))

        fig_sda,ax_sda = plt.subplots(figsize = (3,3))
        ax_sda.hist(grp['d'],300,range = (0,1.5))
        plt.tight_layout()
        fig_sda.savefig(str(save_path +str(i)+ '_D_hist.pdf'))

        
        
        
        ax[n].hist(grp["lifetime"], bins = bins, normed = True, color = 'gray', alpha = 0.5, label = hist_label)
        ax[n].plot(plot_pts, single_exp_fit(plot_pts, scale = scale), "--", color = "red",linewidth = 3)
        ax[n].set_title(str(i))
        ax[n].set_xlim(1,25)
        ax[n].set_xticks(range(2,25, 6))
        ax[n].set_xlabel("Dwell time (frame)")
        ax[n].set_ylabel("Probability Density")
        ax[n].legend(loc = "upper left")
        leg = ax[n].legend(handlelength = 0, handletextpad = 0, fancybox = False)
        [item.set_visible(False) for item in leg.legendHandles]
   
        plt.tight_layout()
        fig.savefig(save_path + '_Lifetimes_LH.pdf')
        
        
        import iminuit
        y_vals,edges = np.histogram(lifetime, bins = 30,range=(0.5,30.5))
        err = np.sqrt(y_vals)/(sum(y_vals))
        err = err+0.0001
        y_vals,edges = np.histogram(lifetime, bins = 30,range=(0.5,30.5),normed = True)
        
        x_values = np.arange(1,31,1)
        x_values_plot = np.arange(1,31,0.01)
       
        chi2 = Chi2Regression(fit_func,x_values,y_vals, error = err)
        m = iminuit.Minuit(chi2,rate=5.,start_val=20.)
                               
        m.migrad()
        
        tau_chi2 = m.values['rate'] *(1/0.097)
        
        tau_er_chi2 = m.errors['rate']*(1/0.097)
        
        aa_chi2 = (un.ufloat(tau_chi2,tau_er_chi2))
        tau_chi2= 1/aa_chi2
        #tau_chi2 = (un.ufloat(tau_chi2,tau_er_chi2))
        #aa_chi2 = 1/tau_chi2
        
        hist_label2 = r"$\tau$ (s) = {:.2f} $\pm$ {:.2f}".format(tau_chi2.n, tau_chi2.s) + \
                     "\n" + \
                     r"k = {:.2f} $\pm$ {:.2f}".format(aa_chi2.n, aa_chi2.s) + \
                     "\n" + \
                     "{} entries ({:.0f} %)".format(n_datapoints, n_percent)
        
        
        #ax2[n].bar(x_values,y_vals, color = 'grey', alpha = 0.5, label = hist_label)
        ax2[n].plot(x_values,fit_func(x_values,*m.args),'r--',linewidth = 3)
        ax2[n].hist(grp["lifetime"], bins = bins, normed = True, color = 'gray', alpha = 0.5, label = hist_label2)
        ax2[n].set_title(str(i))
        ax2[n].set_xticks(range(2,30, 6))
        ax2[n].set_xlabel("Dwell time (frame)")
        ax2[n].set_ylabel("Probability Density")
        ax2[n].legend(loc = "upper left")
        leg2 = ax2[n].legend(handlelength = 0, handletextpad = 0, fancybox = False)
        [item.set_visible(False) for item in leg2.legendHandles]
        #plt.tight_layout()
        fig2.savefig(save_path + '_Lifetimes_ch2.pdf')
        
        
        fig_tmp,ax_tmp = plt.subplots(figsize = (3,3))
        ax_tmp.hist(grp["lifetime"], bins = bins, normed = True, color = 'gray', alpha = 0.5, label = hist_label2)
        ax_tmp.plot(x_values,fit_func(x_values,*m.args),'r--',linewidth = 3)
        ax_tmp.set_title(str(i))
        ax_tmp.set_xlim(1,31)
        ax_tmp.set_xticks(range(2,30, 6))
        ax_tmp.set_xlabel("Dwell time (frame)")
        ax_tmp.set_ylabel("Probability Density")
        ax_tmp.legend(loc = "upper left")
        leg = ax_tmp.legend(handlelength = 0, handletextpad = 0, fancybox = False)
        [item.set_visible(False) for item in leg.legendHandles]   
        #plt.tight_layout()
        fig_tmp.savefig(str(save_path +str(i)+ '_Single_Lifetimes_chi2.pdf'))
        
        plt.clf()
        plt.close('all')
        n += 1
        kB = 1.38065E-23
        h = 6.63E-34
        R = 8.3145e-3  # in kJ/mol
        T = 298
        
        k_kbt = -umath.log((h * rate) / (kB * T))
        k_kcal  = -umath.log((h * rate) / (kB * T)) * (R * T)
        from uncertainties import unumpy as unp
        
        results = pd.DataFrame(
                {"D"            : np.array(E_bf),
                 "D_after"      : np.array(E_af),
                 "r_before"     : np.array(r_bf.n),
                 "r_after"      : np.array(r_af.n),
                 "k"            : np.array(rate.n),
                 "k_err"        : np.array(rate.s),
                 "k_chi2"       : np.array(aa_chi2.n),
                 "k_chi2_err"   : np.array(aa_chi2.s),
                 "tau_chi2"     : np.array(tau_chi2.n),
                 "tau_chi2_err" : np.array(tau_chi2.s),
                 "tau"          : np.array(tau.n),
                 "tau_err"      : np.array(tau.s),
                 "percent"      : np.array(n_percent),
                 "D_std"        : np.array(E_bf_std),
                 "D_after_std"  : np.array(E_af_std),
                 "r_before_std" : np.array(r_bf.s),
                 "r_after_std"  : np.array(r_af.s),
                 "cluster_label": [i],
                 "cluster_half" : np.array(grp["half"].values[0]),
                 "state_grp"    : np.array(grp["state_grp"].values[0]),
                 "Percent_ts"   : np.array(n_percent),
                 "k_kbt"        : np.array(k_kbt),
                 "k_kcal"       : np.array(k_kcal)})

        
        res.append(results)
    ax_iden.set_xlim(-0.1,1)
    ax_iden.set_ylim(-0.1,1)
    fig_iden.tight_layout()
    fig_iden.savefig(str(save_path+'identifier.pdf'))
    res = pd.concat(res, ignore_index = True)
    res.to_csv(str(save_path+'__results__.csv'), header=True, index=None, sep=',', mode='w')
    D_tmp       = res['D'].tolist()
    D_After_tmp = res['D_after'].tolist()
    label_tmp = res['cluster_label'].tolist()
    
    fig,ax = plt.subplots(figsize = (3,3))
    for i in range(len(D_tmp)):
        ax.annotate(label_tmp[i], (D_tmp[i],D_After_tmp[i]), horizontalalignment='right', color = 'blue')
    fig.tight_layout()
    fig_iden.savefig(str(save_path+'identifier.pdf'))
    plt.close('all')
    
    rate_list = res['k'].tolist()
    
    k1      =rate_list[0]
    k1_1    =rate_list[1]
    K1      =k1/k1_1
    
    k2      =rate_list[2]
    k2_1    =rate_list[3]
    K2      =k2/k2_1
    
    

    
    E1 = 1./(1.+K1+K1*K2)
    E2 = K1/(1.+K1+K1*K2)
    E3 = (K1*K2)/(1.+K1+K1*K2)

    def free_E(K_eq):
        R = 8.3145e-3  # in kJ/mol
        Temp = 298
        print ("∆G ", -R*Temp*unp.log(K_eq))
        
    
    with open(save_path+'Data__.txt', "w") as text_file:
        text_file.write("E1 E2 E3" + "\n")
        text_file.write(str(E1) +"  "+ str(E2)+"  "+ str(E3))
        text_file.write("\n\n\n")
        text_file.write("Delta G1     Delta G2")
        text_file.write(str(free_E(K1)) + str(free_E(K2)))

def single_exp_fit(x, scale):
    from scipy import signal, stats
    return stats.expon.pdf(x, loc = 0, scale = scale)



def double_exp_fit(x, scale,scale2,alpha):
    from scipy import signal, stats
    return stats.expon.pdf(x, loc = 0, scale = scale)*alpha+stats.expon.pdf(x, loc = 0, scale = scale2)*(1-alpha)

def read_simulated_data(main_folder_path):
    from glob import glob
    files = glob(str(main_folder_path+'*.txt'))
    
    traces = []
    for filepath in files:
        if filepath.find('trace')      != -1:
            traces.append(filepath)
        
        
    df_r = pd.DataFrame()  
    for file_path,name in tqdm(zip(traces,range(len(traces)))):
        trace_tmp =  pd.read_csv(file_path, low_memory=False, sep = '\t')
        trace_tmp['particle'] = int(name)
        df_r = df_r.append(trace_tmp)
        
    df_r = df_r.rename(columns={'FRET E': 'FRET_E'})
    df_r = df_r.rename(columns={'%t (s)': 'time'})    
    return df_r

def read_exp_data(main_folder_path):
    from glob import glob
    traces = glob(str(main_folder_path+'*.dat'))
        
    df_r = pd.DataFrame()  
    for file_path,name in tqdm(zip(traces,range(len(traces)))):
        trace_tmp =  pd.read_csv((file_path), sep=r'\s{2,}', engine='python',header = None)
        trace_tmp['particle'] = str(file_path[55:])
        df_r = df_r.append(trace_tmp)
        
    df_r = df_r.rename(columns={0: 'time'})
    df_r = df_r.rename(columns={1: 'donor'})
    df_r = df_r.rename(columns={2: 'acceptor'}) 
    df_r["FRET_E"] = (df_r['acceptor'])/(df_r['donor']+df_r['acceptor'])
    df_r= df_r[df_r['time'] <50]

    return df_r


if simulated_data:
    #get dataframe
    df_t = read_simulated_data(simul_data_path_training)
    #train model with n parameters
    model,bic,log = get_model_from_dataframe(df_t,n_states,save_path,int(1+2*int(n_states)))
    print (str("Model BIC: "+str(bic)))
    
    #load challenge set
    df_c = read_simulated_data(simul_data_path_challenge)
    #fit hmm model to dataset
    small,full = run_hmm_treat(df_c,model,save_path)
    #save output
    

    #prep df for tdp
    full['track'] = full['track_id']
    full['lifetime'] = full.groupby(['track_id','unique_state_ID'])['unique_state_ID'].transform(len)
    full.to_csv(str(save_path+'full_hmm_treat.csv'), header=True, index=None, sep=',', mode='w') 

if fit_tdp_simul:
    df_tmp = pd.read_csv(path_to_hmm_treated_challenge, low_memory=False, sep = ',')
    #make clutering and save
    create_tdf_file(df_tmp,'test',save_path,int(n_clusters))

if experimental_sets:
    # get general model
    model = load_model(path_to_general_model)
    df_r = read_exp_data(exp_data_folder_path)
    
    #train on desired dataset
    data = np.asarray(df_r['FRET_E'].tolist())
    data = data.reshape(1, -1)
    data_to_fit = [] 
    group_all = df_r.groupby('particle')
    for name, group in group_all:
        data_to_fit.append(np.asarray(group.FRET_E.tolist()))
    model.fit(data_to_fit)
    model.bake()
    
    #save it
    t = model.to_json()
    import json
    with open(str(save_path+'data.json'), 'w', encoding='utf-8') as f:
        json.dump(t, f)
    
    
    #run hmm
    small,full = run_hmm_treat(df_r,model,save_path)
    full['track'] = full['track_id']
    full['lifetime'] = full.groupby(['track_id','unique_state_ID'])['unique_state_ID'].transform(len)
    
    full['track'] = full['track_id']
    full['lifetime'] = full.groupby(['track_id','unique_state_ID'])['unique_state_ID'].transform(len)
    full= full[full['idealized'] <1]
    full= full[full['after'] <1]
    full= full[full['idealized'] >0]
    full= full[full['after'] >0]
    full.to_csv(str(save_path+'full_hmm_treat.csv'), header=True, index=None, sep=',', mode='w') 
if fit_tdp_exp:
    df_tmp = pd.read_csv(path_to_hmm_treated_challenge, low_memory=False, sep = ',')
    #make clutering and save
    create_tdf_file(df_tmp,'test',save_path,int(n_clusters))




# oct 2020
    
main_folder_path = '/Users/sorensnielsen/Documents/Group stuff/FRET challenge/Oct 2020/expDatasets2/Traces EG 1ms/'

test_save ='/Users/sorensnielsen/Documents/Group stuff/FRET challenge/Oct 2020/expDatasets2/1 ms hmm_/20201111 3 states/'
from glob import glob
files = glob(str(main_folder_path+'*.txt'))

traces = []
names = []
for filepath in files:
    if filepath.find('Readme')      == -1:
        traces.append(filepath)
        names.append(filepath[len(filepath)-7:len(filepath)-4])
        
    
    
df_r = pd.DataFrame()  
for file_path,name in tqdm(zip(traces,names)):
    trace_tmp =  pd.read_csv(file_path, low_memory=False,header = None, sep = '\t')
    
    trace_tmp = trace_tmp.rename(columns={0: 'time'})
    trace_tmp = trace_tmp.rename(columns={1: 'donor'})
    trace_tmp = trace_tmp.rename(columns={2: 'acceptor'})     
    
#    donor = trace_tmp['donor'].tolist()
#    acceptor = trace_tmp['acceptor'].tolist()
#
#    donor = [sum(donor[i:i+2]) for i in range(0, len(donor), 2)]
#    acceptor = [sum(acceptor[i:i+2]) for i in range(0, len(acceptor), 2)]
#    time = np.arange(len(donor))
#    trace_tmp = pd.DataFrame(
#        {'time'   : time,
#         'donor'  : donor,
#         'acceptor'   : acceptor})
    trace_tmp['particle'] = int(name)
    df_r = df_r.append(trace_tmp)


df_r['FRET_E'] = df_r.acceptor.values/(df_r.acceptor.values+df_r.donor.values)


fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(df_r['FRET_E'].values,100,range = (0,1),density = True, color ='royalblue',ec = "black" ,label = 'E FRET 2ms')
ax.legend()
ax = fix_ax_probs(ax,'E FRET','Density')
fig.tight_layout()
fig.savefig(str(test_save+'E_FRET.pdf'))

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(df_r['donor'].values,20,range = (0,20),density = True, color ='seagreen',ec = "black" ,label = 'Donor 2ms')
ax.legend()
ax = fix_ax_probs(ax,'donor','Density')
fig.tight_layout()
fig.savefig(str(test_save+'Donor.pdf'))

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(df_r['acceptor'].values,20,range = (0,20),density = True, color ='firebrick',ec = "black" ,label = 'Acceptor 2ms')
ax.legend()
ax = fix_ax_probs(ax,'acceptor','Density')
fig.tight_layout()
fig.savefig(str(test_save+'Acceptor.pdf'))

grp = df_r.groupby('particle')


for name, df in tqdm(grp):
    t = df.time.values
    t = t-t[0]
    fig,ax = plt.subplots(2,1,figsize = (5,2.5))
    ax[0].plot(t[:200],df.donor.values[:200],color = 'seagreen',alpha = 0.8,label = "Donor")
    ax[0].plot(t[:200],df.acceptor.values[:200],color = 'firebrick',alpha = 0.8,label = "Acceptor")
    ax[0].legend()
    ax[0] = fix_ax_probs(ax[0],'','Density')
    
    
    ax[1].plot(t[:200],df.FRET_E.values[:200],color = 'royalblue',alpha = 0.8,label = "E FRET")
    ax[1].set_ylim(0,1)
    ax[1].legend()
    ax[1] = fix_ax_probs(ax[1],'time [ms]','Density')
    

    fig.tight_layout()
    fig.savefig(str(test_save+str(name)+'.pdf'))
    plt.close('all')

save_path = '/Users/sorensnielsen/Documents/Group stuff/FRET challenge/Oct 2020/expDatasets2/training plots/traces_10ms/final 10 ms/'


model,bic,log = get_model_from_dataframe(df_r,3,test_save,4)

small,big = run_hmm_treat(df_r,model,test_save)

create_tdp_basic(small,test_save)
create_tdf_file(small,'4states',test_save,1)

save_path = '/Users/sorensnielsen/Documents/Group stuff/FRET challenge/Oct 2020/expDatasets2/training plots/hmm 1 ms/final 2 state/'

model,bic,log = get_model_from_dataframe(df_r,2,save_path,4)
small,big = run_hmm_treat(df_r,model,save_path)
create_tdf_file(small,'2states',test_save,3)


def plot_big(big,test_save):
    grp = big.groupby('track_id')
    for name, df in tqdm(grp):
        fig,ax = plt.subplots(4,1,figsize=(6, 10))   
        ax[0].plot(df['step'].values[:300], "gray", linewidth = 1,alpha = 0.8)
        ax[0].plot(df['idealized'].values[:300], "firebrick",linewidth = 1,alpha = 0.8)
        ax[0].set_ylabel("FRET")
        ax[0].set_xlabel("t")
        ax[0].set_ylim(0,1)
        ax[0].tick_params(axis = 'both', which = 'major', labelsize = 14)
        ax[0].tick_params(axis = 'both', which = 'minor', labelsize = 14)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].grid(False)
        
        ax[1].plot(df['step'].values[300:600], "gray", linewidth = 1,alpha = 0.8)
        ax[1].plot(df['idealized'].values[300:600], "firebrick",linewidth = 1,alpha = 0.8)
        ax[1].set_ylabel("FRET")
        ax[1].set_xlabel("t")
        ax[1].set_ylim(0,1)
        ax[1].tick_params(axis = 'both', which = 'major', labelsize = 14)
        ax[1].tick_params(axis = 'both', which = 'minor', labelsize = 14)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].grid(False)
        
        ax[2].plot(df['step'].values[600:900], "gray", linewidth = 1,alpha = 0.8)
        ax[2].plot(df['idealized'].values[600:900], "firebrick",linewidth = 1,alpha = 0.8)
        ax[2].set_ylabel("FRET")
        ax[2].set_xlabel("t")
        ax[2].set_ylim(0,1)
        ax[2].tick_params(axis = 'both', which = 'major', labelsize = 14)
        ax[2].tick_params(axis = 'both', which = 'minor', labelsize = 14)
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['top'].set_visible(False)
        ax[2].grid(False)
        
        ax[3].plot(df['step'].values[900:1200], "gray", linewidth = 1,alpha = 0.8)
        ax[3].plot(df['idealized'].values[900:1200], "firebrick",linewidth = 1,alpha = 0.8)
        ax[3].set_ylabel("FRET")
        ax[3].set_xlabel("t")
        ax[3].set_ylim(0,1)
        ax[3].tick_params(axis = 'both', which = 'major', labelsize = 14)
        ax[3].tick_params(axis = 'both', which = 'minor', labelsize = 14)
        ax[3].spines['right'].set_visible(False)
        ax[3].spines['top'].set_visible(False)
        ax[3].grid(False)
        
        fig.tight_layout()    
        fig.savefig(str(test_save+'____'+str(name)+'.pdf')  ) 
        plt.clf()
        plt.close("all")


ms1 = pd.read_csv('/Users/sorensnielsen/Documents/Group stuff/FRET challenge/Oct 2020/expDatasets2/training plots/hmm 1 ms/final 2 state/__TDP_labels__.csv', low_memory=False, sep = ',')
label1 = ms1[ms1.label ==1]
label0 = ms1[ms1.label ==0]

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(label0.lifetime.values/1000,50,density = True,range = (0,2),color = "gray",ec = "black",alpha =0.8,label ="Low-high" )
ax.hist(label1.lifetime.values/1000,50,density = True,range = (0,2),color = "firebrick",ec = "black",alpha =0.8,label ="High-low" )
ax.legend()
ax.set_ylim(0,4)
ax = fix_ax_probs(ax,'lifetime [s]','Density')
fig.tight_layout()
fig.savefig(str(save_path+'1ms lifetime.pdf'))






# 10 ms fitting

save_path = '/Users/sorensnielsen/Documents/Group stuff/FRET challenge/Oct 2020/expDatasets2/finakl _plots/'
import uncertainties as un
import uncertainties.umath as umath
ms1 = pd.read_csv('/Users/sorensnielsen/Documents/Group stuff/FRET challenge/Oct 2020/expDatasets2/10 ms hmm_/__TDP_labels__.csv', low_memory=False, sep = ',')
label1 = ms1[ms1.label ==1]
label0 = ms1[ms1.label ==0]
sec_per_frame = 0.01

lifetime_0 = label0.lifetime.values
lifetime_1 = label1.lifetime.values


def single_exp_fit(x, scale):
    from scipy import signal, stats
    return stats.expon.pdf(x, loc = 0, scale = scale)

scale, err, *_ = lh_fit(data = lifetime_0,
                                         f = single_exp_fit,
                                         binned_likelihood = False,
                                         scale = 2.,
                                         limit_scale = (0.1, 50.))
plot_pts = np.linspace(0, 2, 300)
tau = un.ufloat(scale[0], err[0]) /(1/sec_per_frame)
rate = 1/tau

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(label0.lifetime.values/100,25,density = True,range = (0,2),color = "black",ec = "black",alpha =0.8,label ="Low-high" )
ax.plot(plot_pts, single_exp_fit(plot_pts, scale = scale/100), "--", color = "red",linewidth = 2)
ax.annotate(str('Rate: '+format(rate,'.2f')), (0.8,2), horizontalalignment='left', color = 'black')
ax.legend()
ax.set_ylim(0,3)
ax = fix_ax_probs(ax,'lifetime [s]','Density')
fig.tight_layout()
fig.savefig(str(save_path+'10ms lifetime fit_low-high.pdf'))


def single_exp_fit(x, scale):
    from scipy import signal, stats
    return stats.expon.pdf(x, loc = 0, scale = scale)

scale, err, *_ = lh_fit(data = lifetime_1,
                                         f = single_exp_fit,
                                         binned_likelihood = False,
                                         scale = 2.,
                                         limit_scale = (0.1, 50.))
plot_pts = np.linspace(0, 2, 300)
tau = un.ufloat(scale[0], err[0]) /(1/sec_per_frame)
rate = 1/tau

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(label1.lifetime.values/100,25,density = True,range = (0,2),color = "gray",ec = "black",alpha =0.8,label ="Low-high" )
ax.plot(plot_pts, single_exp_fit(plot_pts, scale = scale/100), "--", color = "red",linewidth = 2)
ax.annotate(str('Rate: '+format(rate,'.2f')), (0.8,2), horizontalalignment='left', color = 'black')
ax.legend()
ax.set_ylim(0,3)
ax = fix_ax_probs(ax,'lifetime [s]','Density')
fig.tight_layout()
fig.savefig(str(save_path+'10ms lifetime fit_high-low.pdf'))



fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(label0.lifetime.values/100,25,density = True,range = (0,2),color = "black",ec = "black",alpha =0.8,label ="Low-high" )
ax.hist(label1.lifetime.values/100,25,density = True,range = (0,2),color = "gray",ec = "black",alpha =0.8,label ="high-low" )
#ax.plot(plot_pts, single_exp_fit(plot_pts, scale = scale/100), "--", color = "red",linewidth = 2)
#ax.annotate(str('Rate: '+format(rate,'.2f')), (0.8,2), horizontalalignment='left', color = 'black')
plt.yscale('log')
ax.legend()
ax = fix_ax_probs(ax,'lifetime [s]','Density')
fig.tight_layout()
fig.savefig(str(save_path+'10ms lifetime fit_low-high.pdf'))



## 1 ms, two labels

ms1 = pd.read_csv('/Users/sorensnielsen/Documents/Group stuff/FRET challenge/Oct 2020/expDatasets2/1 ms hmm_/__TDP_labels__.csv', low_memory=False, sep = ',')
label1 = ms1[ms1.label ==1]
label0 = ms1[ms1.label ==0]
sec_per_frame = 0.001

lifetime_0 = label0.lifetime.values
lifetime_1 = label1.lifetime.values


def single_exp_fit(x, scale):
    from scipy import signal, stats
    return stats.expon.pdf(x, loc = 0, scale = scale)

scale, err, *_ = lh_fit(data = lifetime_0,
                                         f = single_exp_fit,
                                         binned_likelihood = False,
                                         scale = 2.,
                                         limit_scale = (0.1, 500.))
plot_pts = np.linspace(0, 2, 300)
tau = un.ufloat(scale[0], err[0]) /(1/sec_per_frame)
rate = 1/tau

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(label0.lifetime.values/1000,25,density = True,range = (0,2),color = "black",ec = "black",alpha =0.8,label ="Low-high" )
ax.plot(plot_pts, single_exp_fit(plot_pts, scale = scale/1000), "--", color = "red",linewidth = 2)
ax.annotate(str('Rate: '+format(rate,'.2f')), (0.8,2), horizontalalignment='left', color = 'black')
ax.legend()
ax.set_ylim(0,5)
ax = fix_ax_probs(ax,'lifetime [s]','Density')
fig.tight_layout()
fig.savefig(str(save_path+'1ms lifetime fit_low-high.pdf'))



def single_exp_fit(x, scale):
    from scipy import signal, stats
    return stats.expon.pdf(x, loc = 0, scale = scale)

scale, err, *_ = lh_fit(data = lifetime_1,
                                         f = single_exp_fit,
                                         binned_likelihood = False,
                                         scale = 2.,
                                         limit_scale = (0.1, 500.))
plot_pts = np.linspace(0, 2, 300)
tau = un.ufloat(scale[0], err[0]) /(1/sec_per_frame)
rate = 1/tau

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(label1.lifetime.values/1000,25,density = True,range = (0,2),color = "gray",ec = "black",alpha =0.8,label ="high-low" )
ax.plot(plot_pts, single_exp_fit(plot_pts, scale = scale/1000), "--", color = "red",linewidth = 2)
ax.annotate(str('Rate: '+format(rate,'.2f')), (0.8,2), horizontalalignment='left', color = 'black')
ax.legend()
ax.set_ylim(0,5)
ax = fix_ax_probs(ax,'lifetime [s]','Density')
fig.tight_layout()
fig.savefig(str(save_path+'1ms lifetime fit_high-low.pdf'))




# 1ms 4 states
save= '/Users/sorensnielsen/Documents/Group stuff/FRET challenge/Oct 2020/expDatasets2/1 ms hmm_/4 states/take monday/3 states/'




model,bic,log = get_model_from_dataframe(df_r,2,test_save,2)

model =load_model(str(test_save+'data.json'))


small,big = run_hmm_treat(df_r,model,test_save)
plot_big(big,str(test_save+'__'))

create_tdf_file(small,'2states',test_save,1)







tester = small[small.state ==0]

tester = tester[tester.after<0.5]
tester = tester[tester.after>0.1]


fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(tester.lifetime.values/1000,50,range = (0,50/1000),density = True,color = "gray",ec = "black",alpha =0.8,label ="high-low" )
ax.legend()
ax.set_ylim(0,800)
ax = fix_ax_probs(ax,'lifetime [s]','Density')
fig.tight_layout()
fig.savefig(str(save+'1ms lifetime 3 state high to low only.pdf'))


sec_per_frame = 0.001

def single_exp_fit(x, scale):
    from scipy import signal, stats
    return stats.expon.pdf(x, loc = 0, scale = scale)

scale, err, *_ = lh_fit(data = tester.lifetime.values,
                                         f = single_exp_fit,
                                         binned_likelihood = True,
                                         scale = 2.,
                                         limit_scale = (0.1, 1000.))
plot_pts = np.linspace(0, 50/1000, 300)
tau = un.ufloat(scale[0], err[0]) /(1/sec_per_frame)
rate = 1/tau



fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(tester.lifetime.values/1000,50,range = (0,50/1000),density = True,color = "gray",ec = "black",alpha =0.8,label ="high-low" )
ax.plot(plot_pts, single_exp_fit(plot_pts, scale = scale/1000), "--", color = "red",linewidth = 2)
ax.annotate(str('Rate: '+format(rate,'.2f')+'s$^-1$'), (0.017,500), horizontalalignment='left', color = 'black')
ax.legend()
ax.set_ylim(0,800)
ax = fix_ax_probs(ax,'lifetime [s]','Density')
fig.tight_layout()
fig.savefig(str(save+'1ms lifetime 3 state high to low only.pdf'))








tester = small[small.state ==1]

tester = tester[tester.after>0.5]

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(tester.lifetime.values/1000,50,range = (0,50/1000),density = True,color = "black",ec = "black",alpha =0.8,label ="low-high" )
ax.legend()
ax.set_ylim(0,800)
ax = fix_ax_probs(ax,'lifetime [s]','Density')
fig.tight_layout()
fig.savefig(str(save+'1ms lifetime 3 state low to high only.pdf'))


sec_per_frame = 0.001

def single_exp_fit(x, scale):
    from scipy import signal, stats
    return stats.expon.pdf(x, loc = 0, scale = scale)

scale, err, *_ = lh_fit(data = tester.lifetime.values,
                                         f = single_exp_fit,
                                         binned_likelihood = True,
                                         scale = 2.,
                                         limit_scale = (0.1, 1000.))
plot_pts = np.linspace(1, 50/1000, 300)
tau = un.ufloat(scale[0], err[0]) /(1/sec_per_frame)
rate = 1/tau

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(tester.lifetime.values/1000,50,range = (0,50/1000),density = True,color = "black",ec = "black",alpha =0.8,label ="low-high" )
ax.plot(plot_pts, single_exp_fit(plot_pts, scale = scale/1000), "--", color = "red",linewidth = 2)
ax.annotate(str('Rate: '+format(rate,'.2f')+'s${^-1}$'), (0.017,500), horizontalalignment='left', color = 'black')

ax.legend()

ax = fix_ax_probs(ax,'lifetime [s]','Density')
fig.tight_layout()
fig.savefig(str(save+'1ms lifetime 3 state low to high only.pdf'))


# fitting double exp to 1 ms 2 state models

save = test_save#/Users/sorensnielsen/Documents/Group stuff/FRET challenge/Oct 2020/expDatasets2/finakl _plots/'
df = pd.read_csv('/Users/sorensnielsen/Documents/Group stuff/FRET challenge/Oct 2020/expDatasets2/1 ms hmm_/20201111 3 states/__TDP_labels__.csv', low_memory=False, sep = ',')
set(df['label'])

df["life_after"] = np.roll(df["lifetime"], -1)

sec_per_frame = 0.001
tester = df[df.label ==0]
on = tester
def double_exp_fit(x, scale,scale2,alpha):
    from scipy import signal, stats
    return stats.expon.pdf(x, loc = 0, scale = scale)*alpha+stats.expon.pdf(x, loc = 0, scale = scale2)*(1-alpha)
scale_s, err_s, single_on_log = lh_fit(data = tester.lifetime.values,
                                         f = single_exp_fit,
                                         binned_likelihood = True,
                                         scale = 2.,
                                         limit_scale = (0.1, 1000.))
plot_pts = np.linspace(1, 2, 300)
tau_s = un.ufloat(scale_s[0], err_s[0]) /(1/sec_per_frame)
rate_s = 1/tau_s

scale, err, double_on_log = lh_fit(data = tester.lifetime.values,
                                         f = double_exp_fit,
                                         binned_likelihood = True,
                                         scale = 449.,
                                         scale2 = 177.,
                                         limit_scale = (0.1, 2000.),
                                         alpha = 0.5,
                                         limit_alpha = (0.05,0.15))
plot_pts = np.linspace(0, 2, 300)
tau1 = un.ufloat(scale[0], err[0]) /(1/sec_per_frame)
rate1 = 1/tau1

tau2 = un.ufloat(scale[1], err[1]) /(1/sec_per_frame)
rate2 = 1/tau2

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(tester.lifetime.values/1000,30,range = (0,2),density = True,color = "gray",ec = "black",alpha =0.8,)
ax.plot(plot_pts, double_exp_fit(plot_pts, scale = scale[0]/1000,scale2 =scale[1]/1000,alpha =scale[2]), "--", color = "red",linewidth = 2,label = "Double")
ax.plot(plot_pts, single_exp_fit(plot_pts, scale = scale[0]/1000)*scale[2], "--", color = "red",linewidth = 1)
ax.plot(plot_pts, single_exp_fit(plot_pts, scale = scale[1]/1000)*(1-scale[2]), "--", color = "red",linewidth = 1)

ax.annotate(str('D1: '+format(rate1,'.1f')+'s${^-1}$'+'\n'+'D2: '+format(rate2,'.1f')+'s${^-1}$'+'\n'+'$a$: '+format(scale[2],'.2f')+'\n'+'S: '+format(rate_s,'.1f')+'s${^-1}$'), (0.5,1.0), horizontalalignment='left', color = 'black')
ax.set_title("k$_{on}$" )
ax.legend()

ax = fix_ax_probs(ax,'lifetime [s]','Density')
fig.tight_layout()
fig.savefig(str(save+'___corrected_uncon____lifetime dobule exp low_to_high time 2 state.pdf'))

## off rate
tester = df[df.label ==1]
off = tester
def double_exp_fit(x, scale,scale2,alpha):
    from scipy import signal, stats
    return stats.expon.pdf(x, loc = 0, scale = scale)*alpha+stats.expon.pdf(x, loc = 0, scale = scale2)*(1-alpha)
scale_s, err_s, single_off_log = lh_fit(data = tester.lifetime.values,
                                         f = single_exp_fit,
                                         binned_likelihood = True,
                                         scale = 2.,
                                         limit_scale = (0.1, 1000.))
plot_pts = np.linspace(1, 2, 300)
tau_s = un.ufloat(scale_s[0], err_s[0]) /(1/sec_per_frame)
rate_s = 1/tau_s

scale, err, double_off_log = lh_fit(data = tester.lifetime.values,
                                         f = double_exp_fit,
                                         binned_likelihood = True,
                                         scale = 500.,
                                         scale2 = 200.,
                                         limit_scale = (0.1, 2000.),
                                         alpha = 0.4,
                                         limit_alpha = (0.85,0.95))
plot_pts = np.linspace(0, 2, 300)
tau1 = un.ufloat(scale[0], err[0]) /(1/sec_per_frame)
rate1 = 1/tau1

tau2 = un.ufloat(scale[1], err[1]) /(1/sec_per_frame)
rate2 = 1/tau2

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(tester.lifetime.values/1000,30,range = (0,2),density = True,color = "gray",ec = "black",alpha =0.8)
ax.plot(plot_pts, double_exp_fit(plot_pts, scale = scale[0]/1000,scale2 =scale[1]/1000,alpha =scale[2]), "--", color = "red",linewidth = 2,label = "Double")
ax.plot(plot_pts, single_exp_fit(plot_pts, scale = scale[0]/1000)*(scale[2]), "--", color = "red",linewidth = 1)
ax.plot(plot_pts, single_exp_fit(plot_pts, scale = scale[1]/1000)*(1-scale[2]), "--", color = "red",linewidth = 1)


ax.annotate(str('D1: '+format(rate1,'.1f')+'s${^-1}$'+'\n'+'D2: '+format(rate2,'.1f')+'s${^-1}$'+'\n'+'$a$: '+format(scale[2],'.2f')+'\n'+'S: '+format(rate_s,'.1f')+'s${^-1}$'), (0.5,1.0), horizontalalignment='left', color = 'black')
ax.set_title("k$_{off}$" )
ax.legend()

ax = fix_ax_probs(ax,'lifetime [s]','Density')
fig.tight_layout()
fig.savefig(str(save+'___corrected_uncon____lifetime dobule exp high to low time 2 state.pdf'))


fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.plot([1,2],[_bic(len(on), 1, single_on_log),_bic(len(on), 3, double_on_log)])

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.plot([1,2],[_bic(len(off), 1, single_off_log),_bic(len(off), 3, double_off_log)])


fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist2d(df['lifetime'].values/10,df['life_after'].values/10,20,range = ((0,10),(0,10)))



_bic(len(off), 3, double_off_log)
_bic(len(on), 3, double_on_log)

