#Include depedencies
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib import cm
from . import signal as sig
from . import phsTools
from scipy.interpolate import interp1d
from scipy.constants import c
import multiprocessing as mp

    
def img_plane_dict(platform, n_hat = np.array([0,0,1]), numPixels=False, \
                   length=False, upsample=False, res_factor=False,
                   aspect=False, checkme=False, force_xy=False):
    '''
    This function defines the image plane parameters.
    
    Parameters
    ----------
    platform  :  dict    
        The platform dictionary. Must include keys 'nsamples',
        'npulses', 'freq', 'R_c', 'pos' and if keyword `res_factor` is
        used 'delta_r'    
    n_hat  :  array, default=numpy.array([0,0,1])    
        Normal vector of the plane in which you wish to form the image    
    numPixels  :  int, array, tuple, or list. optional
        Allows the user to specify the number of pixels of the output image in
        range(nu) and cross range(nv) directions. int will result in a square.    
    length  :  int, array, tuple, or list. optional    
        Size (m) of scene to form image in range and cross range 
        directions (determined by center of aperture) about scene center.    
    upsample  :  bool, optional    
        Autodetermines image size by rounding nsamples and npulses up 
        to the nearest power of 2. Overridden by numPixels    
    res_factor  :  float, optional    
        A res_factor of 1 yields a (u,v) image plane whose pixels are 
        sized at the theoretical resolution limit of the system derived
        using delta_r which in turn was derived using the bandwidth.
        Using this or aspect will override `length`    
    aspect  :  float, optional    
        Using this will force a given aspect (dv/du) on the output image
        and will select the lower resolution (larger) between du and dv 
        at the given aspect. This will override length    
    checkme  :  bool, optional    
        Prints theoretical maximum scene size and resolution limits.
        Then prints chosen size and resolution limits.    
    force_xy  :  bool, optional    
        Used for comparison with matlab code. Forces u_vec=[1,0,0] and
        v_vec=[0,1,0]. Note that this overrides n_hat    
            
    Returns
    ----------
    output  :  dict    
        Contains keys: 'n_hat', 'u_hat', 'v_hat', 'du', 'dv', 'u', 'v', 'k_u',
        'k_v', and 'pixel_locs'. 'du'/'dv' (m) represents the spacing along 
        'u' and 'v' while 'u_hat' and 'v_hat' give their direction.'k_u' and 
        'k_v' (1/m) represent frequencies on the image plane. 'pixel_locs' is 
        a 3x(nu,nv) array of xyz coordinates to use for sampling locations.    
        
    '''
    
    nsamples = platform['nsamples']
    npulses  = platform['npulses']
    freq     = platform['freq']
    
    if freq.ndim ==1:
        freq=freq[None,:]
    deltaF=freq[0,2]-freq[0,1]
    
    #Import relevant platform parameters
    R_c = platform['R_c']    
    def isany(x):
        isin=isinstance
        return np.any([isin(x,list),isin(x,np.ndarray),isin(x,tuple)])

    #Define image plane parameters
    if isany(numPixels):
        nu,nv=numPixels[0],numPixels[-1]
    elif np.all([isinstance(numPixels,int),numPixels>0]):
        nu,nv=numPixels,numPixels
    elif np.all([not numPixels,upsample]):
        nu= 2**int(np.log2(nsamples)+bool(np.mod(np.log2(nsamples),1)))
        nv= 2**int(np.log2(npulses)+bool(np.mod(np.log2(npulses),1)))
    else:
        nu,nv= nsamples,npulses
    
    #Determine scene bounds
    # Determine the azimuth angles of the image pulses (radians)
    pos=platform['pos']
    AntAz = np.unwrap(np.arctan2(pos[:,1],pos[:,0]))

    # Determine the average azimuth angle step size (radians)
    deltaAz = np.abs(np.mean(np.diff(AntAz)))

    # Determine the total azimuth angle of the aperture (radians)
    totalAz = np.max(AntAz) - np.min(AntAz)

    # Determine the maximum scene size of the image (m)
    maxWr = c/(2*deltaF)
    maxWx = c/(2*deltaAz*np.mean(freq[0,:]))
    
    if isany(length):
        ulength,vlength=length[0],length[-1]
    elif np.all([isinstance(length,int),length>0]):
        ulength,vlength=length,length
    else:
        ulength,vlength=1.22*maxWr,1.22*maxWx #1.22 for Airy disk limit, will alias
    
    #Build Pixel Grid
    if res_factor: #included for compatibility
        #Define resolution.  This should be less than the system resolution limits
        res_factor=res_factor if res_factor else 1.0
        du = platform['delta_r']*res_factor*nsamples/nu
        aspect=aspect if aspect else 1.0
        dv = aspect*du
        #Define range and cross-range locations
        u = np.arange(-nu/2, nu/2)*du
        v = np.arange(-nv/2, nv/2)*dv
        ulength,vlength=np.abs(u[-1]-u[0]),np.abs(v[-1]-v[0])
    elif np.all([not res_factor, aspect]):
        ubound = 0.5*ulength*np.array([-1, 1]);
        vbound = 0.5*vlength*np.array([-1, 1]);
        du=np.max([ulength/nu,vlength/(aspect*nv)])
        dv=aspect*du
        #Define range and cross-range locations
        u = np.arange(-nu/2, nu/2)*du
        v = np.arange(-nv/2, nv/2)*dv
        ulength,vlength=np.abs(u[-1]-u[0]),np.abs(v[-1]-v[0])
    else:
        ubound = 0.5*ulength*np.array([-1, 1]);
        vbound = 0.5*vlength*np.array([-1, 1]);
        u = np.linspace(ubound[0],ubound[1],nu);
        v = np.linspace(vbound[0],vbound[1],nv);  
        du=u[2]-u[1]
        dv=v[2]-v[1]

    if checkme:
        # Determine the resolution of the image (m)
        dr = c/(2*deltaF*nsamples);
        dx = c/(2*totalAz*np.mean(freq[0,:]));
        
        # Display maximum scene size and resolution
        print('Maximum Scene Size:  %.4f m range, %.4f m cross-range'%(maxWr,maxWx))
        print('Max Resolution:  %.4fm range, %.4f m cross-range'%(dr,dx))
        print('Chosen Size: %.4f m range %.4f m cross-range' %(ulength,vlength))
        print('Chosen Sample Spacing: %.4f m range, %.4f m cross-range'%(du,dv))

    #Derive image plane spatial frequencies
    k_u = 2*np.pi*np.linspace(-1.0/(2*du), 1.0/(2*du), nu)
    k_v = 2*np.pi*np.linspace(-1.0/(2*dv), 1.0/(2*dv), nv)
    
    if force_xy:
        u_hat=np.array([1,0,0])
        v_hat=np.array([0,1,0])
    else:
        #Derive representation of u_hat and v_hat in (x,y,z) space
        v_hat = np.cross(n_hat, R_c)/np.linalg.norm(np.cross(n_hat, R_c))
        u_hat = np.cross(v_hat, n_hat)/np.linalg.norm(np.cross(v_hat, n_hat))

    #Represent u and v in (x,y,z)
    [uu,vv] = np.meshgrid(u,v)
    uu = uu.flatten(); vv = vv.flatten()

    A = np.asmatrix(np.hstack((
        np.array([u_hat]).T, np.array([v_hat]).T 
            ))) 
    b = np.asmatrix(np.vstack((uu,vv)))
    pixel_locs = np.asarray(A*b)

    #Construct dictionary and return to caller
    img_plane =\
    {
    'n_hat'     :   n_hat,
    'u_hat'     :   u_hat,
    'v_hat'     :   v_hat,
    'du'        :   du,
    'dv'        :   dv,
    'u'         :   u,
    'v'         :   v,
    'k_u'       :   k_u,
    'k_v'       :   k_v,
    'pixel_locs':   pixel_locs # 3 x N_pixel array specifying x,y,z location
                               # of each pixel
    }

    return(img_plane)

def subaperture(phs,platform,pulses=False,angle=False,keep_R_c=False):
    '''
    Separates a phase history and associated platform dictionary into multiple 
    segments called 'subapertures'. Note: if the last subaperture does not fill
    the given angle or pulses requirement it is discarded.

    Parameters
    ----------
    phs : ndarray
        Phase history of shape (npulses,nsamples)
    platform : dict
        Must include at a minimum 'npulses' and 'pos'. All other keys which have 
        length the same as 'npulses' will be segmented together. All keys which do
        not (with exception of 'R_c') will be copied to each subaperture.
    pulses : int, optional
        Number of pulses to use for each subaperture. Simply performs a slice
        with the same number of pulses in each aperture. Default is false.
    angle : float, optional
        Minimum subtended angle (degrees) for each subaperture. This first 
        calculates the change in angle between pulses then groups pulses with a 
        cumulative sum between integer multiples of the input angle plus the average
        change in angle.
    keep_R_c: bool, optional
        If `True`, each subaperture will be given the same 'R_c' as the original
        platform dictionary. Otherwise it is calculated for each subaperture.
        
    Returns
    ----------
    phs_list : list of arrays
        Phase history for each subaperture.
    platform_list : list of dicts
        Platform dictionary for each subaperture.
    '''
    
    
    npulses = platform['npulses']
    pos     = platform['pos']

    phs_list=[]
    platform_list=[]
    keys=platform.keys()
    keys=[key for key in keys if isinstance(platform[key],(list,np.ndarray))]
    keys=[key for key in keys if len(platform[key])==npulses]
    keys.remove('pos')
    keys.remove('k_y')
    if pulses:
        num_cuts=npulses // pulses
        starts=pulses*np.arange(num_cuts)

    elif angle:
        # Determine the azimuth angles of the image pulses (degrees)
        AntAz = np.unwrap(np.arctan2(pos[:,1],pos[:,0]))*180/np.pi
        starts=[]
        mark_start=True
        for i,az in enumerate(AntAz):
            if mark_start:
                starts.append(i)
                mark_start=False
            if np.abs(az-AntAz[starts[-1]])>=angle:
                mark_start=True #this will mark the next pulse

    for i in range(len(starts)-1):
        phs_list.append(phs[starts[i]:starts[i+1]])
        platform_list.append(platform.copy())
        pos_temp=pos[starts[i]:starts[i+1]]
        npul=starts[i+1]-starts[i]
        if np.mod(pulses,2)>0:
            R_c_temp = pos_temp[npul//2]
        else:
            R_c_temp = np.mean(pos_temp[npul//2-1:npul//2+1],axis = 0)

        L = np.linalg.norm(pos_temp[-1]-pos_temp[0])
        k_y = np.linspace(-npul/2,npul/2,npul)*2*np.pi/L

        platform_temp={
            'pos' : pos_temp,
            'R_c' : platform['R_c'] if keep_R_c else R_c_temp,
            'npulses' : npul,
            'k_y' : k_y
        }
        for key in keys:
            platform_temp[key]=platform[key][starts[i]:starts[i+1]]
        platform_list[-1].update(platform_temp)

    return phs_list,platform_list

def phs_inscribe(img):
##############################################################################
#                                                                            #
#  This program enables a user to inscribe data from a signal's spectrum.    #
#  The user clicks and drags from the top-left to bottom-right and the       #
#  corner points are output.  OpenCV is required.                            #
#                                                                            #
##############################################################################
    import cv2
    
    #Define behavior for mouse callback function
    def onMouse(event,x,y, flags, param):
        pos1    = param[0]
        pos2    = param[1]
        img_new = param[2]
        
        if event == cv2.EVENT_LBUTTONDOWN:
            param[0] = (x,y)
            pos1     = param[0]
            
        elif event == cv2.EVENT_LBUTTONUP:
            param[1] = (x,y)
            pos2     = param[1]
            img_out = img_new[pos1[1]:pos2[1],
                              pos1[0]:pos2[0]]
            cv2.imshow('win', img_out)
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            img_out = img_new
            cv2.imshow('win', img_out)
        
    #Create inscribe function
    print('\
Click and drag from top-left to bottom-right\n\
to inscribe phase history \n\n\
Right-click to reset image\n\
Press enter when finished\
         ')
    
    #Scale intensity values for an 8-bit display
    img_new = img-img.min()
    img_new = img_new/img_new.max()
    
    #Confortably size window for a 1920 x 1080 display
    rows = int(1080/2)
    cols = int(1920/2)    
    
    cv2.namedWindow('win', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('win', cols, rows)
    
    #Install mouse callback
    pos1 = (0,0)
    pos2 = (-1,-1)
    params = [pos1, pos2, img_new]
    cv2.setMouseCallback('win',onMouse, param = params)
    
    #Display image
    cv2.imshow('win',img_new)
    
    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            cv2.destroyAllWindows()
            break

    return(params[0:2])


def polar_format(phs, platform, img_plane, taylor = 20, prnt=100):
##############################################################################
#                                                                            #
#  This is the Polar Format algorithm.  The phase history data as well as    #
#  platform and image plane dictionaries are taken as inputs.                #
#                                                                            #
#  The phase history data is collected on a two-dimensional surface in       #
#  k-space.  For each pulse, a strip of this surface is collected.  The      #
#  first step in this program is to project each strip onto the (ku,kv)      #
#  plane defined by the normal vector contained in the image plane           #
#  dictionary.  This will result in data that is unevenly spaced in (ku,kv). # 
#  This unevenly spaced data is interpolated onto an evenly spaced (ku,kv)   #
#  grid defined in the image plane dictionary.  The interpolation is done    #
#  along the radial direction first, then along the along-track direction.   #
#  Further details of this method are given in both the Jakowitz and Carrera #
#  texts.                                                                    #
#                                                                            #
##############################################################################

    #Retrieve relevent parameters
    c           =   299792458.0
    npulses     =   platform['npulses']
    f_0         =   platform['f_0']
    pos         =   np.asarray(platform['pos'])
    k           =   platform['k_r']
    R_c         =   platform['R_c']
    n_hat       =   img_plane['n_hat']
    k_ui        =   img_plane['k_u']
    k_vi        =   img_plane['k_v']

    #Compute k_xi offset
    psi = np.pi/2-np.arccos(np.dot(R_c,n_hat)/np.linalg.norm(R_c))
    k_ui = k_ui + 4*np.pi*f_0/c*np.cos(psi)

    #Compute number of samples in scene
    nu = k_ui.size
    nv = k_vi.size

    #Compute x and y unit vectors. x defined to lie along R_c.
    #z = np.cross(vec[0], vec[-1]); z =z/np.linalg.norm(z)
    u_hat = (R_c-np.dot(R_c,n_hat)*n_hat)/\
            np.linalg.norm((R_c-np.dot(R_c,n_hat)*n_hat))
    v_hat = np.cross(u_hat,n_hat)

    #Compute r_hat, the diretion of k_r, for each pulse
    r_norm = np.linalg.norm(pos,axis=1)
    r_norm = np.array([r_norm]).T
    r_norm = np.tile(r_norm,(1,3))

    r_hat = pos/r_norm

    #Convert to matrices to make projections easier
    r_hat = np.asmatrix(r_hat)
    u_hat = np.asmatrix([u_hat])
    v_hat = np.asmatrix([v_hat])

    k_matrix = np.asmatrix(k)

    #Compute kx and ky meshgrid
    ku = r_hat*u_hat.T*k_matrix; ku = np.asarray(ku)
    kv = r_hat*v_hat.T*k_matrix; kv = np.asarray(kv)

    #Create taylor windows
    win1 = sig.taylor(int(phs.shape[1]), S_L = taylor)
    win2 = sig.taylor(int(phs.shape[0]), S_L = taylor)

    #Radially interpolate kx and ky data from polar raster
    #onto evenly spaced kx_i and ky_i grid for each pulse
    rad_interp = 0j*np.zeros([npulses,nu])
    ky_new = np.zeros([npulses,nu])
    for i in range(npulses):
        if prnt:
            if i%prnt==0:
                print('range interpolating for pulse %i'%(i+1))
        rad_interp[i,:] = np.interp(k_ui, ku[i,:], phs[i,:]*win1,
            left = 0, right = 0)
        ky_new[i,:] = np.interp(k_ui, ku[i,:], kv[i,:])  

    #Interpolate in along track direction to obtain polar formatted data
    polar = 0j*np.zeros([nv,nu])
    isSort = (ky_new[npulses//2, nu//2] < ky_new[npulses//2+1, nu//2])
    if isSort:
        for i in range(nu):
            if prnt:
                if i%prnt==0:
                    print('cross-range interpolating for sample %i'%(i+1))
            polar[:,i] = np.interp(k_vi, ky_new[:,i], rad_interp[:,i]*win2,
                left = 0, right = 0)
    else:
        for i in range(nu):
            if prnt:
                if i%prnt==0:
                    print('cross-range interpolating for sample %i'%(i+1))
            polar[:,i] = np.interp(k_vi, ky_new[::-1,i], 
                rad_interp[::-1,i]*win2, left = 0, right = 0)  
            
    phs_polar=np.nan_to_num(polar)

    img = np.abs(sig.ft2(phs_polar))

    return(img)


def omega_k(phs, platform, taylor = 20, upsample = 6):
##############################################################################
#                                                                            #
#  This is an omega-k algorithm based off of the algorithm prescribed in the #
#  Carrera text.  Only the phase history and platform files are taken as     #
#  inputs, an img_plane dictionary is not required.                          #
#                                                                            #
#  The input phase history needs to have been demodulated to a fixed         #
#  reference.  If demodulated to scene center, the included phs_const_ref    #
#  file can do the conversion for you.  A straight line flight path is       #
#  also assumed.                                                             #
#                                                                            #
#  The first step in the algorithm is to perform a 1D FT along azimuth.      #
#  A matched filter is applied to the resultant data to perfectly compensate #
#  the range curvature of all scatterers having minimum range R_s.  The      #
#  default setting for R_s is the minimum range of scene center.  To         #
#  correct the range curvature for other scatterers, the data is mapped      #
#  onto a new grid.                                                          #
#                                                                            #
##############################################################################

    #Retrieve relevent parameters
    K_r     =   platform['k_r']
    K_y     =   platform['k_y']
    R_s     =   np.linalg.norm(platform['pos'], axis = -1).min()
    nsamples=   platform['nsamples']
    npulses =   platform['npulses']
    
    #Take azimuth FFT
    S_Kx_Kr = sig.ift(phs, ax = 0)
    
    #Create K_r, K_y grid
    [K_r, K_y] = np.meshgrid(K_r, K_y)
    
    #Mathed filter for compensating range curvature
    phase_mf = -R_s*K_r + R_s*np.sqrt(K_r**2-K_y**2)
    phase_mf = np.nan_to_num(phase_mf)
    S_Kx_Kr_mf = S_Kx_Kr*np.exp(1j*phase_mf)
    
    #Stolt interpolation
    K_xi_max = np.nan_to_num(np.sqrt(K_r**2-K_y**2)).max()
    K_xi_min = np.nan_to_num(np.sqrt(K_r**2-K_y**2)).min()
    K_xi = np.linspace(K_xi_min, K_xi_max, nsamples)
    
    S = np.zeros([npulses,nsamples])+0j
    for i in range(npulses):
        K_x = np.nan_to_num(np.sqrt(K_r[i,:]**2-K_y[i,:]**2))
        
        f_real = interp1d(K_x, S_Kx_Kr_mf[i,:].real, kind = 'linear', bounds_error = 0, fill_value = 0)
        f_imag = interp1d(K_x, S_Kx_Kr_mf[i,:].imag, kind = 'linear', bounds_error = 0, fill_value = 0)
        
        S[i,:] += f_real(K_xi) + 1j*f_imag(K_xi)
    
    S = np.nan_to_num(S)
    [p1,p2] = phs_inscribe(np.abs(S))
    S_new = S[p1[1]:p2[1],
              p1[0]:p2[0]]
    
    #Create window
    win_x = sig.taylor(S_new.shape[1],taylor)
    win_x = np.tile(win_x, [S_new.shape[0],1])
    
    win_y = sig.taylor(S_new.shape[0],taylor)
    win_y = np.array([win_y]).T
    win_y = np.tile(win_y, [1,S_new.shape[1]])
    
    win = win_x*win_y
    
    #Apply window
    S_win = S_new*win
    
    #Pad Spectrum
    length = 2**(int(np.log2(S_new.shape[0]*upsample))+1)
    pad_x = length-S_win.shape[1]
    pad_y = length-S_win.shape[0]
    S_pad = np.pad(S_win,((pad_y//2, pad_y//2),(pad_x//2,pad_x//2)), mode = 'constant')
    
    img = sig.ift2(S_pad)
    
    return(img)
    
    
def backprojection( phs, platform, img_plane, taylor = 20, upsample = 6, prnt = False):
    ##############################################################################
    #                                                                            #
    #  This is the Backprojection algorithm.  The phase history data as well as  #
    #  platform and image plane dictionaries are taken as inputs.  The (x,y,z)   #
    #  locations of each pixel are required, as well as the size of the final    #
    #  image (interpreted as [size(v) x size(u)]).                               #
    #                                                                            #
    ##############################################################################

    #Retrieve relevent parameters
    nsamples    =   platform['nsamples']
    npulses     =   platform['npulses']
    k_r         =   platform['k_r']
    pos         =   platform['pos']
    delta_r     =   platform['delta_r']
    u           =   img_plane['u']
    v           =   img_plane['v']
    r           =   img_plane['pixel_locs']

    #Derive parameters
    nu = u.size
    nv = v.size
    k_c = k_r[nsamples//2]

    #Create window
    win_x = sig.taylor(nsamples,taylor)
    win_x = np.tile(win_x, [npulses,1])

    win_y = sig.taylor(npulses,taylor)
    win_y = np.array([win_y]).T
    win_y = np.tile(win_y, [1,nsamples])

    win = win_x*win_y

    #Filter phase history    
    filt = np.abs(k_r)
    phs_filt = phs*filt*win

    #Zero pad phase history
    N_fft = 2**(int(np.log2(nsamples*upsample))+1)
    phs_pad = sig.pad(phs_filt, [npulses,N_fft])

    #Filter phase history and perform FT w.r.t t
    Q = sig.ft(phs_pad)    
    dr = np.linspace(-nsamples*delta_r/2, nsamples*delta_r/2, N_fft)

    #Perform backprojection for each pulse
    img = np.zeros(nu*nv)+0j
    for i in range(npulses):
        if prnt:
            if np.all([prnt,i%prnt==0]):
                print("Calculating backprojection for pulse %i" %i)
        r0 = np.array([pos[i]]).T
        dr_i = np.linalg.norm(r0)-np.linalg.norm(r-r0, axis = 0)

        Q_hat = np.interp(dr_i, dr, Q[i])     
        img += Q_hat*np.exp(-1j*k_c*dr_i)

    r0 = np.array([pos[npulses//2]]).T
    dr_i = np.linalg.norm(r0)-np.linalg.norm(r-r0, axis = 0)
    img = img*np.exp(1j*k_c*dr_i)   
    img = np.reshape(img, [nv, nu])[::-1,:]
    return(img)

def DSBP(phs, platform, img_plane, center=None, size=None, derate = 1.05, taylor = 20, n = 32, beta = 4, cutoff = 'nyq', factor_max = 6, factor_min = 0):
##############################################################################
#                                                                            #
#  This is the Digital Spotlight Backprojection algorithm based on K. Dungan #
#  et. al.'s 2013 SPIE paper.                                                #
#                                                                            #
##############################################################################

    #Retrieve relevent parameters
    c           =   299792458.0
    pos         =   platform['pos']
    freq        =   platform['freq']
    u           =   img_plane['u']
    v           =   img_plane['v']
    du          =   img_plane['du']
    dv          =   img_plane['dv']  
    p           =   img_plane['pixel_locs']	
    
    #Derive parameters
    if center == None:
        empty_arg = True
        size = [0,0];
        size[1] = len(u)
        size[0] = len(v)
        Vx = u.max()-u.min()
        Vy = v.max()-v.min()
        center = np.mean(p, axis=-1)
        phs = phsTools.reMoComp(phs, platform, center)
        pos = pos-center
    else:
        empty_arg=False
        Vx = size[1]*du
        Vy = size[0]*dv
        phs = phsTools.reMoComp(phs, platform, center)
        pos = pos-center
    
    phsDS       = phs
    platformDS  = dict(platform)
    img_planeDS = dict(img_plane)
    
    #calculate decimation factor along range
    deltaF = abs(np.mean(np.diff(freq)))
    deltaFspot = c/(2*derate*np.linalg.norm([Vx, Vy]))
    N = int(np.floor(deltaFspot/deltaF))
    
    #force the decimation factor if specified by the user
    if N > factor_max:
        N = factor_max
    if N < factor_min:
        N = factor_min

    #decimate frequencies and phase history
    if N > 1:
        freq = sig.decimate(freq, N, n = n, beta = beta, cutoff = cutoff)
        phsDS = sig.decimate(phsDS, N, n = n, beta = beta, cutoff = cutoff)
    
    #update platform
    platformDS['nsamples'] = freq.size
    platformDS['freq']     = freq
    deltaF = freq[freq.size//2]-freq[freq.size//2-1] #Assume sample spacing can be determined by difference between last two values (first two are distorted by decimation filter)
    freq   = freq[freq.size//2]+np.arange(-freq.size//2,freq.size//2)*deltaF
    platformDS['k_r'] = 4*np.pi*freq/c

    #interpolate phs and pos using uniform azimuth spacing
    sph = sig.cart2sph(pos)
    sph[:,0] = np.unwrap(sph[:,0])
    RPP = sph[1:,0]-sph[:-1,0]
    abs_RPP = abs(RPP)
    I = np.argsort(abs_RPP); sort_RPP = abs_RPP[I]
    im=I[4]; RPPdata = sort_RPP[4]#len(I)/2
    
    az_i = np.arange(sph[0,0], sph[-1,0], RPP[im])
    sph_i = np.zeros([az_i.size, 3])
    sph_i[:,0] = az_i
    sph_i[:,1:3] = interp1d(sph[:,0], sph[:,1:3], axis = 0)(sph_i[:,0])
    phsDS = interp1d(sph[:,0], phsDS, axis = 0)(sph_i[:,0])
    
    sph = sph_i
    pos = sig.sph2cart(sph)

    #decimate slowtime positions and phase history
    fmax = freq[-1]
    PPRspot = derate*2*np.linalg.norm([Vx, Vy])*fmax*np.cos(sph[:,1].min())/c
    PPRdata = 1.0/RPPdata
    M = int(np.floor(PPRdata/PPRspot))
    
    #force the decimation factor if specified by the user
    if M > factor_max:
        M = factor_max
    if M < factor_min:
        M = factor_min
    
    if M > 1:        
        FilterScale = np.array([sig.decimate(np.ones(sph.shape[0]), M, n = n, beta = beta, cutoff = cutoff)]).T
        phsDS = sig.decimate(phsDS, M, axis=0, n = n, beta = beta, cutoff = cutoff)/FilterScale
        sph = sig.decimate(sph, M, axis=0, n = n, beta = beta, cutoff = cutoff)/FilterScale
    
    platformDS['npulses'] = int(phsDS.shape[0])
    platformDS['pos']     = sig.sph2cart(sph)
    
    #Update platform
    if empty_arg:
        img_planeDS['pixel_locs']     = p-np.array([center]).T
    else:
        #Find cordinates of center pixel
        p = img_plane['pixel_locs'].T
        center_index = np.argsort(np.linalg.norm(p-center, axis = -1))[0]
        center_index = np.array(np.unravel_index(center_index, [v.size, u.size]))
		
        #Update u and v
        img_planeDS['u']    = np.arange(-size[1]/2,size[1]/2)*du
        img_planeDS['v']    = np.arange(-size[0]/2,size[0]/2)*dv
		
        #get pixel locs for sub_image
        u_index = np.arange(center_index[1]-size[1]//2,center_index[1]+size[1]//2)
        v_index = np.arange(center_index[0]-size[0]//2,center_index[0]+size[0]//2)
        uu,vv = np.meshgrid(u_index,v_index)
        locs_index = np.ravel_multi_index((vv.flatten(),uu.flatten()),(v.size,u.size))
        img_planeDS['pixel_locs']     = img_plane['pixel_locs'][:,locs_index]-np.array([center]).T
    
    #Backproject using spotlighted data
    img = backprojection(phsDS, platformDS, img_planeDS, taylor = taylor, prnt = False)
    
    return(img)
    
def DS(phs, platform, img_plane, center=None, size=None, derate = 1.05, taylor = 20, n = 32, beta = 4, cutoff = 'nyq', factor_max = 6, factor_min = 0):
##############################################################################
#                                                                            #
#  This is the Digital Spotlight algorithm based on K. Dungan et. al.'s      #
#  2013 SPIE paper.  This is essentially the same as the DSBP algorithm,     #
#  only it returns the digitally spotlighted phase history, platform object, #
#  and img_plane object instead of the backprojected image.                  #
#                                                                            #
##############################################################################

    #Retrieve relevent parameters
    c           =   299792458.0
    pos         =   platform['pos']
    freq        =   platform['freq']
    u           =   img_plane['u']
    v           =   img_plane['v']
    du          =   img_plane['du']
    dv          =   img_plane['dv']  
    p           =   img_plane['pixel_locs']	
    
    #Derive parameters
    if center == None:
        empty_arg = True
        size = [0,0];
        size[1] = len(u)
        size[0] = len(v)
        Vx = u.max()-u.min()
        Vy = v.max()-v.min()
        center = np.mean(p, axis=-1)
        phs = phsTools.reMoComp(phs, platform, center)
        pos = pos-center
    else:
        empty_arg=False
        Vx = size[1]*du
        Vy = size[0]*dv
        phs = phsTools.reMoComp(phs, platform, center)
        pos = pos-center
    
    phsDS       = phs
    platformDS  = dict(platform)
    img_planeDS = dict(img_plane)
    
    #calculate decimation factor along range
    deltaF = abs(np.mean(np.diff(freq)))
    deltaFspot = c/(2*derate*np.linalg.norm([Vx, Vy]))
    N = int(np.floor(deltaFspot/deltaF))
    
    #force the decimation factor if specified by the user
    if N > factor_max:
        N = factor_max
    if N < factor_min:
        N = factor_min

    #decimate frequencies and phase history
    if N > 1:
        freq = sig.decimate(freq, N, n = n, beta = beta, cutoff = cutoff)
        phsDS = sig.decimate(phsDS, N, n = n, beta = beta, cutoff = cutoff)
    
    #update platform
    platformDS['nsamples'] = freq.size
    platformDS['freq']     = freq
    deltaF = freq[freq.size//2]-freq[freq.size//2-1] #Assume sample spacing can be determined by difference between last two values (first two are distorted by decimation filter)
    freq   = freq[freq.size//2]+np.arange(-freq.size//2,freq.size//2)*deltaF
    platformDS['k_r'] = 4*np.pi*freq/c

    #interpolate phs and pos using uniform azimuth spacing
    sph = sig.cart2sph(pos)
    sph[:,0] = np.unwrap(sph[:,0])
    RPP = sph[1:,0]-sph[:-1,0]
    abs_RPP = abs(RPP)
    I = np.argsort(abs_RPP); sort_RPP = abs_RPP[I]
    im=I[4]; RPPdata = sort_RPP[4]#len(I)/2
    
    az_i = np.arange(sph[0,0], sph[-1,0], RPP[im])
    sph_i = np.zeros([az_i.size, 3])
    sph_i[:,0] = az_i
    sph_i[:,1:3] = interp1d(sph[:,0], sph[:,1:3], axis = 0)(sph_i[:,0])
    phsDS = interp1d(sph[:,0], phsDS, axis = 0)(sph_i[:,0])
    
    sph = sph_i
    pos = sig.sph2cart(sph)

    #decimate slowtime positions and phase history
    fmax = freq[-1]
    PPRspot = derate*2*np.linalg.norm([Vx, Vy])*fmax*np.cos(sph[:,1].min())/c
    PPRdata = 1.0/RPPdata
    M = int(np.floor(PPRdata/PPRspot))
    
    #force the decimation factor if specified by the user
    if M > factor_max:
        M = factor_max
    if M < factor_min:
        M = factor_min
    
    if M > 1:        
        FilterScale = np.array([sig.decimate(np.ones(sph.shape[0]), M, n = n, beta = beta, cutoff = cutoff)]).T
        phsDS = sig.decimate(phsDS, M, axis=0, n = n, beta = beta, cutoff = cutoff)/FilterScale
        sph = sig.decimate(sph, M, axis=0, n = n, beta = beta, cutoff = cutoff)/FilterScale
    
    platformDS['npulses'] = int(phsDS.shape[0])
    platformDS['pos']     = sig.sph2cart(sph)
    
    #Update platform
    if empty_arg:
        img_planeDS['pixel_locs']     = p-np.array([center]).T
    else:
        #Find cordinates of center pixel
        p = img_plane['pixel_locs'].T
        center_index = np.argsort(np.linalg.norm(p-center, axis = -1))[0]
        center_index = np.array(np.unravel_index(center_index, [v.size, u.size]))
		
        #Update u and v
        img_planeDS['u']    = np.arange(-size[1]/2,size[1]/2)*du
        img_planeDS['v']    = np.arange(-size[0]/2,size[0]/2)*dv
		
        #get pixel locs for sub_image
        u_index = np.arange(center_index[1]-size[1]/2,center_index[1]+size[1]/2)
        v_index = np.arange(center_index[0]-size[0]/2,center_index[0]+size[0]/2)
        uu,vv = np.meshgrid(u_index,v_index)
        locs_index = np.ravel_multi_index((vv.flatten(),uu.flatten()),(v.size,u.size))
        img_planeDS['pixel_locs']     = img_plane['pixel_locs'][:,locs_index]-np.array([center]).T

    return(phsDS, platformDS, img_planeDS)
    
def FFBP(phs, platform, img_plane, N=3, derate = 1.05, taylor = 20, n = 32, beta = 4, cutoff = 'nyq', factor_max = 2, factor_min = 0, prnt = True):
##############################################################################
#                                                                            #
#  This is the Fast Factorized Backprojection Algorithm.  Factorization at   #
#  each level of recursion is handled by the Digital Spotlight algorithm     #
#  based on K. Dungan et. al.'s 2013 SPIE paper.  This algorithm is intended #
#  to work with image sizes that are a factor of 2.  The size of the phase   #
#  can be arbitrary - the AFRL digital spotlight agorithm will automatically #
#  determine the optimal factorization factor.  Alternatively, the user can  #
#  specify the maximum and minimum default factorization factor at each      #
#  stage of recursion with the parameters factor_max and factor_min,         #
#  respectively.  Performance can be increased by decreasing the order of    #
#  the interpolation window (n) or increasing factor_min.  Image fidelity    #
#  can be increased by increasing the order of the interpolation window or   #
#  reducing factor_max.                                                      #
#                                                                            #
##############################################################################

    #add sub_image_index key to img_plane
    img_plane['index'] = np.array([0,0])
        
    #create parent containers
    phsDS_list          = tuple([phs])
    platformDS_list     = tuple([platform])
    img_planeDS_list    = tuple([img_plane])
    
    #Begin factorization
    for i in range(N):
        if prnt:
            print('processing recursion level %i of %i'%((i+1),N))
        
        #create temporary child containers
        phsDS_list_tmp      = []
        platformDS_list_tmp = []
        img_planeDS_list_tmp= [];
        
        #Determine number of image patches    
        n_img = 4**i
        
        image_number = 1
        for j in range(n_img):
            phsDS           =   phsDS_list[j]
            platformDS      =   platformDS_list[j]
            img_planeDS     =   img_planeDS_list[j]
            
            #Retrieve relevent parameters
            u        = img_planeDS['u']
            v        = img_planeDS['v']
            pos      = img_planeDS['pixel_locs']
            index    = img_planeDS['index']*2

        
            #Derive parameters
            img_plane_sub = dict(img_planeDS)
            full_size = np.array([v.size, u.size], dtype = np.int)
            sub_size = np.array(full_size/2, dtype = int)
            img_FFBP = np.zeros(full_size)
        
            #For each pixel, assign a position index
            pos_array = np.arange(len(pos[0,]))
            pos_array = np.reshape(pos_array, full_size)[::-1]
        
            #Break image into sub images (get sub_image indices)
            img_FFBP = np.zeros(full_size)+0j
            for k in range(2):
                for l in range(2):
                    if prnt:
                        print('digitally spotlighting sub-image %i of %i'
                            %(image_number, n_img*4))
                    
                    #update img_plane['u','v']
                    r = np.arange(sub_size[0]*k, sub_size[0]*(k+1))[::-1]
                    c = np.arange(sub_size[1]*l, sub_size[1]*(l+1))
                    img_plane_sub['u'] = u[c]
                    img_plane_sub['v'] = v[r]
                    cc,rr = np.meshgrid(c,r)
                
                    #Get pixel_locs for each sub image
                    pos_index = pos_array[rr,cc].flatten()
                    img_plane_sub['pixel_locs']=\
                        pos[:,pos_index]
                    
                    #digitally spotlight data data
                    tmp = DS(phsDS, platformDS, img_plane_sub,
                        derate = derate, taylor=17, n = n, beta = beta, cutoff = cutoff, factor_max = factor_max, factor_min = factor_min)
                        
                    #update sub_image index
                    tmp[2]['index'] = index + [k,l]
                    
                    phsDS_list_tmp.append(tmp[0])
                    platformDS_list_tmp.append(tmp[1])
                    img_planeDS_list_tmp.append(tmp[2])
                    
                    image_number+=1
        
        #replace parent containers with child containers
        phsDS_list          = tuple(phsDS_list_tmp)
        platformDS_list     = tuple(platformDS_list_tmp)
        img_planeDS_list    = tuple(img_planeDS_list_tmp)
        
    #backproject factorized data
    #####################################################
    
    #Retrieve relevent parameters
    u        = img_plane['u']
    v        = img_plane['v']
    pos      = img_plane['pixel_locs']
    
    #Determine number of image patches
    n_img = image_number-1
    
    #Derive parameters
    full_size = np.array([v.size, u.size], dtype = np.int)
    sub_size = np.array(full_size/(2**N), dtype = int)
    img_FFBP = np.zeros(full_size)
    
    #Break image into sub images (get sub_image indices)
    img_FFBP = np.zeros([v.size, u.size])+0j
    for image_number in range(n_img):
        print('creating sub-image %i of %i'%((image_number+1), n_img))
        i,j = img_planeDS_list[image_number]['index']
    
        #update img_plane['u','v']
        r = np.arange(sub_size[0]*i, sub_size[0]*(i+1))
        c = np.arange(sub_size[1]*j, sub_size[1]*(j+1))
        cc,rr = np.meshgrid(c,r)
        
        #Backproject using spotlighted data
        img_FFBP[rr, cc] = backprojection(
            phsDS_list[image_number], platformDS_list[image_number], img_planeDS_list[image_number], prnt=False)
                
    return(img_FFBP)


def FFBPmp(phs, platform, img_plane, N=3, derate = 1.05, taylor = 20, n = 32, beta = 4, cutoff = 'nyq', factor_max = 2, factor_min = 0):
##############################################################################
#                                                                            #
#  This is the Fast Factorized Backprojection Algorithm with                 #
#  multi-processing.  Processing begins with breaking the image up into 4    #
#  sub images using the digital spotlight algorithm and forcing the          #
#  factorization factor to 1.  The multiprocessing library is then used to   #
#  create 4 processes.  For each process, the sub images are generated using #
#  the factorized backprojection algorithm.  It is extremely import that     #
#  this function is only run from a .py script and the first line of that    #
#  script contains the statement "if __name__ == "__main__": immediately     #
#  after all import statements.  Reference the FFBPmp demo for more details. #   
#                                                                            #
##############################################################################
    
    #add sub_image_index key to img_plane
    img_plane['index'] = np.array([0,0])
        
    #create parent containers
    phsDS_list          = tuple([phs])
    platformDS_list     = tuple([platform])
    img_planeDS_list    = tuple([img_plane])
    
    #Begin factorization
    #create temporary child containers
    phsDS_list_tmp      = []
    platformDS_list_tmp = []
    img_planeDS_list_tmp= [];
        
    #Retrieve relevent parameters
    u        = img_plane['u']
    v        = img_plane['v']
    pos      = img_plane['pixel_locs']
    index    = img_plane['index']*2
    
    #Derive parameters
    img_plane_sub = dict(img_plane)
    full_size = np.array([v.size, u.size], dtype = np.int)
    sub_size = np.array(full_size/2, dtype = int)
    img_FFBP = np.zeros(full_size)
    
    #For each pixel, assign a position index
    pos_array = np.arange(len(pos[0,]))
    pos_array = np.reshape(pos_array, full_size)[::-1]
    
    print('creating 4 sub images and assigning them to 4 processes')
    #Break image into sub images (get sub_image indices)
    img_FFBP = np.zeros(full_size)+0j
    for k in range(2):
        for l in range(2):        
            #update img_plane['u','v']
            r = np.arange(sub_size[0]*k, sub_size[0]*(k+1))[::-1]
            c = np.arange(sub_size[1]*l, sub_size[1]*(l+1))
            img_plane_sub['u'] = u[c]
            img_plane_sub['v'] = v[r]
            cc,rr = np.meshgrid(c,r)
        
            #Get pixel_locs for each sub image
            pos_index = pos_array[rr,cc].flatten()
            img_plane_sub['pixel_locs']=\
                pos[:,pos_index]
            
            #digitally spotlight data data
            tmp = DS(phs, platform, img_plane_sub,
                derate = derate, taylor=17, n = n, beta = beta, cutoff = cutoff, factor_max = 1, factor_min = 1)
                
            #update sub_image index
            tmp[2]['index'] = index + [k,l]
            
            phsDS_list_tmp.append(tmp[0])
            platformDS_list_tmp.append(tmp[1])
            img_planeDS_list_tmp.append(tmp[2])
    
    #replace parent containers with child containers
    phsDS_list          = tuple(phsDS_list_tmp)
    platformDS_list     = tuple(platformDS_list_tmp)
    img_planeDS_list    = tuple(img_planeDS_list_tmp)
        
    #backproject factorized data
    #####################################################
    
    #Retrieve relevent parameters
    u        = img_plane['u']
    v        = img_plane['v']
    pos      = img_plane['pixel_locs']
    
    #Determine number of image patches
    n_img = 4
    
    #Create 4 processes and assign a sub image to each process    
    pool = mp.Pool(processes=4)
    results = [pool.apply_async(FFBP,
    args=(phsDS_list[i], platformDS_list[i], img_planeDS_list[i],
        N-1, derate, taylor, n, beta, cutoff, factor_max, factor_min, False))
    for i in range(4)]
    output = [p.get() for p in results]
    
    #Derive parameters
    full_size = np.array([v.size, u.size], dtype = np.int)
    sub_size = np.array(full_size/2, dtype = int)
    img_FFBP = np.zeros(full_size)
    
    print('proessing 4 sub images, please wait...')
    #Break image into sub images (get sub_image indices)
    img_FFBP = np.zeros([v.size, u.size])+0j
    for image_number in range(n_img):
        print('creating sub-image %i of %i'%((image_number+1), n_img))
        i,j = img_planeDS_list[image_number]['index']
    
        #update img_plane['u','v']
        r = np.arange(sub_size[0]*i, sub_size[0]*(i+1))
        c = np.arange(sub_size[1]*j, sub_size[1]*(j+1))
        cc,rr = np.meshgrid(c,r)
        
        #Insert insert sub images into full image
        img_FFBP[rr, cc] = output[image_number]
                
    return(img_FFBP)


    

def autoFocus(img, win = 'auto', win_params = [100,0.5]):
##############################################################################
#                                                                            #
#  This program autofocuses an image using the Phase Gradient Algorithm.     #
#  If the parameter win is set to auto, an adaptive window is used.          #
#  Otherwise, the user sets win to 0 and defines win_params.  The first      #
#  element of win_params is the starting windows size.  The second element   #
#  is the factor by which to reduce it by for each iteration.  This version  #
#  is more suited for an image that is mostly focused.  Below is the paper   #
#  this algorithm is based off of.                                           #
#                                                                            #
#  D. Wahl, P. Eichel, D. Ghiglia, and J. Jakowatz, C.V., \Phase gradient    #
#  autofocus-a robust tool for high resolution sar phase correction,"        #
#  Aerospace and Electronic Systems, IEEE Transactions on, vol. 30,          #
#  pp. 827{835, Jul 1994.                                                    #
#                                                                            #
##############################################################################
    
    #Derive parameters
    npulses = int(img.shape[0])
    nsamples = int(img.shape[1])
    
    #Initialize loop variables
    img_af = 1.0*img
    max_iter = 30
    af_ph = 0
    rms = []
    
    #Compute phase error and apply correction
    for iii in range(max_iter):
        
        #Find brightest azimuth sample in each range bin
        index = np.argsort(np.abs(img_af), axis=0)[-1]
        
        #Circularly shift image so max values line up   
        f = np.zeros(img.shape)+0j
        for i in range(nsamples):
            f[:,i] = np.roll(img_af[:,i], npulses//2-index[i])
        
        if win == 'auto':
            #Compute window width    
            s = np.sum(f*np.conj(f), axis = -1)
            s = 10*np.log10(s/s.max())
            width = np.sum(s>-30)
            window = np.arange(npulses//2-width//2,npulses//2+width//2)
        else:
            #Compute window width using win_params if win not set to 'auto'    
            width = int(win_params[0]*win_params[1]**iii)
            window = np.arange(npulses//2-width//2,npulses//2+width//2)
            if width<5:
                break
        
        #Window image
        g = np.zeros(img.shape)+0j
        g[window] = f[window]
        
        #Fourier Transform
        G = sig.ift(g, ax=0)
        
        #take derivative
        G_dot = np.diff(G, axis=0)
        a = np.array([G_dot[-1,:]])
        G_dot = np.append(G_dot,a,axis = 0)
        
        #Estimate Spectrum for the derivative of the phase error
        phi_dot = np.sum((np.conj(G)*G_dot).imag, axis = -1)/\
                  np.sum(np.abs(G)**2, axis = -1)
                
        #Integrate to obtain estimate of phase error(Jak)
        phi = np.cumsum(phi_dot)
        
        #Remove linear trend
        t = np.arange(0,npulses) #originally np.arange(0,nsamples)
        slope, intercept, r_value, p_value, std_err = linregress(t,phi)
        line = slope*t+intercept
        phi = phi-line
        rms.append(np.sqrt(np.mean(phi**2)))
        
        if win == 'auto':
            if rms[iii]<0.1:
                break
        
        #Apply correction
        phi2 = np.tile(np.array([phi]).T,(1,nsamples))
        IMG_af = sig.ift(img_af, ax=0)
        IMG_af = IMG_af*np.exp(-1j*phi2)
        img_af = sig.ft(IMG_af, ax=0)
        
        #Store phase
        af_ph += phi    
       
    fig = plt.figure(figsize = (12,10))
    ax1 = fig.add_subplot(2,2,1)
    ax1.set_title('original')
    ax1.imshow(10*np.log10(np.abs(img)/np.abs(img).max()), cmap = cm.Greys_r)
    ax2 = fig.add_subplot(2,2,2)
    ax2.set_title('autofocused')
    ax2.imshow(10*np.log10(np.abs(img_af)/np.abs(img_af).max()), cmap = cm.Greys_r)
    ax3 = fig.add_subplot(2,2,3)
    ax3.set_title('rms phase error vs. iteration')
    plt.ylabel('Phase (radians)')
    ax3.plot(rms)
    ax4 = fig.add_subplot(2,2,4)
    ax4.set_title('phase error')
    plt.ylabel('Phase (radians)')
    ax4.plot(af_ph)
    plt.tight_layout()
    

    print('number of iterations: %i'%(iii+1))
                     
    return(img_af, af_ph)
    

def autoFocus2(img, win = 'auto', win_params = [100,0.5]):
##############################################################################
#                                                                            #
#  This program autofocuses an image using the Phase Gradient Algorithm.     #
#  If the parameter win is set to auto, an adaptive window is used.          #
#  Otherwise, the user sets win to 0 and defines win_params.  The first      #
#  element of win_params is the starting windows size.  The second element   #
#  is the factor by which to reduce it by for each iteration.  This version  #
#  is more suited for an image that is severely degraded (such as for the    #
#  auto_focusing demo)                                                       #
#  since, for the adaptive window, it uses most of the data for the first    #
#  few iterations.  Below is the paper this algorithm is based off of.       #
#                                                                            #
#  D. Wahl, P. Eichel, D. Ghiglia, and J. Jakowatz, C.V., \Phase gradient    #
#  autofocus-a robust tool for high resolution sar phase correction,"        #
#  Aerospace and Electronic Systems, IEEE Transactions on, vol. 30,          #
#  pp. 827{835, Jul 1994.                                                    #
#                                                                            #
##############################################################################
    
    #Derive parameters
    npulses = int(img.shape[0])
    nsamples = int(img.shape[1])
    
    #Initialize loop variables
    img_af = 1.0*img
    max_iter = 10
    af_ph = 0
    rms = []
    
    #Compute phase error and apply correction
    for iii in range(max_iter):
        
        #Find brightest azimuth sample in each range bin
        index = np.argsort(np.abs(img_af), axis=0)[-1]
        
        #Circularly shift image so max values line up   
        f = np.zeros(img.shape)+0j
        for i in range(nsamples):
            f[:,i] = np.roll(img_af[:,i], npulses//2-index[i])
        
        if win == 'auto':
            #Compute window width    
            s = np.sum(f*np.conj(f), axis = -1)
            s = 10*np.log10(s/s.max())
            #For first two iterations use all azimuth data 
            #and half of azimuth data, respectively
            if iii == 0:
                width = npulses
            elif iii == 1:
                width = npulses//2
            #For all other iterations, use twice the 30 dB threshold
            else:
                width = np.sum(s>-30)
            window = np.arange(npulses//2-width//2,npulses//2+width//2)
        else:
            #Compute window width using win_params if win not set to 'auto'    
            width = int(win_params[0]*win_params[1]**iii)
            window = np.arange(npulses//2-width//2,npulses//2+width//2)
            if width<5:
                break
        
        #Window image
        g = np.zeros(img.shape)+0j
        g[window] = f[window]
        
        #Fourier Transform
        G = sig.ift(g, ax=0)
        
        #take derivative
        G_dot = np.diff(G, axis=0)
        a = np.array([G_dot[-1,:]])
        G_dot = np.append(G_dot,a,axis = 0)
        
        #Estimate Spectrum for the derivative of the phase error
        phi_dot = np.sum((np.conj(G)*G_dot).imag, axis = -1)/\
                  np.sum(np.abs(G)**2, axis = -1)
                
        #Integrate to obtain estimate of phase error(Jak)
        phi = np.cumsum(phi_dot)
        #Remove linear trend
        t = np.arange(0,npulses) #originally np.arange(0,nsamples)
        slope, intercept, r_value, p_value, std_err = linregress(t,phi)
        line = slope*t+intercept
        phi = phi-line
        rms.append(np.sqrt(np.mean(phi**2)))
        
        if win == 'auto':
            if rms[iii]<0.1:
                break
        
        #Apply correction
        phi2 = np.tile(np.array([phi]).T,(1,nsamples))
        IMG_af = sig.ift(img_af, ax=0)
        IMG_af = IMG_af*np.exp(-1j*phi2)
        img_af = sig.ft(IMG_af, ax=0)
        
        #Store phase
        af_ph += phi    
       
    fig = plt.figure(figsize = (12,10))
    ax1 = fig.add_subplot(2,2,1)
    ax1.set_title('original')
    ax1.imshow(10*np.log10(np.abs(img)/np.abs(img).max()), cmap = cm.Greys_r)
    ax2 = fig.add_subplot(2,2,2)
    ax2.set_title('autofocused')
    ax2.imshow(10*np.log10(np.abs(img_af)/np.abs(img_af).max()), cmap = cm.Greys_r)
    ax3 = fig.add_subplot(2,2,3)
    ax3.set_title('rms phase error vs. iteration')
    plt.ylabel('Phase (radians)')
    ax3.plot(rms)
    ax4 = fig.add_subplot(2,2,4)
    ax4.set_title('phase error')
    plt.ylabel('Phase (radians)')
    ax4.plot(af_ph)
    plt.tight_layout()
    

    print('number of iterations: %i'%(iii+1))
                     
    return(img_af, af_ph)
    
def imshow(img, dB_scale = [0,0], extent = None):
###############################################################################
#                                                                             #
#  This program displays the processed data in dB.  The brightest point in    #
#  the image is used as the reference and the user can define the scale for   #
#  the intensity range.                                                       #
#                                                                             #
###############################################################################

    #Convert to dB
    img = 10*np.log10(np.abs(img)/np.abs(img).max())
    img[img == -np.inf] = dB_scale[0]

    #Determine if the image is RGB
    if len(img.shape) != 3:
    
        #Display the image
        if dB_scale == [0,0]:
            plt.imshow(img, cmap=cm.Greys_r, extent = extent)
        else:
            plt.imshow(img, cmap=cm.Greys_r,
                       vmin = dB_scale[0], vmax = dB_scale[-1], extent = extent)
    
    #If the image is RGB                 
    else:
        #Display the image
        if dB_scale == [0,0]:
            img_RGB = (img-img.min())/(img.max()-img.min())
            plt.imshow(img_RGB, extent = extent)
        else:
            img[img<=dB_scale[0]] = dB_scale[0]
            img[img>=dB_scale[-1]] = dB_scale[-1]
            img_RGB = (img-img.min())/(img.max()-img.min())
            plt.imshow(img_RGB, extent = extent)
