#Include dependencies
import numpy as np
from scipy.stats import linregress
from fnmatch import fnmatch
import pathlib
import os
import scipy.io as sio 
import xml.etree.ElementTree as ET


#%%

def AFRL(directory, start_az, pol=False, n_az=3):
    '''
    This function reads in the AFRL *.mat files from the user supplied 
    directory and exports both the phs and a Python dictionary compatible
    with ritsar.

    Parameters
    ----------
    directory : str
        Path to the directory in which your phase history files are housed. If using 'pol'
        this instead houses the polarization files such as 'HH' and 'VV'.
    start_az : int
        AFRL ends its filenames in a number. I haven't quite figured out what the number 
        means yet. But for the user it means which file in the folder you want to start with.
    pol : str, optional
        Sub folder which houses the actual data in polarized sets, does not need to be used 
        if you input the direct path in `directory`. 
    n_az : int, optional
        The number of files to join together. I noticed this means something in older data
        but I'm not sure if it has any meaning in newer sets. Default is 1.
    '''
    
    #Get filenames
    if pol:
        root=pathlib.Path(directory+'/'+pol)
    else:
        root=pathlib.Path(directory)
    items=list(root.glob('*.mat'))
    az=np.arange(start_az, start_az+n_az)
    if pol:
        suffix=[str('%03d_%s.mat'%(a,pol)) for a in az]
    else:
        suffix=[str('%04d.mat'%a) for a in az]
    fnames=[item for item in items if any(s in item.name for s in suffix)]

    #Grab n_az phase histories
    phs = []; platform = []
    for fname in sorted(fnames):
        #Convert MATLAB structure to Python dictionary
        data=sio.loadmat(fname, simplify_cells=True)['data']

        #Define phase history
        keys=data.keys()
        key='fp' if 'fp' in keys else 'fq' #Used for compatibility
        phs_tmp     = data[key].T
        phs.append(phs_tmp)

        #Transform data to be compatible with ritsar
        c           = np.float64(299792458.0)
        nsamples    = int(phs_tmp.shape[1])
        npulses     = int(phs_tmp.shape[0])
        freq        = np.float64(data['freq'])
        pos         = np.vstack((data['x'], data['y'], data['z'])).T
        k_r         = 4*np.pi*freq/c
        B_IF        = freq.max()-freq.min()
        delta_r     = c/(2*B_IF)
        delta_t     = 1.0/B_IF
        t           = np.linspace(-nsamples/2, nsamples/2, nsamples)*delta_t

        chirprate, f_0, r, p, s\
                    = linregress(t, freq)

        #Vector to scene center at synthetic aperture center
        if np.mod(npulses,2)>0:
            R_c = pos[npulses//2]
        else:
            R_c = np.mean(
                    pos[npulses//2-1:npulses//2+1],
                    axis = 0)

        #Save values to dictionary for export
        platform_tmp = \
        {
            'f_0'       :   f_0,
            'freq'      :   freq,
            'chirprate' :   chirprate,
            'B_IF'      :   B_IF,
            'nsamples'  :   nsamples,
            'npulses'   :   npulses,
            'pos'       :   pos,
            'delta_r'   :   delta_r,
            'R_c'       :   R_c,
            't'         :   t,
            'k_r'       :   k_r,
            'r0'        :   data['r0'],
            'th'        :   data['th'],
            'phi'       :   data['phi']
        }
        platform.append(platform_tmp)

    #Stack data from different azimuth files
    phs = np.vstack(phs)
    npulses = int(phs.shape[0])
    
    
    #Stack position, per pulse reference range, theta(degrees), and depression angle
    pos = platform[0]['pos']
    r0  = platform[0]['r0']
    th  = platform[0]['th']
    phi = platform[0]['phi']
    for i in range(1, n_az):
        pos = np.vstack((pos, platform[i]['pos']))
        r0  = np.append(r0,platform[i]['r0'])
        th  = np.append(th,platform[i]['th'])
        phi = np.append(phi,platform[i]['phi'])

    if np.mod(npulses,2)>0:
        R_c = pos[npulses//2]
    else:
        R_c = np.mean(
                pos[npulses//2-1:npulses//2+1],
                axis = 0)

    #Replace Dictionary values
    platform = platform_tmp
    platform_update=\
    {
            'npulses'    :   npulses,
            'pos'        :   pos,
            'R_c'        :   R_c,
            'r0'         :   r0,
            'th'         :   th,
            'phi'        :   phi,
    }
    platform.update(platform_update)
    
    #Synthetic aperture length
    L = np.linalg.norm(pos[-1]-pos[0])

    #Add k_y
    platform['k_y'] = np.linspace(-npulses/2,npulses/2,npulses)*2*np.pi/L

    return(phs, platform)
    
def Sandia(directory):
    '''
    ##############################################################################
    #                                                                            #
    #  This function reads in the Sandia *.phs and *.au2 files from the user     #
    #  supplied directoryand exports both the phs and a Python dictionary        #
    #  compatible with ritsar.                                                   #
    #                                                                            #
    ##############################################################################
    '''
    #get filename containing auxilliary data
    for file in os.listdir(directory):
            if fnmatch(file, '*.au2'):
                aux_fname = directory+file    
    
    #import auxillary data
    f=open(aux_fname,'rb')
    
    #initialize tuple
    record=['blank'] #first record blank to ensure
                     #indices match record numbers
    
    #record 1
    data = np.fromfile(f, dtype = np.dtype([
        ('version','S6'),
        ('phtype','S6'),
        ('phmode','S6'),
        ('phgrid','S6'),
        ('phscal','S6'),
        ('cbps','S6')
        ]),count=1)
    record.append(data[0])
    
    #record 2
    f.seek(44)
    data = np.fromfile(f, dtype = np.dtype([
        ('npulses','i4'),
        ('nsamples','i4'),
        ('ipp_start','i4'),
        ('ddas','f4',(5,)),
        ('kamb','i4')
        ]),count=1)
    record.append(data[0])
    
    #record 3    
    f.seek(44*2)
    data = np.fromfile(f, dtype = np.dtype([
        ('fpn','f4',(3,)),
        ('grp','f4',(3,)),
        ('cdpstr','f4'),
        ('cdpstp','f4')
        ]),count=1)
    record.append(data[0])
    
    #record 4
    f.seek(44*3)
    data = np.fromfile(f, dtype = np.dtype([
        ('f0','f4'),
        ('fs','f4'),
        ('fdot','f4'),
        ('r0','f4')
        ]),count=1)
    record.append(data[0])
    
    #record 5 (blank)rvr_au_read.py
    f.seek(44*4)
    data = []
    record.append(data)
    
    #record 6
    npulses = record[2]['npulses']
    rpoint = np.zeros([npulses,3])
    deltar = np.zeros([npulses,])
    fscale = np.zeros([npulses,])
    c_stab = np.zeros([npulses,3])
    #build up arrays for record(npulses+6)
    for n in range(npulses):
        f.seek((n+5)*44)
        data = np.fromfile(f, dtype = np.dtype([
            ('rpoint','f4',(3,)),
            ('deltar','f4'),
            ('fscale','f4'),
            ('c_stab','f8',(3,))
            ]),count=1)
        rpoint[n,:] = data[0]['rpoint']
        deltar[n] = data[0]['deltar']
        fscale[n] = data[0]['fscale']
        c_stab[n,:] = data[0]['c_stab']
    #consolidate arrays into a 'data' dataype
    dt = np.dtype([
            ('rpoint','f4',(npulses,3)),
            ('deltar','f4',(npulses,)),
            ('fscale','f4',(npulses,)),
            ('c_stab','f8',(npulses,3))
            ])        
    data = np.array((rpoint,deltar,fscale,c_stab)
            ,dtype=dt)
    #write to record file
    record.append(data)
    
    #import phase history
    for file in os.listdir(directory):
        if fnmatch(file, '*.phs'):
            phs_fname = directory+file
            
    nsamples = record[2][1]
    npulses = record[2][0]
    
    f=open(phs_fname,'rb')    
    dt = np.dtype('i2')
        
    phs = np.fromfile(f, dtype=dt, count=-1)
    real = phs[0::2].reshape([npulses,nsamples])  
    imag = phs[1::2].reshape([npulses,nsamples])
    phs = real+1j*imag
    
    #Create platform dictionary
    c       = 299792458.0
    pos     = record[6]['rpoint']
    n_hat   = record[3]['fpn']
    delta_t = record[4]['fs']
    t       = np.linspace(-nsamples/2, nsamples/2, nsamples)*1.0/delta_t
    chirprate = record[4]['fdot']*1.0/(2*np.pi)
    f_0     = record[4]['f0']*1.0/(2*np.pi) + chirprate*nsamples/(2*delta_t)
    B_IF    = (t.max()-t.min())*chirprate
    delta_r = c/(2*B_IF)
    freq = f_0+chirprate*t
    omega = 2*np.pi*freq
    k_r = 2*omega/c
    
    if np.mod(npulses,2)>0:
        R_c = pos[npulses//2]
    else:
        R_c = np.mean(
                pos[npulses//2-1:npulses//2+1],
                axis = 0)
    
    platform = \
    {
        'f_0'       :   f_0,
        'chirprate' :   chirprate,
        'B_IF'      :   B_IF,
        'nsamples'  :   nsamples,
        'npulses'   :   npulses,
        'delta_r'   :   delta_r,
        'pos'       :   pos,
        'R_c'       :   R_c,
        't'         :   t,
        'k_r'       :   k_r,
        'n_hat'     :   n_hat,
        'freq'      :   freq
    }
    
    return(phs, platform)



def get(root, entry):
    for entry in root.iter(entry):
        out = entry.text
        
    return(out)

def getWildcard(directory, char):
    for file in os.listdir(directory):
            if fnmatch(file, char):
                fname = directory+file
    
    return(fname)

def DIRSIG(directory):
    '''
    ##############################################################################
    #                                                                            #
    #  This function reads in the DIRSIG xml data as well as the envi header     #
    #  file from the user supplied directory. The phs and a Python dictionary    #
    #  compatible with ritsar are returned to the function caller.               #
    #                                                                            #
    ##############################################################################
    '''
    from spectral.io import envi
    
    #get phase history
    phs_fname = getWildcard(directory, '*.hdr')
    phs = envi.open(phs_fname).load(dtype = np.complex128)
    phs = np.squeeze(phs)
    
    #get platform geometry
    ppd_fname = getWildcard(directory, '*.ppd')
    tree = ET.parse(ppd_fname)
    root = tree.getroot()
    
    pos_dirs = []
    for children in root.iter('point'):
        pos_dirs.append(float(children[0].text))
        pos_dirs.append(float(children[1].text))
        pos_dirs.append(float(children[2].text))
    pos_dirs = np.asarray(pos_dirs).reshape([len(pos_dirs)//3,3])
    
    t_dirs=[]
    for children in root.iter('datetime'):
        t_dirs.append(float(children.text))
    t_dirs = np.asarray(t_dirs)
    
    #get platform system paramters
    platform_fname = getWildcard(directory, '*.platform')
    tree = ET.parse(platform_fname)
    root = tree.getroot()
    
    #put metadata into a dictionary
    metadata = root[0]
    keys = []; vals = []
    for children in metadata:
        keys.append(children[0].text)
        vals.append(children[1].text)
    metadata = dict(zip(keys,vals))
    
    #obtain key parameters
    c           = 299792458.0
    nsamples    = int(phs.shape[1])
    npulses     = int(phs.shape[0])
    vp          = float(get(root, 'speed'))
    delta_t     = float(get(root, 'delta'))
    t           = np.linspace(-nsamples/2, nsamples/2, nsamples)*delta_t
    prf         = float(get(root, 'clockrate'))
    chirprate   = float(get(root, 'chirprate'))/np.pi
    T_p         = float(get(root, 'pulseduration'))
    B           = T_p*chirprate
    B_IF        = (t.max() - t.min())*chirprate
    delta_r     = c/(2*B_IF)
    f_0         = float(get(root, 'center'))*1e9
    freq        = f_0+chirprate*t
    omega       = 2*np.pi*freq
    k_r         = 2*omega/c
    T0          = float(get(root, 'min'))
    T1          = float(get(root, 'max'))
    
    #compute slowtime position
    ti = np.linspace(0,1.0/prf*npulses, npulses)
    x = np.array([np.interp(ti, t_dirs, pos_dirs[:,0])]).T
    y = np.array([np.interp(ti, t_dirs, pos_dirs[:,1])]).T
    z = np.array([np.interp(ti, t_dirs, pos_dirs[:,2])]).T
    pos = np.hstack((x,y,z))
    L = np.linalg.norm(pos[-1]-pos[0])
    k_y = np.linspace(-npulses/2,npulses/2,npulses)*2*np.pi/L
    
    #Vector to scene center at synthetic aperture center
    if np.mod(npulses,2)>0:
        R_c = pos[npulses//2]
    else:
        R_c = np.mean(
                pos[npulses//2-1:npulses//2+1],
                axis = 0)
                
    #Derived Parameters
    if np.mod(nsamples,2)==0:
        T = np.arange(T0, T1+0*delta_t, delta_t)
    else:
        T = np.arange(T0, T1, delta_t)
    
    #Mix signal
    signal = np.zeros(phs.shape)+0j
    for i in range(0,npulses,1):
        r_0 = np.linalg.norm(pos[i])
        tau_c = 2*r_0/c
        ref = np.exp(-1j*(2*np.pi*f_0*(T-tau_c)+np.pi*chirprate*(T-tau_c)**2))
        signal[i,:] = ref*phs[i,:]
    
    platform = \
    {
        'f_0'       :   f_0,
        'freq'      :   freq,
        'chirprate' :   chirprate,
        'B'         :   B,
        'B_IF'      :   B_IF,
        'nsamples'  :   nsamples,
        'npulses'   :   npulses,
        'delta_r'   :   delta_r,
        'delta_t'   :   delta_t,
        'vp'        :   vp,
        'pos'       :   pos,
        'R_c'       :   R_c,
        't'         :   t,
        'k_r'       :   k_r,
        'k_y'       :   k_y,
        'metadata'  :   metadata
    }
    
    return(signal, platform)
