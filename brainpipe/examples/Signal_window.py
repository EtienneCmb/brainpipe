#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""this scripts illustrate how to take the mean of a signal
inside defined windows.
"""
import numpy as np
import matplotlib.pyplot as plt
from brainpipe.features import Window

if __name__ == "__main__":
    # -----------------------------------------
    #          SIGNAL CREATION
    # -----------------------------------------
    # First, create a noisy 10hz sinus:
    sf = 512.0 # Sampling rate
    f = 10.0    # Sinus frequency
    npts = 1000 # Number of points
    t = np.arange(npts, dtype=float) / sf
    sig = np.sin(2*np.pi*f*t) + np.random.rand(npts)

    # -----------------------------------------
    #          WINDOW
    # -----------------------------------------
    # Take the mean inside a predefined window (defined in secondes)
    pwin = Window(sf, window=[0.4, 0.45], unit='s')
    sig_pwin = pwin.apply(sig)
    t_pwin = pwin.apply(t)
    pwin_ms = Window(sf, window=[400, 450], unit='ms')
    # Alternatively, define window using milliseconds :
    print('Window (s): ', pwin.window, 'Window (ms)', pwin_ms.window)

    # Define an automatic sliding window :
    pwin_sl = Window(sf, auto=(100, 600, 10, 5), unit='ms')
    sig_sl = pwin_sl.apply(sig)
    t_sl = pwin_sl.apply(t)
    pwin_S = Window(sf, auto=(700, 990, 20, 10), unit='sample')
    sig_S = pwin_S.apply(sig)
    t_S = pwin_S.apply(t)
    pwin_P = Window(sf, window=[[595, 605], [1360, 1370]], unit='ms')
    sig_P = pwin_P.apply(sig)
    t_P = pwin_P.apply(t)

    # Ndimentional windowing :
    rep = 100
    signd = np.matlib.repmat(sig, rep, 1) + np.random.rand(rep, npts)
    sig_nd = pwin_sl.apply(signd, axis=1)

    # Plot :
    plt.subplot(211)
    plt.plot(t, sig, color='lightgray', label='Original signal')
    plt.xlabel('Time'), plt.ylabel('Amplitude')
    plt.plot(t_pwin, sig_pwin, marker='o', color='r', label='Mean inside [400ms, 450ms]')
    plt.plot(t_sl, sig_sl, marker='^', color='slateblue', label='Sliding window (unit=ms)')
    plt.plot(t_S, sig_S, marker='o', color='olive', label='Sliding window (unit=sample)')
    plt.plot(t_P, sig_P, marker='o', color='firebrick', label='Multi windows')
    plt.legend(ncol=2)

    plt.subplot(212)
    plt.pcolormesh(t_sl, np.arange(rep), sig_nd)
    plt.axis('tight'), plt.xlabel('Time'), plt.ylabel('Rep')
    plt.title('Multi-dimentional windowing')
    plt.show()
