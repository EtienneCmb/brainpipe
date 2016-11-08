#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This script illustrate how to define a chain of transformations including :

- Pre-processing (de-trending and de-meaning)
- Filtering (lowpass, highpass and bandpass filter)
- Complex decomposition (wavelet or hilbert)
- Type extraction (amplitude, phase or power)

Then, use .apply() method to apply the transformation on a muli-dimentional
array.
"""
import numpy as np
import matplotlib.pyplot as plt

from brainpipe.features import Chain

if __name__ == '__main__':
    # --------------------------------------------------
    #                  PREPROCESSING
    # --------------------------------------------------
    # # Illustration of de-trending and de-meaning :
    sf = 1024.0   # Sampling frequency
    npts = 4000   # Number of time points
    timevec = np.arange(npts) / sf
    ct_detrend = Chain(sf, npts, detrend=True)
    ct_demean = Chain(sf, npts, demean=True)
    # Create a random signal with a linear trend :
    sig = np.arange(npts) / npts + np.random.rand(npts) + 10
    sigD = ct_detrend.apply(sig).ravel()
    sigM = ct_demean.apply(sig).ravel()
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(timevec, sig), plt.title('Original signal'), plt.xlabel('Time')
    plt.subplot(1, 3, 2)
    plt.plot(timevec, sigD), plt.title('De-trended signal'), plt.xlabel('Time')
    plt.subplot(1, 3, 3)
    plt.plot(timevec, sigM), plt.title('De-meaning signal'), plt.xlabel('Time')

    # --------------------------------------------------
    #                  FILTERING
    # --------------------------------------------------
    # Define a random signal containing multiple frequencies :
    randsig = np.random.rand(npts)
    # Bandpass filter between 2 and 5hz
    ct_band = Chain(
        sf, npts, f=[200, 300], filtname='butter', ftype='bandpass')
    sigF = ct_band.apply(randsig).ravel()
    plt.figure()
    plt.subplot(151)
    plt.plot(timevec, randsig), plt.title(
        'Original signal'), plt.xlabel('Time')
    plt.subplot(152)
    plt.plot(timevec, sigF), plt.title(
        'Bandpass filtered signal'), plt.xlabel('Time')
    plt.subplot(153)
    plt.psd(sigF, 256, Fs=sf), plt.title(
        'PSD bandpass [200, 300]hz (Butterworth)')
    # Lowpass under 200hz :
    ct_low = Chain(
        sf, npts, f=200, filtname='butter', ftype='lowpass')
    sigL = ct_low.apply(randsig).ravel()
    plt.subplot(154)
    plt.psd(sigL, 256, Fs=sf), plt.title('PSD lowpass <200hz (Butterworth)')
    # Highpass over 300hz :
    ct_high = Chain(
        sf, npts, f=300, filtname='bessel', ftype='highpass')
    sigH = ct_high.apply(randsig).ravel()
    plt.subplot(155)
    plt.psd(sigH, 256, Fs=sf), plt.title('PSD highpass >300hz (Bessel)')

    # --------------------------------------------------
    #             COMPLEX TRANSFORMATION
    # --------------------------------------------------
    # The complex decomposition is used to extract either
    # amplitude/phase/power informations of a signal. In this
    # section, we create a noisy 10hz sinus. The, we extract the amplitude,
    # phase and power around 10hz ([9, 11])
    # Create a 10hz sinus with noise :
    f = 10
    sin = np.sin(2 * np.pi * f * timevec) + np.random.rand(npts)
    # Extract amplitude :
    ct_comp = Chain(sf, npts, transname='hilbert', f=[9, 11],
                             featinfo='amplitude', filtname='butter')
    xAmp = ct_comp.apply(sin).ravel()
    # Extract phase :
    ct_comp.featinfo = 'phase'
    ct_comp.update()
    xPha = ct_comp.apply(sin).ravel()
    # Extract power :
    ct_comp.featinfo = 'power'
    ct_comp.update()
    xPow = ct_comp.apply(sin).ravel()
    # Figure :
    plt.figure()
    plt.subplot(411)
    plt.plot(timevec, sin), plt.title('Original signal')
    plt.subplot(412)
    plt.plot(timevec, xAmp, label='Amplitude',
             color='r'), plt.title('Amplitude in [9, 11]hz')
    plt.subplot(413)
    plt.plot(timevec, xPha, label='Phase',
             color='slateblue'), plt.title('Phase in [9, 11]hz')
    plt.subplot(414)
    plt.plot(timevec, xPow, label='Power',
             color='gray'), plt.title('Power in [9, 11]hz')

    # --------------------------------------------------
    #     MULTI-DIMENSION AND PARALLEL PROCESSING
    # --------------------------------------------------
    # The chain function can be applied on multi-dimentional
    # data and can be used in parallel. In this final example,
    # we define a n-dimentional array of sinus.
    # We simulate n-dimentional sinus :
    xndim = np.tile(sin, (10, 50, 1))
    # Now, we compute the amplitude using the wavelet transformation :
    f = np.arange(1, 20, 2)
    xwav = Chain(sf, npts, f=f, transname='wavelet',
                          featinfo='amplitude')
    # The parallel processing is effective on the number of frequency bands
    # defined.
    xwavAmp = xwav.apply(xndim, axis=2, n_jobs=-1)
    # Now, we show a 2D plot (equivalent to a time-frequency map),
    # and select only one dimension :
    plt.figure()
    plt.pcolormesh(timevec, f, xwavAmp[:, 0, 0, :])
    plt.xlabel('Time'), plt.ylabel('Frequency (hz)')
    plt.title(
        'Time frequency map using morlet wavelet transform on a 10hz noisy sinus')
    plt.axis('tight')
    plt.show()
