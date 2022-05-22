"""
This file contains code related to analyzing the correlation between $GME and $OSTK.

The $GME and $OSTK data were downloaded from Yahoo Finance, collecting all data 
in the range of May 18, 2017 - May 17, 2022 according to the 'Hisorical Data' 
page in Yahoo Finance.  
However, once downloading the data, the dates shift by a day, 
making the range May 19, 2017 - May 18, 2022.  

Special shoutout to u/uprclass2002 for doubling down on a bogus claim about 
GME and OSTK being correlated, which inspired me to do this quick analysis.

Best coding practice was sacrificed in the interest of completing this quickly.
"""

import sys, os
import numpy as np
import scipy.stats as ss
from dtw import dtw, DTW
import matplotlib.pyplot as plt

plt.ion()


def readdata(foo):
    with open(foo, 'r') as foop:
        lines = foop.readlines()
    labels = lines[0].replace('\n', '').split(',')
    dat = np.zeros((len(lines)-1, len(labels)-1))
    dates = []
    for i, line in enumerate(lines[1:]):
        splt = line.replace('\n', '').split(',')
        dates.append(splt[0])
        dat[i] = [float(val) for val in splt[1:]]
    return labels, dates, dat


def lagged_corr(a, b):
    f = np.zeros(3*a.size)
    g = np.zeros(3*b.size)
    f[a.size:2*a.size] = a - a.mean()
    g[b.size:2*b.size] = b - b.mean()
    lags = np.linspace(-a.size, a.size, 2*a.size+1, dtype=int)
    corr = np.zeros(lags.size, dtype=float)
    for i in range(lags.size):
        corr[i] = np.correlate(f, np.roll(g, -lags[i]))
    return lags, corr


# Megascript to go through all the cases of interest.
# GME and OSTK historical data from Yahoo Finance
labels, dates, gmedat  = readdata('GME.csv')
labels, dates, ostkdat = readdata('OSTK.csv')
# Other tickers for comparison
labels, dates, spydat  = readdata('SPY.csv')
labels, dates, amcdat  = readdata('AMC.csv')
labels, dates, tsladat = readdata('TSLA.csv')
labels, dates, aapldat = readdata('AAPL.csv')
labels, dates, xrtdat  = readdata('XRT.csv')

plt.plot(gmedat [:,-2], label='GME')
plt.plot(ostkdat[:,-2], label='OSTK')
plt.legend(loc='best')
plt.xlabel('Trading days since May 19, 2017')
plt.ylabel('Adj. Closing Price ($)')
plt.savefig("gme_ostk_5y_data.png")
# Visually, we can see a glaring anti-correlation when GME dropped to $40 after the sneeze
# But, maybe it's offset, so let's withhold judgment until later.
# Plot some other tickers for comparison.
plt.plot(spydat [:,-2], label='SPY')
plt.plot(amcdat [:,-2], label='AMC')
plt.plot(tsladat[:,-2], label='TSLA')
plt.plot(aapldat[:,-2], label='AAPL')
plt.plot(xrtdat [:,-2], label='XRT')
plt.legend(loc='best')
plt.savefig("gme_ostk_others_5y_data.png")
plt.close()
# By eye, popcorn looks like the strongest correlation, 
# but we know that things changed at some point when they went long on popcorn.
# Fuck popcorn, all my homies hate popcorn.

# Pearson correlation over all time
r, p = ss.pearsonr(gmedat[:,-2], ostkdat[:,-2])
print("GME-OSTK overall Pearson r, p:", r, p)
# Wow, it's high. What about the others?
r, p = ss.pearsonr(gmedat[:,-2], spydat[:,-2])
print("GME-SPY overall Pearson r, p:", r, p)
r, p = ss.pearsonr(gmedat[:,-2], amcdat[:,-2])
print("GME-AMC overall Pearson r, p:", r, p)
r, p = ss.pearsonr(gmedat[:,-2], tsladat[:,-2])
print("GME-TSLA overall Pearson r, p:", r, p)
r, p = ss.pearsonr(gmedat[:,-2], aapldat[:,-2])
print("GME-AAPL overall Pearson r, p:", r, p)
r, p = ss.pearsonr(gmedat[:,-2], xrtdat[:,-2])
print("GME-XRT overall Pearson r, p:", r, p)
# Look at that, GME is 'more correlated' with the other tickers I picked!
# This is because the main correlation we're seeing is overall market trend.
# This is the first indication that u/uprclass2002 is wrong.

# But, not totally, because the Pearson coeff is bullshit for stocks.  
# I've seen a few apes use this in the past, and it really pains me to see it.
# Let's illustrate why.  
# Above, the GME-OSTK Pearson coeff was over 0.63.
# So, we should expect that, if we split the data into two sections, 
# one of them should have *at least* that coeff, right?

# Pearson correlation up to May 13, 2020
r, p = ss.pearsonr(gmedat[:750,-2], ostkdat[:750,-2])
print("Pearson r, p before May 13, 2020:", r, p)

# Pearson correlation since May 13, 2020
r, p = ss.pearsonr(gmedat[750:,-2], ostkdat[750:,-2])
print("Pearson r, p since May 13, 2020:", r, p)

# Wrong.  0.53 before May 13 2020, 0.35 after that date.
# Why?  For stonks, the Pearson coeff is just looking at the longterm growth.
# It might make more sense if we were analyzing data on a shorter timescale.

# Does the cutoff point matter?  
# Maybe I just picked a special date that caused this behavior.

# Pearson correlation up to March 3, 2020
r, p = ss.pearsonr(gmedat[:700,-2], ostkdat[:700,-2])
print("Pearson r, p before March 3, 2020:", r, p)

# Pearson correlation since March 3, 2020
r, p = ss.pearsonr(gmedat[700:,-2], ostkdat[700:,-2])
print("Pearson r, p since March 3, 2020:", r, p)

# Pearson correlation up to Jan 4, 2021
r, p = ss.pearsonr(gmedat[:912,-2], ostkdat[:912,-2])
print("Pearson r, p before Jan 4, 2021:", r, p)

# Pearson correlation since Jan 4, 2021
r, p = ss.pearsonr(gmedat[912:,-2], ostkdat[912:,-2])
print("Pearson r, p since Jan 4, 2021:", r, p)

# We see that in all these cases, the Pearson coeff is less over 
# these smaller regions, compared to the overall value over 5 years.
# This is yet another piece of evidence why Pearson coeff is *not* a good 
# measure of correlation for stonks, at least not directly, 
# because it's really just measuring the overall 'stonks go up' market 
# we've been in for over a decade.

# So, what about indirectly?  
# We know that stonks go up.  If we can remove the general market trends, 
# then we can get a better idea of whether GME is correlated with some 
# other stock, like OSTK.
# As a simple test of this, let's normalize the GME and OSTK data by 
# dividing by SPY
r, p = ss.pearsonr(gmedat[:,-2]/spydat[:,-2], ostkdat[:,-2]/spydat[:,-2])
print("GME-OSTK normalized by SPY overall Pearson r, p:", r, p)
r, p = ss.pearsonr(gmedat[:,-2]/spydat[:,-2], amcdat[:,-2]/spydat[:,-2])
print("GME-AMC normalized by SPY overall Pearson r, p:", r, p)
r, p = ss.pearsonr(gmedat[:,-2]/spydat[:,-2], tsladat[:,-2]/spydat[:,-2])
print("GME-TSLA normalized by SPY overall Pearson r, p:", r, p)
r, p = ss.pearsonr(gmedat[:,-2]/spydat[:,-2], aapldat[:,-2]/spydat[:,-2])
print("GME-AAPL normalized by SPY overall Pearson r, p:", r, p)
# That didn't change things.  You can do more sophisticated things like 
# converting to percent changes, but at the end of the day, it's the same story.

# So if Pearson coeff is bullshit, what *should* we do to find correlation
# between two stonks?

# 1. Lagged cross-correlation https://en.wikipedia.org/wiki/Cross-correlation
# 2. Dynamic time warping (DTW) https://en.wikipedia.org/wiki/Dynamic_time_warping

# In both cases, if two stonks are truly correlated, 
# we should expect the maximum cross-correlation to occur at a lag of 0, 
# or close to it.
# If the maximum correlation occurs far from 0, then that could mean that the 
# tickers are offset in time.

# Simple lagged cross-correlation
lags, laggedcorr = lagged_corr(gmedat[:,-2], ostkdat[:,-2])
ilag = lags[np.argmax(laggedcorr)]
print("Maximum cross-correlation occurs for a lag of", ilag)

plt.plot(lags, laggedcorr)
plt.xlabel("Lag")
plt.ylabel("Cross-correlation")
plt.tight_layout()
plt.savefig("crosscorr.png")
plt.close()

# Max is a lag of -110 days! Let's plot it
ostk_lagged = np.roll(ostkdat[:,-2], -ilag)
ostk_lagged[:-ilag] = 0
plt.plot(gmedat [:,-2], label='GME')
plt.plot(ostk_lagged,   label='OSTK, lagged for max cross-correlation')
plt.legend(loc='upper left')
plt.xlabel('Trading days since May 19, 2017')
plt.ylabel('Adj. Closing Price ($)')
plt.savefig("gme_ostk_5y_data_lagged-max-cross-corr.png")
plt.close()

# What about that bump near 0?
lccopy = np.copy(laggedcorr)
lccopy[lccopy > 1667768.3] = 0. #bad code, but it works for what we need
ilag2 = lags[np.argmax(lccopy)]
print("Closest local maximum to 0:", ilag2) #-28
ostk_lagged = np.roll(ostkdat[:,-2], -ilag2)
ostk_lagged[:-ilag2] = 0
plt.plot(gmedat [:,-2], label='GME')
plt.plot(ostk_lagged,   label='OSTK, cross-corr local max closest to 0 lag')
plt.legend(loc='upper left')
plt.xlabel('Trading days since May 19, 2017')
plt.ylabel('Adj. Closing Price ($)')
plt.savefig("gme_ostk_5y_data_lagged-localmax-cross-corr.png")
plt.close()

# So, the absolute max aligns OSTK's spike with the GME spike
# and the local max near 0 aligns GME's post-spike dip with one in OSTK
# But in both cases, the cross-correlation maximum requires a >month shift
# And though it improves our previously used Pearson coeff, it is still less 
# than some of the other tickers we compared with.

# We know XRT is linked to GME, so it is no surprise that 
# XRT has the highest Pearson corr with GME.
# What do we get when using cross-corr between them?
lags, laggedcorr = lagged_corr(gmedat[:,-2], xrtdat[:,-2])
ilag = lags[np.argmax(laggedcorr)]
print("Maximum GME-XRT cross-correlation occurs for a lag of", ilag)

plt.plot(lags, laggedcorr)
plt.xlabel("Lag")
plt.ylabel("Cross-correlation")
plt.tight_layout()
plt.savefig("crosscorr_gme-xrt.png")
plt.close()

xrt_lagged = np.roll(xrtdat[:,-2], -ilag)
xrt_lagged[:-ilag] = 0
plt.plot(gmedat [:,-2], label='GME')
plt.plot(xrt_lagged,   label='OSTK, lagged for max cross-correlation')
plt.legend(loc='upper left')
plt.xlabel('Trading days since May 19, 2017')
plt.ylabel('Adj. Closing Price ($)')
plt.savefig("gme_xrt_5y_data_lagged-max-cross-corr.png")
plt.close()
# Our original hypothesis of a lag~0 for correlated stocks came true.

# There are more involved approaches to cross-correlation of course, 
# but cross-corr assumes that the stonk algos would exactly correspond 
# in terms of frequency.
# But, what if the pattern is over a longer period of time 
# for one stock vs the other?

# DTW to the rescue.
# DTW can measure similarity (correlation) even if the speeds differ.
# It has a lot of other great uses too
# Here, we are going to use it to map between the derivatives of 
# each ticker adj. closing price
# Why derivative? Because DTW is sensitive to the magnitude of the signals.
# The derivative will reduce the data to "did it go up or down?", and while 
# magnitude still plays a role, it is far less substantial.
# If you're curious to read more, see Keogh & Pazzani
# https://doi.org/10.1137/1.9781611972719.1
# as well as papers that have cited that one
align = dtw(np.diff(ostkdat[:,-2]), np.diff(gmedat[:,-2]), keep_internals=True)
align.plot(type='threeway')
plt.savefig("dtw_gme-ostk.png")
plt.close()

# For convenience, here are 2 examples of DTW for correlated series, 
# modified from the dtw package's quick start for simplicity
idx = np.linspace(0,6.28,num=100)
query    = np.sin(idx)
template = np.cos(idx) #offset by pi/2, or 25 samples
alignment = dtw(query, template, keep_internals=True)
alignment.plot(type="threeway")
plt.savefig("dtw_sin-cos.png")
plt.close()
# We see that the top-right plot doesn't begin increasing until a query index 
# of 25.  This is the exact offset in our example.

# Let's do it with differing speeds now.
query    = np.sin(2*idx)
alignment = dtw(query, template, keep_internals=True)
alignment.plot(type="threeway")
plt.savefig("dtw_sin2x-cosx.png")
plt.close()
# We see that it begins increasing earlier now, 
# at index 13 (factor of 2 change from before)

# What happens when the data isn't perfect, and there's some noise?
# https://dynamictimewarping.github.io/py-images/Figure_1.png
# It still looks mostly smooth, but with some noise.

# Let's return to the GME-OSTK plot now that we know what to look for.
# We see the plot begins increasing at a query index of perhaps 375.  
# But, generally, it looks like a step-wise plot, not the smooth increase 
# we saw before.
# What does DTW look like for two totally unrelated series?
np.random.seed(0)
query = np.random.uniform(size=100)
alignment = dtw(query, template, keep_internals=True)
alignment.plot(type="threeway")
plt.savefig("dtw_random-cos.png")
plt.close()
# Looks pretty familiar.  Let's get multiple cycles of the cosine.
template = np.cos(3*idx)
alignment = dtw(query, template, keep_internals=True)
alignment.plot(type="threeway")
plt.savefig("dtw_random-cos3x.png")
plt.close()
# And that looks a lot like the GME-OSTK DTW plot, doesn't it?
# If we had two totally random data sets, 
# we'll go back to seeing a noisy but steady increase.
# This step-wise behavior occurs for data that come from totally different 
# distributions, and would not be expected for correlated stocks. 

# But does DTW show any correlation between tickers?
# How about GME-SPY?
align = dtw(np.diff(spydat[:,-2]), np.diff(gmedat[:,-2]), keep_internals=True)
align.plot(type='threeway')
plt.savefig("dtw_gme-spy.png")
plt.close()
# How about GME-XRT?  We know there is a strong connection there.
align = dtw(np.diff(xrtdat[:,-2]), np.diff(gmedat[:,-2]), keep_internals=True)
align.plot(type='threeway')
plt.savefig("dtw_gme-xrt.png")
plt.close()



