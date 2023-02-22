#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:49:39 2022

@author: matthew
"""

KK2pos = []
KK1pos = []
POC_CCA_pos = []
nSurveySamples = 10000
for i in range(nSurveySamples):
    KK1pos.append( conductKKSurvey(SD, params, t, sampleSize, nSamples, 'KK1'))
    KK2pos.append( conductKKSurvey(SD, params, t, sampleSize, nSamples, 'KK2'))
    POC_CCA_pos.append(conductPOCCCASurvey(SD, params, t, sampleSize, nSamples))

print("KK1 mean positivity = ", np.mean(KK1pos))
print("KK2 mean positivity = ", np.mean(KK2pos))
print("POC_CCA_pos mean positivity = ", np.mean(    POC_CCA_pos))

print("KK1 sqrt positivity = ", np.sqrt(np.var(KK1pos)))
print("KK2 sqrt positivity = ", np.sqrt(np.var(KK2pos)))
print("POC_CCA_pos sqrt positivity = ", np.sqrt(np.var(POC_CCA_pos)))

