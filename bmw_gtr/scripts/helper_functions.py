#!/usr/bin/python3

# HELPER FUNCTIONS
import numpy as np
import cv2 as cv
import time

def diff_angle(angle1, angle2):
    return np.arctan2(np.sin(angle1-angle2), np.cos(angle1-angle2))

const_verysmall = 3541/15.0
#M_R2L = np.array([[1.0, 0.0], [0.0, -1.0]])   #BFMC_2023
#T_R2L = np.array([0, 15.0])                    #BFMC_2023    conversion between frames
                                            #the bosch graph needed a change of coordinates
M_R2L = np.array([[1.0, 0.0], [0.0, 1.0]])      #BFMC_2024
T_R2L = np.array([0, 0.0])                      #BFMC_2024


def mL2pix(ml):
    # meters to pixel (left frame)
    return np.int32(ml*const_verysmall)


def mR2pix(mr):
    # meters to pixel (right frame), return directly a cv point
    if mr.size == 2:
        pix = mL2pix((mr - np.array([0, 15.0])) @ np.array([[1.0, 0.0], [0.0, -1.0]])) 
        return (pix[0], pix[1])
    else:
        return mL2pix((mr - np.array([0, 15.0])) @ np.array([[1.0, 0.0], [0.0, -1.0]]))

        

def pix2mL(pix):
    # pixel to meters (left frame)
    return pix/const_verysmall


def mL2mR(m):
    # meters left frame to meters right frame
    return m @ M_R2L + T_R2L


def mR2mL(m):
    # meters right frame to meters left frame
    return (m - T_R2L) @ M_R2L