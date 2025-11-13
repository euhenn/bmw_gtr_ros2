import numpy as np
import matplotlib.pyplot as plt

class VehicleParams():
    def __init__(self):
        self.g = 9.81
        self.mi = 0.9

        #air charachteristics for Bosch car
        self.ro = 1.2
        self.Cz = 0.32
        self.Az=  0.021
        self.Az=  0.4

    def JaguarX(self):
        self.lf, self.lr = 1.432, 1.472 # distance from CoG to front and rear wheels
        self.L = self.lr + self.lf # wheel base

        self.m, self.I_z = 2200, 3344 
        self.h = 0.4 
        self.rr = 0.333

        #have to be calculated  or guessed properly
        self.Bcf, self.Ccf, self.Dcf = 0.2, 1.3, 4870 
        self.Bcr, self.Ccr, self.Dcr = 0.18, 1.3, 4750 

        self.Blf, self.Clf, self.Dlf = 0.2, 1.65, 5620 
        self.Blr, self.Clr, self.Dlr = 0.2, 1.65, 5480 
        self.Cm1, self.Cm2 = 40.0, 5.05

    def RaceCar43(self):
        self.lf, self.lr = 0.0308, 0.0305 # distance from CoG to front and rear wheels
        self.L = self.lr + self.lf # wheel base

        self.m, self.I_z = 0.0467, 5.6919 *1e-5
        self.c1,self.c2 = 0.5, 17.06 # geometrical values-lr/l,  1/l

        self.Bcf, self.Ccf, self.Dcf = 3.47, 0.1021, 5.003  
        self.Bcr, self.Ccr, self.Dcr = 3.173, 0.01921, 19.01 

        self.Cm1, self.Cm2 = 12, 2.17
        self.cr0, self.cr2, self.cr3 = 0.01, 0.006, 5
        



    def RandomModel(self):
        self.lf, self.lr = 1.035, 1.655  # distance from CoG to front and rear wheels
        self.L = self.lr + self.lf # wheel base

        self.m, self.I_z = 1704.7, 3048.1 
        self.h = 0.4 

        #have to be calculated  or guessed properly
        self.Bcf, self.Ccf, self.Dcf = 9.094, 1.193, 4870 #8.0, 1.5, 6000
        self.Bcr, self.Ccr, self.Dcr = 10.11, 1.193, 3273 # 10.0, 1.4, 7000

        self.Blf, self.Clf, self.Dlf =11.39, 1.685, 6164 #8.0, 1.5, 6000
        self.Blr, self.Clr, self.Dlr = 0.2, 1.685, 3912 # 10.0, 1.4, 7000
        self.Cm1, self.Cm2 = 40.0, 5.05

    def BoschCar(self):
        self.lf,self.lr = 0.13, 0.13
        self.L = self.lf+self.lr
        self.m = 1.415
        self.h = 0.03
        self.I_z = 0.17423
        self.Bcf, self.Ccf, self.Dcf = 0.425, 1.3, 6.246 #0.23, 1.57, 6.94
        self.Bcr, self.Ccr, self.Dcr = 0.425, 1.3, 6.246


if __name__ == "__main__":
    vp = VehicleParams()
    vp.JaguarX()  

    print(vp.m)   
    print(vp.lf) 
    

