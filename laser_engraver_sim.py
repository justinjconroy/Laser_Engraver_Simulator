from scipy.signal import TransferFunction as tf
from scipy.signal import square
from scipy.signal import lsim

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

#Inputs:
ImageName = 'beachsmall.jpg'
OutputName = 'output.png'
RiseFallTime = 0.000050  #sec   LAser optical rise/fall time
MaxPeakPower = 100  #W    Power of the laser source
DPI = 200 #dots per inch
SpotSize = 0.2    #mm    Focused spot size achieved throught he process head
vel = 7500 #mm/sec    Velocity of the engraver
Fs = 20      #Render Steps per cycle (pixel)


print("Loading Image")
imageGray = Image.open(ImageName).convert('L') #Load image and convert to grayscale
arrayGray = np.array(imageGray, dtype=np.uint8)

NumCyc = imageGray.size[0]
NumRows = imageGray.size[1]

Freq = (vel * DPI) / 25.4
TotalTime = NumCyc / Freq
w0 = SpotSize/2.0
w2 = w0 ** 2

#setup the transfer H function  - Simple RC filter which simulates rise/fall of CO2 laser nicely
#ignore the overzealous scipy warnings about badcoeffcients. This transfer function operates correctly
H = tf([0, 0, 0, 1.0],[0, 0, (RiseFallTime/np.log(9)), 1.0])  #10% to 90% rise fall definition

t = np.linspace(0, TotalTime, num=Fs * NumCyc)

FsPerMM = DPI * Fs / 25.4
rowStep = int(2 * w0 * FsPerMM * 1.57)
columnStep = int(2 * w0 * FsPerMM * 1.57)
engraveResult = np.zeros(((Fs * NumRows)+rowStep-Fs, (Fs * NumCyc)+columnStep))


y = np.linspace(-w0*1.57, w0*1.57, rowStep)
x = np.linspace(-w0*1.57, w0*1.57, columnStep)
X,Y = np.meshgrid(x,y, indexing='xy')


rowPixel = 0
rowStart = 0
rowEnd = rowStep
while rowPixel < NumRows:
	
	DutyCycle = np.repeat((arrayGray[rowPixel,:] / 255)*100,Fs) 
	
	#Create the square wave train
	s = (MaxPeakPower/2) * ((1+square(2*np.pi*Freq*t, duty=DutyCycle/100)))


	#Simulate LTI model - apply the RC filter to sqr wave
	pwr = lsim(H, s, t)[1]


	columnPixel = 0
	intensity = np.zeros((rowStep, (Fs * NumCyc)+columnStep))
	
	while columnPixel < NumCyc:
		print("Processing Row: " + str(rowPixel+1) + "/" + str(NumRows) + "  Processing Pixel: " + str(columnPixel+1) + "/" + str(NumCyc), end="\r")
		currentF = 0
		while currentF < Fs:
			i = (columnPixel*Fs)+currentF
			
			#calculate Gaussian Heat Flux
			intensity[:,i:i+columnStep] = intensity[:,i:i+columnStep] + (2*pwr[i]/(np.pi * (w2)))*np.exp(-2 * ((np.square(X)) + np.square(Y))/(w2))

			currentF += 1
		columnPixel += 1 

		 
	rowEnd = rowStart + (rowStep)
	
	engraveResult[rowStart:rowEnd,:] = engraveResult[rowStart:rowEnd,:] + intensity
	rowPixel += 1
	rowStart = rowStart+Fs 


print("Freq: ", Freq, "  Total Time Per Pixel:", TotalTime)
print ("max pixel:", np.amax(engraveResult), "  min pixel: ", np.amin(engraveResult))
plt.imshow(engraveResult, cmap='gray')
plt.savefig(OutputName)
plt.show()


