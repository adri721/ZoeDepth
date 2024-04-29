"""
Data format for Ranging package Data

| Needle | Data Length High byte | Data Length Low Byte | Protocol Type 
| Command Type | Command ID | Parameter Length High Byte | Parameter Length Low Byte 
| Speed | Angle Offset High Byte | Angle Offset Low Byte 
| Current frame angle start High Byte | Current Frame Angle Low Byte 
| Current frame end angle High Byte | Current frame end angle Low Byte 
| Ranging Point 1 energy | Ranging Point 1 lenght High Byte | Ranging Point 1 length Low Byte 
| Ranging Point 2 energy | Ranging Point 2 lenght High Byte | Ranging Point 2 length Low Byte 
| ... 
| Ranging Point n energy | Ranging Point n lenght High Byte | Ranging Point n length Low Byte 
| Check digit high byte | Check digit low byte |


Needle              : fixed at 0xAA

Data Length         : All the data length from the needle to one byte before the check digit

Protocol Type       : 0x10

Command type        : 0x61

Command ID          : 0xAD

Parameter length    : length of all data except thhe check digit after this byte

Speed               : Actual Speed x20 (rev/sec)

Zero angle offset   : Absolute angle offset x100 (deg)

Current frame 
    start angle     : Starting angle of the current frame x100 (degs)
    
Current frame 
    end angle       : Ending angle of the current frame x100 (degs)
    
Energy of 
    ranging point 1 : Energy of the first ranging point
    
Distance of 
    ranging point 1 : Distance data of the first ranging point x4 (mm)
    
Energy of 
    ranging point 2 : Energy of the second ranging point
    
Distance of 
    ranging point 2 : Distance data of the second ranging point x4 (mm)
    
...

Energy of 
    ranging point n : Energy of the n-th ranging point

Distance of 
    ranging point n : Distance data of the n-th ranging point x4 (mm)
    
Check Digit         : Checksum of thhe all data from the needle to the previous byte




* the number of measurement points can be estimated from the data lengh or parameter length. *




Data format for Stall package Data

| Needle | Data Length High byte | Data Length Low Byte | Protocol Type | Command Type 
| Command ID | Parameter Length High Byte | Parameter Length Low Byte | Speed 
| Product UID | Radar Information | Check digit high byte | Check digit low byte


Needle              : fixed at 0xAA

Data Length         : All the data length from the needle to one byte before the check digit

Protocol Type       : 0x10

Command type        : 0x61

Command ID          : 0xAE

Parameter length    : length of all data except thhe check digit after this byte

Speed               : Actual Speed x20 (rev/sec)

Product UID         : 24 bytes long and uses the UID of the MCU. 
                    : Sugkawa Radar Product Information Coding Rules

Radar Information   : 31 bytes long and includes the radar type, MCU model, 
                    : software and hardware version

Check Digit         : Checksum of thhe all data from the needle to the previous byte

"""




import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import time

plt.axes(projection = 'polar')


rows=[]

filename = "C:/Users/ARajaraman/OneDrive - SharkNinja/Documents/pythontests/ZoeDepth/Intel_data/2024_4_17_17_2_3/lidar/0.csv"

## reading raw data byte by byte
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        try:
            f= float(row[0])
            rows.append(f)
        except:
            continue    
        


ii = 0

## the following are all lists and their defiition is given
## if the defition mentions a list, then the resultant is a list of lists

l_p=[]          # length of packet from the needle head to one byte before the check byte

ao=[]           # absolute angle offset
cfs=[]          # starting angle of currrent frame
cfe=[]          # end angle of current frame
rangA=[]        # list of ranging energies in the packet
rangB=[]        # list of ranging lengths in the packet
rangC=[]        # ranging lengths of the entire dataset
rangD=[]        # ranging energies of the entire dataset
anf=[]          # list of interpolated angles for the ranging lengths in the packet
angF=[]         # interpolated angles of the entire dataset to correspong for each ranging length 
angR=[]         # aforementioned data in radians 
angN=[]

# #
# the 3i LiDAR has two modes of operation
#     -> ranging mode: when the LiDAR is operating normally
#     -> stall mode:   when the LiDAR is stalled (not only stopped but either too fast or too slow)

# the following lists separate the two types of data packets and also categorise packets that
# don't fit the above two formats


mainList = []       # list of all the data in a packet (irrespective of which mode)    
Lsize = []          # length of the packet
sp=[]               # speed of LiDAR in that packet
ranList = []        # list of the data when LiDAR is in ranging mode
stallList = []      # list of the data when LiDAR is in stall mode
otherList = []      # list of the data when LiDAR is in not in the two modes

RLindex = []        # list containg the indices for the ranging packets in thhe entire dataset
SLindex = []        # list containg the indices for the stall packets in thhe entire dataset
OLindex = []        # list containg the indices for the unidentified packets in thhe entire dataset

rSpeed = []         # list of the speeds in the packet when the LiDAR is in the ranging mode
sSpeed = []         # list of the speeds in the packet when the LiDAR is in the stall mode

pUID = []           # list of the product UID in the stall packets (showuld be same for every packet)
radarInfo = []      # list of the radar info in the stall packets (showuld be same for every packet)

rCS = []            # list of the checksum in the ranging packets
sCS = []            # list of the checksum in the stall packets


# count variable
cnt = 0

for ii in range(len(rows)-200):
    if (rows[ii] == 170 and (rows[ii+2]-rows[ii+7])==8):
        lenP = int(rows[ii+1]*256 + rows[ii+2] + 2)
        Lsize.append(lenP)
        mainList.append(rows[ii:ii+lenP])
        cnt+=1
        
        speed = rows[ii+8]
        # checkSum = rows[ii+lenP-2]*256 + rows[ii+lenP-1]
        
        if(rows[ii+3]==16 and rows[ii+4]==97 and rows[ii+5]==173):
            ranList.append(rows[ii:ii+lenP])
            RLindex.append(cnt-1)
            rSpeed.append(speed)
            # rCS.append(checkSum)
            angleOffset = rows[ii+9]*256 + rows[ii+10]
            ao.append(angleOffset/100)
            # print(angleOffset)
            currentFrameStartAngle = rows[ii+11]*256 + rows[ii+12]
            cfs.append(currentFrameStartAngle/100)
            currentFrameEndAngle = rows[ii+13]*256 + rows[ii+14]
            cfe.append(currentFrameEndAngle/100)
            jj = 0
            ranging_a = []
            ranging_b = []
            while(jj<lenP - 18):
                # ranging_a = []
                # ranging_b = []
                ranging_a.append(rows[ii+jj+15])
                rangD.append(rows[ii+jj+15])
                ranging_b.append((rows[ii+jj+16]*256 + rows[ii+jj+17])/4)
                rangC.append((rows[ii+jj+16]*256 + rows[ii+jj+17])/4)
                # ranging_b.append((rows[ii+jj+17]*256 + rows[ii+jj+16])/4)
                # rangC.append((rows[ii+jj+17]*256 + rows[ii+jj+16])/4)
                # ranging_b.append(ranging_a)
                
                jj+=3
            rangA.append(ranging_a)
            rangB.append(ranging_b)
            
            lenF = len(ranging_b)
            
            anf.append(np.linspace(currentFrameStartAngle/100, currentFrameEndAngle/100, lenF))
            for kk in anf[-1]:
                angF.append(kk)
                angR.append(kk*math.pi/180)
                
        elif(rows[ii+3]==16 and rows[ii+4]==97 and rows[ii+5]==174): 
            stallList.append(rows[ii:ii+lenP])
            SLindex.append(cnt-1)
            sSpeed.append(speed)
            pUID.append(rows[ii+9:ii+32])
            radarInfo.append(rows[ii+33:ii+73])
            # sCS.append(checkSum)
            
            
        else:
            otherList.append(rows[ii:ii+lenP])
            OLindex.append(cnt-1)


angPlot = []
distPlot = []
scanCnt = 0
print(len(angF))
for ii in range(100, len(angF)):
    if(angF[ii]>225 and angF[ii]<315):
        angPlot.append(angR[ii])
        distPlot.append(rangC[ii])
        scanCnt= 1
    elif (scanCnt == 1):
        break



#plotting the second scan data points in a polar plot
fig1 = plt.figure(1)

plt.plot(angPlot, distPlot)
plt.title('3i LiDAR distance plot')


# fig2 = plt.figure(2)
# plt.polar(angR[bias+492:bias+984], rangD[bias+492:bias+984])
# plt.title('3i LiDAR signal strength plot')

plt.show()
