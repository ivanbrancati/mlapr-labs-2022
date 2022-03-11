########################
#####LAB1-EXERCISE2#####
########################

#####LIBRARIES#####
import sys
import math

#####EXECUTION#####
flag = sys.argv[2][1]
if flag == 'b':
    #---Bus case---#
    busId = sys.argv[3]
    #---Reading file---#
    input = open(sys.argv[1], "r") 

    selectedLines = [line.rstrip() for line in input if line.split(" ")[0] == busId]
    xStart = int(selectedLines[0].split(" ")[2]) 
    xEnd = int(selectedLines[-1].split(" ")[2])
    yStart = int(selectedLines[-0].split(" ")[3])
    yEnd = int(selectedLines[-1].split(" ")[3])
    distance = math.sqrt((xEnd - xStart) ** 2 + (yEnd - yStart) ** 2)

    #---Printing result---#
    print("%s - Total Distance: %.1f" % (busId, distance))

    #---Closing file---#
    input.close()
elif flag == 'l':
    #---Line case---#
    lineId = sys.argv[3]
    #---Reading file---#
    input = open(sys.argv[1], "r") 

    selectedLines = [line.rstrip() for line in input if line.split(" ")[1] == lineId]
    lineBuses = list(set([elem.split(" ")[0] for elem in selectedLines]))
    speeds = [] 
    for bus in lineBuses:
        tempList = []
        tempSpeeds = []
        for line in selectedLines:
            if line.split(" ")[0] == bus:
                tempList.append(line)
        for i in range(1, len(tempList)):
            xStart = int(tempList[i-1].split(" ")[2]) 
            xEnd = int(tempList[i].split(" ")[2])
            yStart = int(tempList[i-1].split(" ")[3])
            yEnd = int(tempList[i].split(" ")[3])
            distance = math.sqrt((xEnd - xStart) ** 2 + (yEnd - yStart) ** 2)
            
            timeStart = int(tempList[i-1].split(" ")[4])
            timeEnd = int(tempList[i].split(" ")[4])
            time = timeEnd - timeStart
            
            speed = distance / time
            tempSpeeds.append(speed)
        avgSpeed = sum(tempSpeeds) / len(tempSpeeds)
        speeds.append(avgSpeed)
    avgSpeed = sum(speeds) / len(speeds)

    #---Printing result---#
    print("%s - Avg Speed: %f" % (lineId, avgSpeed))

    #---Closing file---#
    input.close()
else:
    print("Wrong flag!(-b/-l only)")
        