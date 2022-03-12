########################
#####LAB1-EXERCISE1#####
########################

#####LIBRARIES#####
import sys

#####EXECUTION#####
#---Reading file---#
input = open(sys.argv[1], "r")

#---Preparing structures---#
lightspots = [line.rstrip().split(" ") for line in input]
N = 7
tiles = [[0 for i in range(N)]for i in range (N)]

#---Tiles Updating---#
for i, j in lightspots:
    i = int(i)
    j = int(j)
    if(i < N-1 and j < N-1):
        #1
        tiles[i][j] += 10
        #0.5
        #--Row = i-1
        if(i-1 >= 0):
            if(j-1 >= 0):
                tiles[i-1][j-1] += 5
            tiles[i-1][j] += 5
            if(j+1 < N-1):
                tiles[i-1][j+1] += 5

        #--Row = i        
        if(j-1 >= 0):
            tiles[i][j-1] += 5
        if(j+1 < N-1):
            tiles[i][j+1] += 5

        #--Row = i+1
        if(i+1 < N-1):
            if(j-1 >= 0):
                tiles[i+1][j-1] += 5
            tiles[i+1][j] += 5
            if(j+1 < N-1):
                tiles[i+1][j+1] += 5
        #0.2
        #--Row = i-2
        if(i-2 >= 0):
            if(j-2 >= 0):
                tiles[i-2][j-2] += 2
            if(j-1 >= 0):
                tiles[i-2][j-1] += 2
            tiles[i-2][j] += 2
            if(j+1 < N-1):
                tiles[i-2][j+1] += 2
            if(j+2 < N-1):
                tiles[i-2][j+2] += 2

        #--Row = i-1
        if(i-1 >= 0):
            if(j-2 >= 0):
                tiles[i-1][j-2] += 2
            if(j+2 < N-1):
                tiles[i-1][j+2] += 2

        #--Row = i
        if(j-2 >= 0):
            tiles[i][j-2] += 2
        if(j+2 < N-1):
            tiles[i][j+2] += 2

        #--Row = i+1
        if(i+1 < N-1):
            if(j-2 >= 0):
                tiles[i+1][j-2] += 2
            if(j+2 <N-1):
                tiles[i+1][j+2] += 2


        #--Row = i+2
        if(i+2 < N-1):
            if(j-2 >= 0):
                tiles[i+2][j-2] += 2
            if(j-1 >= 0):
                tiles[i+2][j-1] += 2
            tiles[i+2][j] += 2
            if(j+1 < N-1):
                tiles[i+2][j+1] += 2
            if(j+2 < N-1):
                tiles[i+2][j+2] += 2
    else:
        print("Wrong indices encountered: %d, %d" %(i, j))

#---Printing Results---#
print("Output:")
# Dividing by 10 (otherwise problems with floats)            
for i in range(len(tiles)):
    print()
    for j in range(len(tiles[i])):
        tiles[i] [j] /= 10
        print(tiles[i][j], end=" ")

#---Closing file---#        
input.close()