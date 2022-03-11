########################
#####LAB1-EXERCISE1#####
########################

#####LIBRARIES#####
import sys

#####EXECUTION#####
#---Reading file---#
input = open(sys.argv[1], "r") 

#---Computing results---#
nations = {}
competitors = []
for line in input:
    values = line.rstrip().split(" ")
    nameSurname = values[0]+" "+values[1]
    nation = values[2]
    evaluations = sorted(values[3:])[1:4]
    total = 0
    for elem in evaluations:
        total += float(elem)
    competitors.append((total, nameSurname)) ;
    if nation in nations.keys():
        nations[nation] += total
    else:
        nations[nation] = total
sortedCompetitors = sorted(competitors, reverse = True)
sortedNations = sorted([(value, key) for key, value in nations.items()], reverse = True)

#---Printing results---#
print("Final ranking:")
for i in range(3):
    print("%d: %s -- Score: %2.1f" % (i+1, sortedCompetitors[i][1], sortedCompetitors[i][0]))
print("\nBest Country:")
print("%s -- Total Score: %2.1f" % (sortedNations[0][1], sortedNations[0][0]))

#---Closing file---#
input.close() 