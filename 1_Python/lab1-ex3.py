########################
#####LAB1-EXERCISE2#####
########################

#####LIBRARIES#####
import sys

#####EXECUTION#####
#---Reading file---#
input = open(sys.argv[1], "r") 

months = {"01":"January", "02":"February", "03":"March", "04":"April", "05":"May", "06":"June", "07":"July", "08":"August", "09":"September", "10":"October", "11":"November", "12":"December"}
citiesBirths = {}
monthsBirths = {}

for line in input:
    values = line.rstrip().split(" ")
    month = values[3].split("/")[1] 
    city = values[2] 
    if month in monthsBirths.keys():
        monthsBirths[month] += 1
    else:
        monthsBirths[month] = 1
    if city in citiesBirths.keys():
        citiesBirths[city] += 1
    else:
        citiesBirths[city] = 1
        
#---Printing result---#
print("Births per city:")
for elem in citiesBirths.items():
    print("%s: %d" % (elem[0], elem[1]))

print("\nBirths per month:")
for elem in monthsBirths.items():
    print("%s: %d" % (months.get(elem[0]), elem[1]))
sum = 0
for elem in citiesBirths.values():
    sum += elem
print("\nAverage number of births: %f" % (sum/len(citiesBirths)))    

#---Closing file---#
input.close()