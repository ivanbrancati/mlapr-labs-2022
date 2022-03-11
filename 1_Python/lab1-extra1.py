########################
#####LAB1-EXERCISE1#####
########################

#####LIBRARIES#####
import sys

#####EXECUTION#####
#---Reading file---#
input = open(sys.argv[1], "r")

#---Preparing dictionaries---#
lines = [line.rstrip() for line in input]
availableCopies = {}
monthlySold = {}
pricesBought = {}
pricesSold = {}

months={"01" : "January", "02" : "February", "03" : "March", "04" : "April", "05" : "May", "06" : "June", "07" : "July", "08" : "August", "09" : "September", "10" : "October", "11" : "November", "12" : "December"}

#---Updating dictionaries---#
for line in lines:
    isbn, code, date, num, price = line.split(" ")
    month = date[3:]
    if (code == "B"):
        availableCopies[isbn] = availableCopies.get(isbn, 0) + int(num)
        oldPrice, oldNum = pricesBought.get(isbn, [0.0, 0])
        pricesBought[isbn] = [oldPrice + (float(price) * int(num)), oldNum + int(num)]
    else:
        availableCopies[isbn] = availableCopies.get(isbn, 0) - int(num)
        monthlySold[month] = monthlySold.get(month, 0) + int(num)
        oldPrice, oldNum = pricesSold.get(isbn, [0.0, 0])
        pricesSold[isbn] = [oldPrice + (float(price) * int(num)), oldNum + int(num)]
        
#---Printing results---#        
print("Available Copies:")
for elem in availableCopies.keys():
    print("\t%s: %d" % (elem, availableCopies.get(elem)))

print("Sold books per month:")
for elem in monthlySold.keys():
    month, year = elem.split("/")
    print("\t%s, %s: %d" % (months.get(month), year, monthlySold.get(elem)))
       
print("Gain per book:")
for elem in pricesSold.keys():
    avg = pricesBought.get(elem)[0] / pricesBought.get(elem)[1]
    avgGain = (pricesSold.get(elem)[0] / pricesSold.get(elem)[1]) - avg
    num = pricesSold.get(elem)[1]
    gain = pricesSold.get(elem)[0] - (avg * num)
    print("\t%s: %.1f (avg %.1f, sold %d)" % (elem, gain, avgGain, num))

#---Closing file---#
input.close()