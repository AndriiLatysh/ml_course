#1
a = int(input("Your number:"))
max = 0
for i in range(a):
    k = int(input("num {i} "))
    max = k if k > max else k
print(max)
print (k)
