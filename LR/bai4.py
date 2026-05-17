n = int(input("nhap so n nguyen duong : "))
if n <= 0:
    print("hay nhap dung so nguyen duong ")
else:
    t=1
    for i in range(1,n+1):
        t = t * i
print(t)