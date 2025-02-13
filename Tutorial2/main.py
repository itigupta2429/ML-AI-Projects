def swap(first, second):
    first=first+second
    second=first-second
    first=first-second
    return first, second
    

first_number=78
second_number=89
###the following print returns the tuple because it returns the 2 values and just printing
#print(swap(first_number,second_number))
#print(first_number, second_number)
first_swapped, second_swapped=swap(first_number, second_number)
###the following print returns the int because it returns the 2 values and storing it into 2 variables

print(first_swapped, second_swapped)
