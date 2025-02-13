#3 essential tools of programming: 1) assigning variables; 2) if else statements; 3) loops
def calculate(first, second, op):
    if (op=="+"):
        return first + second
    elif (op == '-'):
        return first - second
    elif (op == '*'):
        return first * second
    elif (op == '/'):
        if second != 0:
            return first / second
        else:
            return "Cannot divide by Zero"    



##input by default returns strings
first_number = float(input('Please enter the first number: '))
Second_number = float(input('Please enter the second number: '))
operation=input("Please enter the operation: ")
while operation not in ['+','-','*','/']:
    operation=input("Please enter a valid operation")
    results = calculate(first_number, Second_number, operation)
print(results)