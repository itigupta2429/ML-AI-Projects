'''
def factorial(num):
    if num == 1:
        return 1
    return(num * factorial(num-1))

number=4
print(factorial(number))
'''

## 1,2,3,5,8
def fibonacci(num):
    if num <= 1:
        return 1
    return fibonacci(num-1) + fibonacci(num-2)

number=5
print(fibonacci(number))
