class Car:
    country_of_origin="USA"
    def __init__(self,make, model, color):
        self.make=make
        self.model=model
        self.color=color
    def description(self):
        print(f"The make {self.make} , model {self.model}, color {self.color}")

my_car1=Car("Hyundai", "Sonata","Black")
my_car1.description()

#del my_car1

class ElectricCar(Car):
    def __init__(self, make, model, color, battery_capacity):
        super().__init__(make,model, color)
        self.battery_capacity=battery_capacity
    def display_battery(self):
        print(f"This electric car has a battery capacity {self.battery_capacity} in {Car.country_of_origin}") ## here country_of_origin is the global variable

electric_car1=ElectricCar("Tesla", "Model S", "Purple", 100)
electric_car1.description()
electric_car1.display_battery()


class Animal:
    def __init__(self,name):
        self.name=name
    def speek(self):
        pass

class Dog(Animal):
    def speak(self):
        return f"{self.name} barks"
    
class Cat(Animal):
    def speak(self):
        return f"{self.name} Meows"

dog=Dog("Tommy")
cat=Cat("Kitty")
print(dog.speak(), cat.speak())