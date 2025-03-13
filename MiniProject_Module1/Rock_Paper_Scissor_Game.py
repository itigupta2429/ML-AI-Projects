##Import random library
import random

##Function that takes input from the User and prints error message if the selected option is incorrect
def take_user_input():
    User_num=int(input("Choose a number from 1,2,or 3; 1=Rock; 2=Paper; 3=Scissor: "))

    if(User_num==1 or User_num==2 or User_num==3):
        return(User_num)
    else:
        print("ERROR: Please choose the correct number from the options provided! ")

##Main function for the Game. Calls the take_user_input function from inside the function and returns the result of the game 
def rock_paper_scissor():
    User_play=take_user_input()
    print(f"You Entered: {User_play}")
    Computer_play=random.randint(1,3)
    print(f"Computer chose: {Computer_play}")
    if(Computer_play == User_play):
        return("It's a draw. Play Again!")
    elif(Computer_play==1 and User_play==2):
        return("Whoo! You Won.")
    elif(Computer_play==1 and User_play==3):
        return("Oops! You Lost. Better Luck Next Time")
    elif(Computer_play==2 and User_play==1):
        return("Oops! You Lost. Better Luck Next Time") 
    elif(Computer_play==2 and User_play==3):
        return("Whoo! You Won.")
    elif(Computer_play==3 and User_play==1):
        return("Whoo! You Won.") 
    elif(Computer_play==3 and User_play==2):
        return("Oops! You Lost. Better Luck Next Time") 

##Printing the results of the game
print(rock_paper_scissor())