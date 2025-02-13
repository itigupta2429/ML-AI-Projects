#print(number_list[0:len(number_list)])
#print(number_list.index(46))
#print(len(number_list))
#for i in range(0, len(number_list),1):
 #   print(number_list[i])

#for number in number_list:
 #  print(number)


#for i in range(1, 4,1):
 # for j in range(1,i+1,1):
  #  print(j)

number_list =[1,3,46,6,8,-65,-76487,387,45,-9,28734876]

def Average(num_list):
  sum=0
  for i in num_list:
    sum=sum+i
  average=sum/len(num_list)
  return average

print(Average(number_list))

def Maximum(num_list):
  max=0
  for number in num_list:
    if max < number:
      max = number
  return max
  
print(Maximum(number_list))
