def Sort(num_list):
    for i in range(len(num_list)):
        for j in range(len(num_list)):
            if (num_list[i]<num_list[j]):
                a=num_list[i]
                num_list[i]=num_list[j]
                num_list[j]=a

    return(num_list)

my_list=[3,2,5,1,9,6,1.3,1,-100]
print(Sort(my_list))
