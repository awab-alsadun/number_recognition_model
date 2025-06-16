the main code creat and train CNN model build based on tesorflow framework and use cv2 libary to make a small window you draw in it to make the model predict your drawing 
after doing adjustment like padding,gauss-blur ect...,some and by pressing in your keyboard you control the actions that happen in the window from clear 
the window if you mess up(C) or to predict(P) or to quit the program(Q)

everytime you press (P) the model will show you the top 3 predictions and the precentage of the three of them to show the precssion in it choise  and there is a special function
if one of the top 3 is the number 4 due to problem in the model to predict the number 4, the function basicaly just make the Characteristics of the number 4 more clear

the save code is a code that save the valus of images and the labels of the EMNST data into numpy matrix to save time everytime you run,make sure to put your own path of the EMNST in the code
