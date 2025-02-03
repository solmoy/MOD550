'''---------------------------------------------------------------------
Program that plots  thee predicted model and the imported data
---------------------------------------------------------------------'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data
data = pd.read_csv('/Users/austbo/Desktop/MOD550/data/data_MH.csv')

#plot with imported data
plt.subplot(1,2,1)
plt.scatter(data['x'], data['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Imported data')

#Plot with predicted model
plt.subplot(1,2,2)
plt.scatter(data['x'], data['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predicted model')
plt.axvline(x=0, color='r')
plt.legend()

plt.show()
plt.tight_layout()

'''---------------------------------------------------------------------
Program that writes an assesment of 3 different Github repositories
---------------------------------------------------------------------'''
#Save string a s text file

text = ''' WOTD
Users: solmoy and pjheden Repo: WOTD
- Assesment of coding standards: This is my own Repo for an app that I am developing with a friend. Since neither of us are app developers, it seemed very useful to go through and assess our coding practice through what we learned in class, knowing that in the begining I really struggled with what to do in GitHub. Since this is a private code, we don´t necessarily need to use a “Golden Standard”, but it is always useful to check your own code.
    - Readability and clarity:
        - Well structured, and easy to read. But we should have had more inline comments to make the logic of the program seem more obvious.
    - Structure and object oriented:
        - Because the code is created in Swift , it follows SwiftUI practices using @EnvironmentObject for state management. 
    - Consistency and style:
        - The code follows Swift´s standard naming convention.
    - Documentation:
        - The code really lacks documentation everywhere unfortunately. It would really benefit from adding comments explaining the purpose of the different UI components. This will be done ASAP.
    - Maintainability:
        - The code is generally maintainable and “easy” fix and understand as it is a very simple app.
    - Testing:
        - Testing is being done using TestFlight. and version 1.0 is being used by beta testers.
    - Bug fixing:
        - 1 bug has been found by beta testers so far, this have not been resolved yet. But there is an issue created in the repo waiting to be fixed as soon as we have time.

Introducing-python

User: mad scheme  Repo: introducing-python
 In the README file its says this: 
This repository contains the programs featured in the second edition of the book Introducing Python.
This is good information to have, and seems like a very useful repository to use when learning python.

-Readability and clarity: 
The code is concise and easy to read, with clear variable names. I did not check all the programs( As there are way too many) , but went through some random samples to get a good idea.

- Structure and object oriented: 
 Step by step the programs gets more object oriented, and you learn how to create smarter code. 
- Consistency and style: 
The code follows standard Python naming conventions and indentation.

- Documentation: 
 There is not a lot of documentation, and if you don’t know what all the chapters are for, this could be confusing. Would be improved with more comments and docstrings.
- Maintainability:
Not at all sure how this code would be maintained if for example a new edition of the book came out. 

TensorFlow-Examples
User: aymericdamien Repo:TensorFlow-Examples
This Repo has a great README file.  Providing information that makes it way easier to navigate in the repository,. 

- Readability and clarity: 
The code is clear and well structured in Jupyter notebooks. It  also provides  explanations and some background in he beginning of the notebooks. 
- Structure and object oriented: 
 Since it follows certain procedures for teaching how to use Tensor FLow, the notebooks does not always follow modularisations.
- Consistency and style: 
The code follows python conventions.
- Documentation: 
There is good documentations in the beginning and a lot of comments in the code which provides good information to the user.
- Maintainability:
 Here I am also not sure how maintanable the set up is. And it is worth noting that the over 250 commits to the repo has been done at least 5 years ago. So it is quite old. '''

#saves the text to a file
with open('assesment.txt', 'w') as f:
    f.write(text)
