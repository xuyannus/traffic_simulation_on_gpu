#########################################################################
This project tests mesoscopic traffic simulation on CPU/GPU.
#########################################################################
The target platform is:
1) Mesoscopic Traffic Models
2) Singapore Expressway Network
3) Simulation from 7:00AM to 8:00AM in a random day
4) Hardware: a CPU Core and a GPU

#########################################################################
About the Code 
#########################################################################
1) The code is following the Google Code Style @ 
http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml
2) In the research demo project, attributes in class are public accessable,
   for saving time. It needs to be modified in next refactoring. 
   
#########################################################################
memo history
#########################################################################
2014-4-10
The framework has been tested using a Grid Network.
Now, we want to test the framework using Singapore Network.
Also, we want to refactor the code to make it more readable.
-------------------------------