# localization

This is file contains the algorithm I used to localize each connected component I am interested in
and I used them to seperate each component out from the image to feed them to the classifier seperately
to get the prediction on them.

Here (w,x,y,z) coordinates required to extract the local boundary for the components are first initialized with values
zeros and colum/row pixel numbers and when the algorithm completes it run finally we get the (w,x,y,z) value we require.
