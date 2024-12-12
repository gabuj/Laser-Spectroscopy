import numpy as np

def chop(t,i,xmin,xmax):
    #chop data from x value to xvalue
    x= t[(t>xmin) & (t<xmax)]
    y= i[(t>xmin) & (t<xmax)]
    return x, y