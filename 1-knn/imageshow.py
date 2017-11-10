#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import numpy
import knn

def image2vertor(dirname):
    dataset=None
    titles=numpy.array([])

    for vfile in os.listdir(dirname):
        f=open(name=dirname+"/" + vfile, mode="r")

        v_line=numpy.array([])

        for v_char in f.read():
            if not v_char.isdigit():
                continue

            v_line = numpy.hstack((v_line, int(v_char)))

        if dataset is None:
            dataset = v_line.reshape(1024,1)
        else:
            dataset=numpy.hstack((dataset, v_line.reshape(1024,1)))

        tag= int(vfile.split("_")[0])
        titles=numpy.hstack((titles, tag))
    return dataset,titles


def main():
    metaset,metatag = image2vertor("trainingDigits")
    testset,testtag=image2vertor("testDigits")


    print(len(metaset[:,1]))
    print(len(testset[:,1]))
    print(len(metatag))
    print(len(testtag))


    errorCount=0
    for i in range(len(testtag)):
         res =knn.knnGetTitle(inX=testset[:,i].reshape(1024,1), dataset=metaset, titles=metatag,k=3)

         print("%d test result is %s[%s]" % (i, res, testtag[i]))

         if res != testtag[i]:
             errorCount=errorCount+1

    print("error count is %d of %f" % (errorCount, errorCount/float(len(testtag))))

if __name__ == '__main__':
    main()
