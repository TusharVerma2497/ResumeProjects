
import numpy as np

class bucket:
    def __init__(self, pixels, pos):
        self.pixels_=pixels
        self.pos_=pos
        self.avgPixel=np.mean(pixels, axis=0).astype(np.int32)
        self.Range=np.ptp(pixels, axis=0)
        self.mostDynamicChannel=np.argmax(self.Range)
        # we are using heapq module to work with heaps 
        self.heapScore=self.Range[self.mostDynamicChannel]
        # self.heapScore=(self.Range[0]/255)*(self.Range[1]/255)*(self.Range[2]/255)
    
    def __len__(self):
        return len(self.pixels_)
    
    def __lt__(self, other):
        #to create a max-heap
        return self.heapScore > other.heapScore

    def splitMedianOn_mostDynamicChannel(self):
        #sorting the pixels based on mostDynamicChannel
        sorting_indices = np.argsort(self.pixels_[:, self.mostDynamicChannel])
        self.pixels_=self.pixels_[sorting_indices]
        self.pos_=self.pos_[sorting_indices]

        #median of a sorted array is the middle value
        mid=int(np.floor(self.pixels_.shape[0]/2))

        #creating left and right bucket
        bucketLeft=bucket(pixels=self.pixels_[:mid,:], pos=self.pos_[:mid,:])
        bucketRight=bucket(pixels=self.pixels_[mid:,:], pos=self.pos_[mid:,:])

        return bucketLeft, bucketRight