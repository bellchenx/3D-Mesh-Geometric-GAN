# fetcher.py: Data fetcher of ShapeNet

import numpy as np
import sys
import _pickle as pickle
import threading
import queue
import time


class DataFetcher(threading.Thread):
    def __init__(self, file_list):
        super(DataFetcher, self).__init__()
        self.stopped = False
        self.queue = queue.Queue(64)

        self.pkl_list = []
        with open(file_list, 'r') as f:
            while(True):
                line = f.readline().strip()
                if not line:
                    break
                self.pkl_list.append(line)
        self.index = 0
        self.number = len(self.pkl_list)
        np.random.shuffle(self.pkl_list)

    def work(self, idx):
        pkl_path = self.pkl_list[idx]
        pkl = pickle.load(open(pkl_path, 'rb'), encoding='latin1')
        img = pkl[0].astype('float32')/255.0
        label = pkl[1]

        return img, label, pkl_path.split('/')[-1]

    def run(self):
        while self.index < 900000000 and not self.stopped:
            self.queue.put(self.work(self.index % self.number))
            self.index += 1
            if self.index % self.number == 0:
                np.random.shuffle(self.pkl_list)

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()


# DEBUG

from numpy import random
import pickle
if __name__ == '__main__':
    file_list = sys.argv[1]
    dataset = DataFetcher(file_list)
    dataset.start()
    count = 0
    for i in range(0, 2):
        _, point, _ = dataset.fetch()
        if point.shape[0] < 51*51:
            continue
        if point.shape[0] > 51*51:
            index = point.shape[0]
            while index > 51*51:
                rand_int = random.randint(0, index - 2)
                point[rand_int, :] = point[index - 1, :]
                index = index - 1
            point = point[0:51*51][:]
        
        point = point[:, 0:3]
        # pickle.dump([point], open("test.pkl", "wb"))
    dataset.shutdown()