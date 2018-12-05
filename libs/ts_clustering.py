# source: https://github.com/alexminnaar/time-series-classification-and-clustering
import random

import matplotlib.pylab as plt
import numpy as np


class ts_cluster(object):
    paper_src_gen_dir = '/Volumes/Kings500/dev/myWiki/uni/04_statistics_seminar/src-gen/'
    paper_src_gen_figures_dir = paper_src_gen_dir + 'figs/'

    def __init__(self, data, num_clust):
        '''
        num_clust is the number of clusters for the k-means algorithm
        assignments holds the assignments of data points (indices) to clusters
        centroids holds the centroids of the clusters
        '''
        self.num_clust = num_clust
        self.assignments = {}
        self.centroids = []
        self.data = data

    def k_means_clust(self, num_iter, w, progress=False):
        '''
        k-means clustering algorithm for time series data.  dynamic time warping Euclidean distance
         used as default similarity measure.
        '''
        self.centroids = random.sample(list(self.data), self.num_clust)

        for n in range(num_iter):
            if progress:
                print('iteration ' + str(n + 1))
            # assign data points to clusters
            self.assignments = {}
            for ind, i in enumerate(self.data):
                min_dist = float('inf')
                closest_clust = None
                for c_ind, j in enumerate(self.centroids):
                    if self.LB_Keogh(i, j, 5) < min_dist:
                        cur_dist = self.DTWDistance(i, j, w)
                        if cur_dist < min_dist:
                            min_dist = cur_dist
                            closest_clust = c_ind
                if closest_clust in self.assignments:
                    self.assignments[closest_clust].append(ind)
                else:
                    self.assignments[closest_clust] = [ind]

            # recalculate centroids of clusters
            for key in self.assignments:
                clust_sum = 0
                for k in self.assignments[key]:
                    clust_sum = clust_sum + self.data[k]
                self.centroids[key] = [m / len(self.assignments[key]) for m in clust_sum]

    def get_centroids(self):
        return self.centroids

    def get_assignments(self):
        return self.assignments

    def plot_cluster_assigments(self):
        fig, ax = plt.subplots()
        ax.set_ylabel('value', fontsize=10)
        ax.set_xlabel('time', fontsize=10)
        index = 0
        for time_series in self.data:
            color = self.__get_color_for_cluster(index)
            plt.plot(range(1, len(time_series)+1), list(time_series), color=color)
            index += 1
        plt.title('Cluster assigment')
        plt.savefig('{}08_k={}_cluster_assigment.png'.format(self.paper_src_gen_figures_dir, self.num_clust), bbox_inches="tight")

    def __get_color_for_cluster(self, index):
        cluster_assignments = self.get_assignments()
        if (index in cluster_assignments[0]):
            return 'red'
        elif (index in cluster_assignments[1]):
            return 'blue'
        elif (index in cluster_assignments[2]):
            return 'black'
        elif (index in cluster_assignments[3]):
            return 'brown'
        elif (index in cluster_assignments[4]):
            return 'green'
        elif (index in cluster_assignments[5]):
            return 'darkblue'
        elif (index in cluster_assignments[6]):
            return 'orange'
        elif (index in cluster_assignments[7]):
            return 'darkgreen'
        elif (index in cluster_assignments[8]):
            return 'darkred'
        elif (index in cluster_assignments[9]):
            return 'darkgray'

    def plot_centroids(self):
        ax, fig = plt.subplots()
        for i in self.centroids:
            plt.plot(i)
        plt.savefig('{}08_k={}_cluster_centroids.png'.format(self.paper_src_gen_figures_dir, self.num_clust))

    def DTWDistance(self, s1, s2, w=None):
        '''
        Calculates dynamic time warping Euclidean distance between two
        sequences. Option to enforce locality constraint for window w.
        '''
        DTW = {}

        if w:
            w = max(w, abs(len(s1) - len(s2)))

            for i in range(-1, len(s1)):
                for j in range(-1, len(s2)):
                    DTW[(i, j)] = float('inf')

        else:
            for i in range(len(s1)):
                DTW[(i, -1)] = float('inf')
            for i in range(len(s2)):
                DTW[(-1, i)] = float('inf')

        DTW[(-1, -1)] = 0

        for i in range(len(s1)):
            if w:
                for j in range(max(0, i - w), min(len(s2), i + w)):
                    dist = (s1[i] - s2[j]) ** 2
                    DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
            else:
                for j in range(len(s2)):
                    dist = (s1[i] - s2[j]) ** 2
                    DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

        return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])

    def LB_Keogh(self, s1, s2, r):
        '''
        Calculates LB_Keough lower bound to dynamic time warping. Linear
        complexity compared to quadratic complexity of dtw.
        '''
        LB_sum = 0
        for ind, i in enumerate(s1):

            lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
            upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

            if i > upper_bound:
                LB_sum = LB_sum + (i - upper_bound) ** 2
            elif i < lower_bound:
                LB_sum = LB_sum + (i - lower_bound) ** 2

        return np.sqrt(LB_sum)
