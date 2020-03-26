import numpy as np
import cv2
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

class ASSOCIATOR:
    def __init__(
        self,
        active_tracks):

        self.active_tracks = active_tracks
        print('-'*30,'Associator in action','-'*30)
        
        self.MATURITY_THRESHOLD = 5
        self.MULTIPLE_LUGGAGE_ASSOCIATION = True
        self.MAX_LUGGAGE_COUNT = 2
        self.CUSTOM_COST = True

        self.filter_tracks()
        self.extract_luggage_person_tracks()
        self.extract_luggage_person_centres()
        self.get_distance()


    def filter_tracks(self):
        self.tracks = [track for track in self.active_tracks if track.maturity >= self.MATURITY_THRESHOLD]


    def extract_luggage_person_tracks(self):
        self.luggages = [track for track in self.active_tracks if track.label == 1]
        if self.MULTIPLE_LUGGAGE_ASSOCIATION == True: 
            self.people = [track for track in self.active_tracks if track.label == 0]*self.MAX_LUGGAGE_COUNT


    def box_to_centre(self, box):
        """box = [xmin, ymin, xmax, ymax]"""
        cX = int((box[0] + box[2])*0.5)
        cY = int((box[1] + box[3])*0.5)
        return (cX,cY)


    def extract_luggage_person_centres(self):
        people_centres = [self.box_to_centre(person.box) for person in self.people]
        luggage_centres = [self.box_to_centre(luggage.box) for luggage in self.luggages]

        self.people_centres = np.asarray(people_centres)
        self.luggage_centres = np.asarray(luggage_centres)

        # print('people centres and shape:')
        # print(self.people_centres, self.people_centres.shape)
        # print('luggage centres and shape:')
        # print(self.luggage_centres, self.luggage_centres.shape)
        

    def get_distance(self):
        if len(self.people_centres)>0 and len(self.luggage_centres)>0:
            Distance = distance.cdist(self.luggage_centres, self.people_centres, 'euclidean')

            # print('distance:',Distance)
            # print('distance.shape:',Distance.shape)

            if self.CUSTOM_COST == True:
                person_box_width = np.asarray([track.box[2]-track.box[0] for track in self.people])
                # print('$'*30, person_box_width)
                Distance = Distance/person_box_width
                # print('d/w:', Distance)
            self.distance = Distance
        else: self.distance = None


    def associate(self):
        if self.distance is None: self.associations = None

        else: 
            row_ind, col_ind = linear_sum_assignment(self.distance)
            self.associations = zip(row_ind, col_ind)
            # print('associations:',row_ind, col_ind )
            # print('*'*10, self.people)
            box = [np.tile(self.people,(self.MAX_LUGGAGE_COUNT,1))[i][1] for i in col_ind]
            # print(box)
            # print('Width:', box[0][2]-box[0][0])
        return self.associations


    def detect_abandonment(self):
        pass













#active_tracks = [Track(id='107', box=array([283.43491   ,  78.36739064, 326.35808647, 222.41480346]), label=0, maturity=171), Track(id='110', box=array([ 64.09182233, 241.33099177, 150.17510196, 405.24765011]), label=1, maturity=77)]
