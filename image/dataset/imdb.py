import os
import pandas as pd
import numpy as np
import sklearn.model_selection

from datetime import datetime
from scipy.io import loadmat


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


class IMDB(object):
    """Class for loading IMDB gender and age classification dataset."""
    def __init__(self, datasetPath="/home/cv/Disk/data/face/imdb_crop", imageSize=(48, 48)):
        self.imageSize = imageSize
        self.datasetPath = datasetPath
        self.labelFile = os.path.join(datasetPath, 'imdb.mat') 
        
    def get_data(self):
        return self._load_imdb()

    def _load_imdb(self):
        face_score_treshold = 3
        dataset = loadmat(self.labelFile)

        full_paths = dataset['imdb']['full_path'][0, 0][0]
        genders = dataset['imdb']['gender'][0, 0][0]
        dob = dataset['imdb'][0, 0]["dob"][0] 
        photo_taken = dataset['imdb'][0, 0]["photo_taken"][0]
        ages = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]
        face_scores = dataset['imdb']['face_score'][0, 0][0]
        second_face_scores = dataset['imdb']['second_face_score'][0, 0][0]

        face_score_mask = face_scores > face_score_treshold
        second_face_score_mask = np.isnan(second_face_scores)
        unknown_gender_mask = np.logical_not(np.isnan(genders))
        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)

        age = np.array(ages)[mask]
        gender = genders[mask]
        filename = []
        full_paths = full_paths[mask]
        for index in range(full_paths.shape[0]):
            name = full_paths[index][0]
            filename.append(name)

        data = {"filename": filename, "gender": gender, "age": age}
        ground_truth_data = pd.DataFrame(data)

        return ground_truth_data
    

def split_imdb_data(dataset, val_ratio):
    train_sets, val_sets = \
        sklearn.model_selection.train_test_split(dataset, test_size=val_ratio, random_state=2018)
    train_sets.reset_index(drop=True, inplace=True)
    val_sets.reset_index(drop=True, inplace=True)
    # train_nums = train_sets.shape[0]
    # val_nums = val_sets.shape[0]
    # return train_sets, train_nums, val_sets, val_nums
    return train_sets, val_sets
