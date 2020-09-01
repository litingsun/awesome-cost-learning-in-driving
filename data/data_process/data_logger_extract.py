# -^- coding:utf-8 -^-
import pandas as pd
from pandas import read_csv
import pickle as pkl
import os

CHECK = None # changed from boolean flag into a function object lambda df: bool 2019-09-09 17:11:23
RECORD_SOURCE = False
CHECKLENTH = False
SKIPEMPTY = True # added this since 2019-09-09 14:47:59

## the labels in csv
TRACK_ID = 'track_id'
FRAME_ID = 'frame_id'

def extract(csvFile, trackID, startFrame=-1, endFrame=1e10):
    NeedConcate = True
    if(not isinstance(trackID, tuple)): # the track needs to be concatenated
        trid = (trackID,)
        NeedConcate = False
    else:
        trid = trackID
    trackdfs = []

    df = read_csv(csvFile)
    for i in trid:
        trackdfs.append(df.loc[ df[TRACK_ID] == int(i)])
    

    trackdf = pd.concat(trackdfs) if NeedConcate else trackdfs[0]
    lentotal = len(trackdf)

    trackdf = trackdf.loc[ trackdf[FRAME_ID] <= endFrame]
    trackdf = trackdf.loc[ startFrame <= trackdf[FRAME_ID] ]
    lenselect = len(trackdf)
    if(CHECKLENTH):
        print(csvFile, trackID, "lenselect:",lenselect, "lentotal:", lentotal)
    if(CHECKLENTH and lenselect/lentotal < 0.8):
        print("Warning: check lenth failed" ,csvFile, trackID, "lenselect:",lenselect, "lentotal:", lentotal )
    return trackdf

def dump(csvFile, trackID, trackdf):
    outname = "{}_{}.pkl".format(csvFile.split("/")[-1].split(".")[-2] , str(trackID) )
    trackdf.to_pickle(outname)
    return outname

######### THE CHECK FUNCTION FOR SR ##########
# def check(df):
#     # check the track has been moved from one side to the other
#     # print(len(df))
#     if(not len(df)):
#         return False
#     sides = [df.iloc[0]['x'],df.iloc[-1]['x']]
#     if(min(*sides) > 926 or max(*sides) < 1050):
#         print(sides)
#         return False
#     return True
###############################################

def dump_trajs(filename,file_cars,video_csv,appendix):
    """
    collect a list, each item is a panda dataframe of the trajectory of one car
    flie_cars: Dict,  file_name of the video : list of track_id (if the item is tuple, concatenate them)
    video_csv, Dict,  file_name of the video : corresponding csv name
    appendix: the appendix of the path of the csv files
    """
    trajs = []
    traj_sources = []
    for scene, cars  in file_cars.items():
        if(scene not in video_csv.keys()):
            print("CSV file not found for %s",scene)
            continue 
        fn = video_csv[scene]
        fn = os.path.join(appendix, fn)
        for c in cars:
            if(type(c)==tuple and len(c)>2): # c is the tuple of arguments , if c =2, we will concate the tracks
                df = extract(fn,*c)
            else:
                df = extract(fn, c)
            print("ind:",len(trajs),end = " ")
            print(fn, c)
            if(not len(df)):
                print("WARNING: no length:", fn,c)
                if(SKIPEMPTY):
                    continue
            if(CHECK is None or CHECK(df)):
                trajs.append(df)
                traj_sources.append((fn,scene, c))
            else:
                print("CKECK FAILED:", fn, c)

    with open(filename,"wb") as f:
        if(RECORD_SOURCE):
            pkl.dump((trajs,traj_sources),f)
        else:
            pkl.dump(trajs,f)

if __name__ == "__main__":
    import sys
    print(sys.argv)
    if(len(sys.argv) != 3):
        print("usage: python data_logger_extract.py [csv file name] [target track ID]")
        exit()
    df = extract(sys.argv[1],sys.argv[2])
    name = dump(sys.argv[1],sys.argv[2],df)
    ## Load again to check ##
    loaded_df = pd.read_pickle(name)
    print(loaded_df)