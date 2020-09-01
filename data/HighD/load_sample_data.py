import sys
#if sys.version_info.major != 3:
    #print("Error: You must use python3")
    #exit()

import os
import pickle
import csv

from data_management.read_csv import *

import main


def get_vehicle_dict(vehicle_track, i_frame_id, samples_needed_before, samples_needed_after):
    vehicle_dict = dict()
    # x and y of the center must be computed via x and y of the upper left corner + half the width/length
    x_list = [vehicle_track[BBOX][i][0] + vehicle_track[BBOX][i][2] / 2. for i in
              range(i_frame_id - samples_needed_before, i_frame_id + samples_needed_after)]
    y_list = [vehicle_track[BBOX][i][1] + vehicle_track[BBOX][i][3] / 2. for i in
              range(i_frame_id - samples_needed_before, i_frame_id + samples_needed_after)]
    vehicle_dict["pos"] = [[x_list[i], y_list[i]] for i in range(len(x_list))]

    vehicle_dict["vel"] = [[vehicle_track[X_VELOCITY][i], vehicle_track[Y_VELOCITY][i]] for i in
                               range(i_frame_id - samples_needed_before, i_frame_id + samples_needed_after)]
    vehicle_dict["acc"] = [[vehicle_track[X_ACCELERATION][i], vehicle_track[Y_ACCELERATION][i]] for i in
                               range(i_frame_id - samples_needed_before, i_frame_id + samples_needed_after)]
    vehicle_dict["lane_id"] = [vehicle_track[LANE_ID][i] for i in
                               range(i_frame_id - samples_needed_before, i_frame_id + samples_needed_after)]
    return vehicle_dict


def load_scenarios():

    scenarios = []

    created_arguments = main.create_args()

    data_path = "../data/"
    track_file_ending = "_tracks.csv"
    static_file_ending = "_tracksMeta.csv"
    meta_file_ending = "_recordingMeta.csv"
    pickle_file_ending = "_tracks.pkl"

    time_before = 5.
    time_after = 5.

    decision_file = "good_samples.csv"

    sample_identifier_list = None
    with open(decision_file, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        sample_identifier_list = [row for row in readCSV]

    sample_identifier_list = sample_identifier_list[1:]

    for i in range(len(sample_identifier_list)):
    # for i in [0]:
        print("Loading sample " + str(i))
        scenario_dict = dict()

        file_id = sample_identifier_list[i][0]
        file_id_str = str(file_id).zfill(2)
        track_id = int(sample_identifier_list[i][1])
        frame_id = int(sample_identifier_list[i][2])

        # read the information from HighD data
        pickle_file = data_path + file_id_str + pickle_file_ending
        created_arguments["pickle_path"] = pickle_file
        if os.path.exists(created_arguments["pickle_path"]):
            with open(created_arguments["pickle_path"], "rb") as fp:
                tracks = pickle.load(fp)
        else:
            track_file = data_path + file_id_str + track_file_ending
            created_arguments["input_path"] = track_file
            tracks = read_track_csv(created_arguments)
            with open(created_arguments["pickle_path"], "wb") as fp:
                pickle.dump(tracks, fp)

        static_file = data_path + file_id_str + static_file_ending
        created_arguments["input_static_path"] = static_file
        static_info = read_static_info(created_arguments)
        meta_file = data_path + file_id_str + meta_file_ending
        created_arguments["input_meta_path"] = meta_file
        meta_dictionary = read_meta_info(created_arguments)

        created_arguments["background_image"] = data_path + file_id_str + "_highway.jpg"

        time_step = 1 / float(meta_dictionary[FRAME_RATE])
        samples_needed_before = int(time_before / time_step)
        samples_needed_after = int(time_after / time_step)
        first_frame_needed = frame_id - samples_needed_before
        final_frame_needed = frame_id + samples_needed_after -1

        ego_vehicle_track = tracks[track_id - 1]
        assert (ego_vehicle_track[TRACK_ID] == track_id)

        i_frame_id = frame_id - static_info[track_id][INITIAL_FRAME]

        scenario_dict["ego"] = get_vehicle_dict(ego_vehicle_track, i_frame_id, samples_needed_before, samples_needed_after)

        preceding_source_valid = True
        preceding_source_lane_id = ego_vehicle_track[PRECEDING_ID][i_frame_id - samples_needed_before]
        if preceding_source_lane_id == 0:
            preceding_source_valid = False
        elif static_info[preceding_source_lane_id][INITIAL_FRAME] > first_frame_needed:
            preceding_source_valid = False
        elif static_info[preceding_source_lane_id][FINAL_FRAME] < final_frame_needed:
            preceding_source_valid = False

        following_source_valid = True
        following_source_lane_id = ego_vehicle_track[FOLLOWING_ID][i_frame_id - samples_needed_before]
        if following_source_lane_id == 0:
            following_source_valid = False
        elif static_info[following_source_lane_id][INITIAL_FRAME] > first_frame_needed:
            following_source_valid = False
        elif static_info[following_source_lane_id][FINAL_FRAME] < final_frame_needed:
            following_source_valid = False

        preceding_target_valid = True
        preceding_target_lane_id = ego_vehicle_track[PRECEDING_ID][i_frame_id + samples_needed_after]
        if preceding_target_lane_id == 0:
            preceding_target_valid = False
        elif static_info[preceding_target_lane_id][INITIAL_FRAME] > first_frame_needed:
            preceding_target_valid = False
        elif static_info[preceding_target_lane_id][FINAL_FRAME] < final_frame_needed:
            preceding_target_valid = False

        following_target_valid = True
        following_target_lane_id = ego_vehicle_track[FOLLOWING_ID][i_frame_id + samples_needed_after]
        if following_target_lane_id == 0:
            following_target_valid = False
        elif static_info[following_target_lane_id][INITIAL_FRAME] > first_frame_needed:
            following_target_valid = False
        elif static_info[following_target_lane_id][FINAL_FRAME] < final_frame_needed:
            following_target_valid = False

        if not preceding_source_valid or not following_source_valid or not preceding_target_valid or not following_target_valid:
            print("Sample invalid")
            continue

        for id in [preceding_source_lane_id, following_source_lane_id, preceding_target_lane_id, following_target_lane_id]:
            vehicle_track = tracks[id - 1]
            assert (vehicle_track[TRACK_ID] == id)
            i_frame_id = frame_id - static_info[id][INITIAL_FRAME]
            if id == preceding_source_lane_id:
                scenario_dict["preceding_source_lane"] = get_vehicle_dict(vehicle_track, i_frame_id, samples_needed_before,
                                                                          samples_needed_after)
            if id == following_source_lane_id:
                scenario_dict["following_source_lane"] = get_vehicle_dict(vehicle_track, i_frame_id, samples_needed_before,
                                                                          samples_needed_after)
            if id == preceding_target_lane_id:
                scenario_dict["preceding_target_lane"] = get_vehicle_dict(vehicle_track, i_frame_id, samples_needed_before,
                                                                          samples_needed_after)
            if id == following_target_lane_id:
                scenario_dict["following_target_lane"] = get_vehicle_dict(vehicle_track, i_frame_id, samples_needed_before,
                                                                          samples_needed_after)

        scenarios.append(scenario_dict)

    return scenarios

if __name__ == '__main__':
    scenarios = load_scenarios()
    print("Loaded " + str(len(scenarios)) + " lane change(s)")

    # accessing data:
    print("\n First scenario:")
    scenario = scenarios[0]
    print("  First position of ego car is " + str(scenario["ego"]["pos"][0]))
    print("  First X position of ego car is " + str(scenario["ego"]["pos"][0][0]))
    print("  First Lane id of ego car is " + str(scenario["ego"]["lane_id"][0]))
    print("  First Lane id of following car in target lane is " + str(scenario["following_target_lane"]["lane_id"][0]))

