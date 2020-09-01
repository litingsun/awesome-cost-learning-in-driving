import sys
if sys.version_info.major != 3:
    print("Error: You must use python3")
    exit()

import os
import pickle as pkl
import csv


from data_management.read_csv import *
from visualization.visualize_frame import VisualizationPlot

import main

TRAVEL_DIR_LEFT_TO_RIGHT = 2
TRAVEL_DIR_RIGHT_TO_LEFT = 1
LC_LEFT = "LC_LEFT"
LC_RIGHT = "LC_RIGHT"
VISUALIZATION = False

if __name__ == '__main__':

    created_arguments = main.create_args()

    data_path = "../../../HighD/data/"
    track_file_ending = "_tracks.csv"
    static_file_ending = "_tracksMeta.csv"
    meta_file_ending = "_recordingMeta.csv"
    pkl_file_ending = "_tracks.pkl"

    decision_file = "good_samples912.csv"
    with open(decision_file, 'w') as csvfile: # init a new csvfile
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["file_id", "track_id", "frame_id","preceding_source_lane", "following_source_lane", "preceding_target_lane", "following_target_lane"])

    num_lc = 0
    num_lc_too_early = 0
    num_lc_too_late = 0
    num_lc_full_info = 0

    time_before = 5.
    time_after = 5.
    time_step = 1/25.

    for id in range(1, 61):
        id_str = str(id).zfill(2)
        print("analyzing track " + id_str)

        pkl_file = data_path + id_str + pkl_file_ending
        created_arguments["pkl_path"] = pkl_file
        if os.path.exists(created_arguments["pkl_path"]):
            with open(created_arguments["pkl_path"], "rb") as fp:
                tracks = pkl.load(fp)
        else:
            track_file = data_path + id_str + track_file_ending
            created_arguments["input_path"] = track_file
            tracks = read_track_csv(created_arguments)
            with open(created_arguments["pkl_path"], "wb") as fp:
                pkl.dump(tracks, fp)

        try:
            static_file = data_path + id_str + static_file_ending
            created_arguments["input_static_path"] = static_file
            static_info = read_static_info(created_arguments)
            meta_file = data_path + id_str + meta_file_ending
            created_arguments["input_meta_path"] = meta_file
            meta_dictionary = read_meta_info(created_arguments)
        except:
            print("id",id_str,"open meta failed")
            continue
        created_arguments["background_image"] = data_path + id_str + "_highway.jpg"

        time_step = 1/float(meta_dictionary[FRAME_RATE])
        samples_needed_before = int(time_before / time_step)
        samples_needed_after = int(time_after / time_step)

        for track in tracks:
            static_track_info = static_info[track[TRACK_ID]]
            if static_track_info[NUMBER_LANE_CHANGES] > 0:  # did lane change
                lc_frame = []
                initial_frame = static_track_info[INITIAL_FRAME]
                final_frame = static_track_info[FINAL_FRAME]
                lane_id = track[LANE_ID][0]
                for i in range(len(track[LANE_ID])):
                    if track[LANE_ID][i] != lane_id:
                        LC_TYPE = None
                        if track[LANE_ID][i] > lane_id and static_track_info[DRIVING_DIRECTION] == TRAVEL_DIR_LEFT_TO_RIGHT:
                            assert LC_TYPE is None
                            LC_TYPE = LC_RIGHT
                        if track[LANE_ID][i] < lane_id and static_track_info[DRIVING_DIRECTION] == TRAVEL_DIR_LEFT_TO_RIGHT:
                            assert LC_TYPE is None
                            LC_TYPE = LC_LEFT
                        if track[LANE_ID][i] > lane_id and static_track_info[DRIVING_DIRECTION] == TRAVEL_DIR_RIGHT_TO_LEFT:
                            assert LC_TYPE is None
                            LC_TYPE = LC_LEFT
                        if track[LANE_ID][i] < lane_id and static_track_info[DRIVING_DIRECTION] == TRAVEL_DIR_RIGHT_TO_LEFT:
                            assert LC_TYPE is None
                            LC_TYPE = LC_RIGHT
                        lane_id = track[LANE_ID][i]

                        num_lc += 1
                        if i <= samples_needed_before:
                            num_lc_too_early += 1
                        elif i >= len(track[LANE_ID]) - samples_needed_after:
                            num_lc_too_late += 1
                        else:
                            preceding_source_lane = track[PRECEDING_ID][i-samples_needed_before]
                            following_source_lane = track[FOLLOWING_ID][i-samples_needed_before]
                            preceding_target_lane = track[PRECEDING_ID][i+samples_needed_after]
                            following_target_lane = track[FOLLOWING_ID][i+samples_needed_after]

                            first_frame_needed = track[FRAME][i] - samples_needed_before
                            final_frame_needed = track[FRAME][i] + samples_needed_after - 1

                            surrounding_vehicles_valid = True
                            # check if we see those vehicles throughout the whole lane change time
                            for vehicle_id in [preceding_source_lane, following_source_lane, preceding_target_lane, following_target_lane]:
                                if vehicle_id == 0:
                                    surrounding_vehicles_valid = False
                                elif static_info[vehicle_id][INITIAL_FRAME] > first_frame_needed:
                                    surrounding_vehicles_valid = False
                                elif static_info[vehicle_id][FINAL_FRAME] < final_frame_needed:
                                    surrounding_vehicles_valid = False

                            if surrounding_vehicles_valid \
                                    and static_info[preceding_source_lane][NUMBER_LANE_CHANGES] == 0 \
                                    and static_info[following_source_lane][NUMBER_LANE_CHANGES] == 0 \
                                    and static_info[preceding_target_lane][NUMBER_LANE_CHANGES] == 0 \
                                    and static_info[following_target_lane][NUMBER_LANE_CHANGES] == 0:
                                surrounding_objects_list = [preceding_source_lane, following_source_lane, preceding_target_lane, following_target_lane]
                                num_lc_full_info += 1
                                print("file_id=" + id_str + ", track_id=" + str(track[TRACK_ID]) + ", frame_id="+str(track[FRAME][i]) + " " + LC_TYPE)
                                
                                with open(decision_file, 'a') as csvfile:
                                    csvwriter = csv.writer(csvfile)
                                    csvwriter.writerow([int(id_str), track[TRACK_ID], track[FRAME][i],*surrounding_objects_list])
                                if VISUALIZATION:
                                    visualization_plot = VisualizationPlot(created_arguments, tracks, static_info,
                                                                           meta_dictionary, current_frame=track[FRAME][i], relevant_object=track[TRACK_ID], surrounding_objects=surrounding_objects_list)
                                    visualization_plot.show()

                                    if visualization_plot.decision == "Take":
                                        print("take")
                                        with open(decision_file, 'a') as csvfile:
                                            csvwriter = csv.writer(csvfile)
                                            csvwriter.writerow([int(id_str), track[TRACK_ID], track[FRAME][i]],*surrounding_objects_list)
                                    elif visualization_plot.decision == "Reject":
                                        print("reject")
                                    elif visualization_plot.decision == "Close":
                                        exit()
                                    else:
                                        raise RuntimeError("Unexpected")


        print(" currently found " + str(num_lc) + " lane changes, " +str(num_lc_full_info) + " with full info, " + str(num_lc_too_early) + " too early and " +str(num_lc_too_late) + " too late")