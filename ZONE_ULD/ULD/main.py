import time
import numpy as np
import cv2
import motpy
from motpy import MultiObjectTracker, ModelPreset, Detection
from motpy.testing_viz import draw_rectangle, draw_text, draw_box_centre, draw_line
from YOLOv2.Class_Yolo_V2 import YOLOv2 
from YOLOv2.utils import Build_VideoReader, bboxes_to_detections
from ASSOCIATOR.associate import ASSOCIATOR 
from scipy.spatial import distance

def demo_tracking_visualization():
    dt = 1 / 24
    #     active_tracks_kwargs={'min_steps_alive': 2, 'max_staleness': 6},
    #     tracker_kwargs={'max_staleness': 12}

    tracker = MultiObjectTracker(
        dt=dt,
        model_spec=ModelPreset.constant_acceleration_and_static_box_size_2d.value,
        active_tracks_kwargs={'min_steps_alive': -1, 'max_staleness': 6},
        tracker_kwargs={'max_staleness': 20})

    video_inp = '/home/ZONE24X7-CMB/teharaf/Documents/Abandoned_Object_Detection/Data/ABODA-master/AVSS_E2.avi'#AVSS_E2.avi video4
    video_out = 'out/test.avi'

    video_reader,frame_h,frame_w,fps = Build_VideoReader(video_inp)
    video_writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_w, frame_h)) #MP4V
    
    while True:
    	tic = time.time()

    	ret,img = video_reader.read()
    	if ret == False:
    		break
    	boxes = yolo.predict(img)
    	detections = bboxes_to_detections(boxes,frame_h,frame_w)
    	detections = [d for d in detections if d.box is not None]
    	active_tracks = tracker.step(detections=detections)

    	if len(active_tracks) >=2 :
	    	associator = ASSOCIATOR(active_tracks)
	    	associations = associator.associate()

	    	if associations is None: pass
	    	else:
		    	for i,j in associations:
		    		print(i, j)
			    	start_point = tuple(associator.luggage_centres[i])
			    	end_point   = tuple(associator.people_centres[j])
			    	print(start_point, end_point)
			    	draw_line(img, start_point, end_point)
			    	print('Euclidean distance:',distance.cdist(np.asarray([start_point]), np.asarray([end_point]), 'euclidean'))

    	toc = time.time()
    	# print('time_elapsed to process one frame:{} s'.format(toc-tic))

    	if len(active_tracks)>0:
    		print('-'*30)
    		print(len(active_tracks))
    		print(active_tracks)

    	for track in active_tracks:
    		if track.label == 0: img = draw_rectangle(img, track.box, color=(10, 10, 220), thickness=5)
    		if track.label == 1: img = draw_rectangle(img, track.box, color=(220, 10, 10), thickness=5)
    		if track.maturity <= 3 and track.maturity > 0: img = draw_text(img,'New {}!'.format(['person', 'luggage'][track.label]), above_box=track.box,color=(10,220,10), fontScale=0.7, thickness=3)
    		img = draw_box_centre(img, track.box)
    		img = draw_text(img, track.id, above_box=track.box, Yoffset = +10, color=(10, 220, 10))

    	for det in detections:
    		# img = draw_rectangle(img, det.box, color=(10, 220, 20), thickness=2)
    		img = draw_text(img, str(det.score), above_box=det.box, color=(0, 0, 0), Yoffset = -10)

    	cv2.imshow('preview', img)
    	video_writer.write(np.uint8(img))
        # stop the demo by pressing q
    	wait_ms = int(50 * dt)
    	c = cv2.waitKey(wait_ms)
    	if c == ord('q'):
    		break
    video_reader.release()
    video_writer.release() 

    cv2.destroyAllWindows()

############################ MAIN ############################

yolo = YOLOv2()
motpy.set_log_level('DEBUG')

print('*'*30)
demo_tracking_visualization()

