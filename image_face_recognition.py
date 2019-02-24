# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import sys
import time

# to delete
import io


import json

import cv2

import face


def add_overlays(frame, faces):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)

    # cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                # thickness=2, lineType=2)


def get_facenet_results(image, face_recognition, output_image=False):
	# frame_interval = 3  # Number of frames after which to run face detection
	# fps_display_interval = 5  # seconds
	# frame_rate = 0
	# frame_count = 0
	# image=cv2.imread(image,1)
	# cv2.imshow('image',image)
	# cv2.waitKey(0)
	
	# video_capture = cv2.VideoCapture(0)
	# start_time = time.time()


	# Capture frame-by-frame
	# ret, frame = video_capture.read()

	faces = face_recognition.identify(image)

	bounding_boxes = []

	for f in faces :
		bounding_boxes.append({ "bb": f.bounding_box.tolist(), "name": f.name, "proba": f.proba })


	json_object = json.dumps(bounding_boxes)
	
	with io.open('data.json', 'w', encoding='utf-8') as f:
		f.write(json_object)
	
	# with open("data_file.json", "w") as write_file:
		# json.dump(output_dict, write_file)
	# Check our current fps
	# end_time = time.time()
	# if (end_time - start_time) > fps_display_interval:
		# frame_rate = int(frame_count / (end_time - start_time))
		# start_time = time.time()
		# frame_count = 0
	if output_image == True:
		add_overlays(image, faces)

		#output_dict = {'names_list':names, 'bounding_boxes_list':bounding_boxes}
		
		return json_object, image
	
	#output_dict = {'names_list':names, 'bounding_boxes_list':bounding_boxes}
	# output_dict = {names_list:names, bounding_boxes_list:bounding_boxes}
	return json_object
	# frame_count += 1
	# cv2.imshow('Video', frame)

	# if cv2.waitKey(1) & 0xFF == ord('q'):
		# break

    # When everything is done, release the capture
    # video_capture.release()
    # cv2.destroyAllWindows()


	
	
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
	face_recognition = face.Recognition()
    # main(parse_arguments(sys.argv[1:]))
	output_image = True
	out_dict = get_facenet_results("test_image.jpg", output_image = output_image)
	if output_image == True:
		json_out = out_dict[0]
		image_out = out_dict[1]
		cv2.imshow('image',image_out)
		cv2.waitKey(0)
	else:
		json_out = out_dict[0]

	json_data = json.loads(json_out)
	
	# json_file = open('data.json')
	# json_str = json_file.read()
	# json_data = json.loads(json_str)
	
	print(json_data)
	
