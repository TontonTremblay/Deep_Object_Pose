#!/usr/bin/env python3
import random 
import subprocess


# 20 000 images

for i in range(0, 100):
	to_call = [
		"python",'single_video_pybullet.py',
		'--spp','400',
		'--nb_frames', '200',
		'--nb_distractors',str(int(random.uniform(5,12))), 
		'--nb_objects',str(int(random.uniform(5,7))),
		'--scale', str(float(random.uniform(0.03,0.08))),
		"--path_single_obj", "models/cube.obj",
		'--outf',f"dataset/{str(i).zfill(3)}",
	]
	subprocess.call(to_call)


# stolen@10.20.12.200
# password: aithink2