#!/usr/bin/env python

# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

"""
This file starts a ROS node to run DOPE, 
listening to an image topic and publishing poses.
"""

from __future__ import print_function

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw

import sys 
sys.path.append("inference")
from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector
import torchvision.transforms as transforms

import simplejson as json

class Draw(object):
    """Drawing helper class to visualize the neural network output"""

    def __init__(self, im):
        """
        :param im: The image to draw in.
        """
        self.draw = ImageDraw.Draw(im)

    def draw_line(self, point1, point2, line_color, line_width=2):
        """Draws line on image"""
        if point1 is not None and point2 is not None:
            self.draw.line([point1, point2], fill=line_color, width=line_width)

    def draw_dot(self, point, point_color, point_radius=2):
        """Draws dot (filled circle) on image"""
        if point is not None:
            xy = [
                point[0] - point_radius,
                point[1] - point_radius,
                point[0] + point_radius,
                point[1] + point_radius
            ]
            self.draw.ellipse(xy,
                              fill=point_color,
                              outline=point_color
                              )

    def draw_cube(self, points, color=(255, 0, 0)):
        """
        Draws cube with a thick solid line across
        the front top edge and an X on the top face.
        """

        # draw front
        self.draw_line(points[0], points[1], color)
        self.draw_line(points[1], points[2], color)
        self.draw_line(points[3], points[2], color)
        self.draw_line(points[3], points[0], color)

        # draw back
        self.draw_line(points[4], points[5], color)
        self.draw_line(points[6], points[5], color)
        self.draw_line(points[6], points[7], color)
        self.draw_line(points[4], points[7], color)

        # draw sides
        self.draw_line(points[0], points[4], color)
        self.draw_line(points[7], points[3], color)
        self.draw_line(points[5], points[1], color)
        self.draw_line(points[2], points[6], color)

        # draw dots
        self.draw_dot(points[0], point_color=color, point_radius=4)
        self.draw_dot(points[1], point_color=color, point_radius=4)

        # draw x on the top
        self.draw_line(points[0], points[5], color)
        self.draw_line(points[1], points[4], color)


class DopeNode(object):
    """ROS node that listens to image topic, runs DOPE, and publishes DOPE results"""
    def __init__(self,
            config, # config yaml loaded eg dict
        ):
        self.pubs = {}
        self.models = {}
        self.pnp_solvers = {}
        self.pub_dimension = {}
        self.draw_colors = {}
        self.dimensions = {}
        self.class_ids = {}
        self.model_transforms = {}
        self.meshes = {}
        self.mesh_scales = {}
        self.transform = {}

        self.input_is_rectified = config['input_is_rectified']
        self.downscale_height = config['downscale_height']

        try:
            self.padding_width = config['padding_added_width']
            self.padding_height = config['padding_added_height']
        except:
            self.padding_width = 0
            self.padding_height = 0


        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = config['thresh_angle']
        self.config_detect.thresh_map = config['thresh_map']
        self.config_detect.sigma = config['sigma']
        self.config_detect.thresh_points = config["thresh_points"]



        # For each object to detect, load network model, create PNP solver, and start ROS publishers
        print(config['weights'])
        for model in config['weights']:
            print(model)
            self.models[model] = \
                ModelData(
                    model,
                    config['weights'][model]
                )
            self.models[model].load_net_model()
            print('loaded')

            try:
                self.draw_colors[model] = tuple(config["draw_colors"][model])
            except:
                self.draw_colors[model] = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))

            if model in config["dimensions"].keys():
                self.dimensions[model] = tuple(config["dimensions"][model])
                self.class_ids[model] = config["class_ids"][model]
            else:
                self.dimensions[model] = tuple(config["dimensions"][model.split("_")[0]])
                self.class_ids[model] = config["class_ids"][model.split("_")[0]]

            if len(self.dimensions[model]) == 3:
                self.pnp_solvers[model] = \
                    CuboidPNPSolver(
                        model,
                        cuboid3d=Cuboid3d(self.dimensions[model])
                    )

            else:
                self.pnp_solvers[model] = CuboidPNPSolver(model,cuboid3d=Cuboid3d())                
                self.pnp_solvers[model]._cuboid3d._vertices = self.dimensions[model]

            # if 'shiny' in model or 'original' in model:
            #     self.pnp_solvers[model]._cuboid3d._vertices = [


            # different models were trained with different transform inputs 
            self.transform[model] = transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                # transforms.Normalize((0.59, 0.59, 0.59), (0.25, 0.25, 0.25)),
            ])
            if 'visii' in config['weights']:
                self.transform[model] = transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
                    # transforms.Normalize((0.59, 0.59, 0.59), (0.25, 0.25, 0.25)),
                ])    

        # print("Running DOPE...  (Listening to camera topic: '{}')".format(config['~topic_camera')))
        print("Ctrl-C to stop")

    def image_callback(self, 
        img, 
        camera_info, 
        img_name = "00000.png", # this is the name of the img file to save, it needs the .png at the end
        output_folder = 'out_inference', # folder where to put the output
        showbelief = False,
        p_matrix = None,
        ):
        img_name = str(img_name).zfill(5)+'.png'
        """Image callback"""

        # img = self.cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")

        # cv2.imwrite('img.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # for debugging

        # Update camera matrix and distortion coefficients

        
        if self.input_is_rectified:
            P = np.matrix(camera_info['camera_matrix']['data'], dtype='float64').copy()
            # print(P)
            P.resize((3, 3))
            camera_matrix = P
            dist_coeffs = np.zeros((4, 1))
            # print(camera_matrix)
            # raise()
        else:
            # TODO
            camera_matrix = np.matrix(camera_info.K, dtype='float64')
            camera_matrix.resize((3, 3))
            dist_coeffs = np.matrix(camera_info.D, dtype='float64')
            dist_coeffs.resize((len(camera_info.D), 1))
        
        # force setting up the camera matrix from function
        if not p_matrix is None:
            P = np.matrix(p_matrix, dtype='float64').copy()
            P.resize((3, 3))
            camera_matrix = P

        # add padding to the image 
        img = cv2.copyMakeBorder( img, 0, self.padding_height, 0, self.padding_width, cv2.BORDER_CONSTANT,(0,0,0))
        # Downscale image if necessary
        height, width, _ = img.shape
        scaling_factor = float(self.downscale_height) / height
        if scaling_factor < 1.0:
            camera_matrix[:2] *= scaling_factor
            img = cv2.resize(img, (int(scaling_factor * width), int(scaling_factor * height)))

        for m in self.models:
            self.pnp_solvers[m].set_camera_intrinsic_matrix(camera_matrix)
            self.pnp_solvers[m].set_dist_coeffs(dist_coeffs)
        
        # Copy and draw image
        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        draw = Draw(im)


        # dictionary for the final output
        dict_out = {"camera_data":{},"objects":[]}

        beliefs_outputs = []

        for m in self.models:
            # Detect object
            if 'full' in self.models[m].net_path:
                network = 'full'
            elif 'mobile' in self.models[m].net_path:
                network = 'mobile'
            else:
                network = 'dope'
            
            results, beliefs = ObjectDetector.detect_object_in_image(
                self.models[m].net,
                self.pnp_solvers[m],
                img,
                self.config_detect,
                network = network,
                grid_belief_debug = showbelief,
                transform = self.transform[m]
            )
            beliefs_outputs.append(beliefs)
            # print(results)
            # print('---')
            # continue
            # Publish pose and overlay cube on image
            for i_r, result in enumerate(results):
                if result["location"] is None:
                    continue
                loc = result["location"]
                ori = result["quaternion"]
                
                # print(loc)
                raw = np.array(result['raw_points'])
                proj = np.array(result['projected_points'])

                # values_l2 = []
                # for p in raw: 
                #     if p 

                dict_out['objects'].append({
                    'class':m,
                    'location':np.array(loc).tolist(),
                    'quaternion_xyzw':np.array(ori).tolist(),
                    'projected_cuboid':np.array(result['projected_points']).tolist(),
                    'raw_cuboid':np.array(result['raw_points']).tolist(),
                    'confidence':np.array(result['confidence']).tolist(),
                    # 'reprojection_error':
                })
                # print( dict_out )

                # transform orientation
                # TODO 
                # transformed_ori = tf.transformations.quaternion_multiply(ori, self.model_transforms[m])

                # rotate bbox dimensions if necessary
                # (this only works properly if model_transform is in 90 degree angles)
                # dims = rotate_vector(vector=self.dimensions[m], quaternion=self.model_transforms[m])
                # dims = np.absolute(dims)
                # dims = tuple(dims)

                # Draw the cube
                if None not in result['projected_points']:
                    points2d = []
                    for pair in result['projected_points']:
                        points2d.append(tuple(pair))
                    draw.draw_cube(points2d, self.draw_colors[m])
                # draw the raw prediction points 
                for p in result['raw_points']:
                    draw.draw_dot(p,self.draw_colors[m])

                # print (result)

        # save the output of the image. 
        if not opt.json_only:
            # print(f'saving {output_folder}/{img_name}')
            im.save(f"{output_folder}/{img_name}")
            im.save('tmp/'+img_name)
        if opt.save_beliefs:
            beliefs_outputs[0].save(f"{output_folder}/beliefs/{img_name}")
        # save the json files 
        path_json = f"{output_folder}/{img_name.replace('png','')}json".replace("..",".")
        if os.path.exists(path_json):
            with open(path_json) as f:
                data = json.load(f)
            dict_out['objects'] = data['objects'] + dict_out['objects']
        # print(len(dict_out['objects']))
        with open(path_json, 'w') as fp:
            json.dump(dict_out, fp)
        return im, beliefs_outputs

def rotate_vector(vector, quaternion):
    q_conj = tf.transformations.quaternion_conjugate(quaternion)
    vector = np.array(vector, dtype='float64')
    vector = np.append(vector, [0.0])
    vector = tf.transformations.quaternion_multiply(q_conj, vector)
    vector = tf.transformations.quaternion_multiply(vector, quaternion)
    return vector[:3]

if __name__ == "__main__":

    import argparse
    import yaml 
    import glob 
    import os 

    parser = argparse.ArgumentParser()
    parser.add_argument("--pause",
        default=0,
        help='pause between images')
    parser.add_argument("--showbelief",
        action="store_true",
        help='show the belief maps')
    parser.add_argument("--headless",
        action="store_true",
        help='headless mode')
    parser.add_argument("--save_beliefs",
        action="store_true",
        help='save_belief_map')
    parser.add_argument("--json_only",
        action="store_true",
        help='only store the json files')
    parser.add_argument("--graph",
        action="store_true",
        help='make the graphs with GT files')
    parser.add_argument("--outf",
        default="out_experiment",
        help='where to store the output')
    parser.add_argument("--data",
        default=None,
        help='folder for data images to load, *.png (default), see --suffix')
    parser.add_argument("--suffix",
        default='*.png',
        help='default *.png, note that * is needed to search')
    parser.add_argument("--config",
        default="config_inference/config_pose.yaml",
        help='folder for the inference configs')
    parser.add_argument("--camera",
        default="config_inference/camera_info.yaml",
        help='camera info file')
    parser.add_argument("--model",
        default=None,
        nargs='+',        
        help='model to load, this overwrites the config yaml file, although it has to be defined there')

    opt = parser.parse_args()


    # load the configs
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.camera) as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)
    
    # create the output folder
    print (f"output is located in {opt.outf}")
    try:
        shutil.rmtree(f"{opt.outf}")
    except:
        pass

    try:
        os.makedirs(f"{opt.outf}")
    except OSError:
        pass


    # load the images if there are some
    imgs = []
    imgsname = []


    if not opt.data is None:
        videopath = opt.data

        imgs = []
        imgsname = []

        def add_images(path): 
            for j in sorted(glob.glob(path+"/"+opt.suffix)):
                imgs.append(j)
                imgsname.append(j.replace(path,"").replace("/",""))


        def explore(path):
            if not os.path.isdir(path):
                return
            folders = [os.path.join(path, o) for o in os.listdir(path) 
                            if os.path.isdir(os.path.join(path,o))]
            if len(folders)>0:
                for path_entry in folders:                
                    explore(path_entry)
               
            add_images(path)

        explore(opt.data)

    else:
        # if not opt.realsense:
        cap = cv2.VideoCapture(0)
    if opt.model is not None:
        new_weights = {}

        for i, w in enumerate(config['weights']):
            # print('w',w)
            if w in opt.model:
                new_weights[w] = config['weights'][w]
    
            # TODO hardcoded for testing
            # if 'Ketchup' in w:
            #     name = f"Ketchup_{opt.model[0].split('/')[-1].split('_')[-1].replace('.pth','')}"
            #     print(name)
            #     new_weights[name] = opt.model[0]

        config['weights'] = new_weights

        # print(new_weights)
    # An object to run dope node
    dope_node = DopeNode(config)


    # starting the loop here
    i_image = -1 

    while True:
        camera_matrix = None

        i_image+=1
        
        # Capture frame-by-frame
        
        folder_substructure = "" 

        if not opt.data:
            ret, frame = cap.read()
            img_name = i_image

        else:
            if i_image >= len(imgs):
                i_image = 0
                # If you want to loop around your images, please comment the break 
                break
            
            # check if there is a json file 
            try:
                with open(imgs[i_image].replace('png','json').replace('jpg','json')) as f:
                    visii_camera = json.load(f)
                print('found')
                fx = visii_camera['camera_data']['intrinsics']['fx']
                fy = visii_camera['camera_data']['intrinsics']['fy']
                x0 = visii_camera['camera_data']['intrinsics']['cx']
                y0 = visii_camera['camera_data']['intrinsics']['cy']
                camera_matrix = [fx, 0, x0, 0, fy, y0, 0, 0, 1]
            except:
                camera_matrix = None

            frame = cv2.imread(imgs[i_image])
            img_name = imgsname[i_image]

            # find the substructure
            folder_substructure = imgs[i_image].replace(opt.data,"").replace(imgsname[i_image],"")
            print(folder_substructure + img_name)
        frame = frame[...,::-1].copy()
        
        # check if the subfolder exist and create it 
        try:
            os.makedirs(f"{opt.outf}/{folder_substructure}")
        except OSError:
            pass

        if opt.save_beliefs:
            try:
                os.makedirs(f"{opt.outf}/{folder_substructure}/beliefs/")
            except OSError:
                pass

        # call the inference node
        out_img, out_beliefs = dope_node.image_callback(
            frame, 
            camera_info,
            img_name = img_name,
            output_folder = f"{opt.outf}/{folder_substructure}" ,
            showbelief = opt.showbelief,
            p_matrix = camera_matrix
            )

        if not opt.headless:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.imshow('DOPE',np.array(out_img)[...,::-1])
            if opt.showbelief:
                # TODO remove the [0] for a for loop when multiple objects. 
                cv2.imshow('DOPE BELIEFS', np.array(out_beliefs[0])[...,::-1])

    # Produce an output video
    import subprocess 
    subprocess.call(['ffmpeg', '-y',\
        '-framerate', '10', \
        # "-hide_banner", "-loglevel", "panic"\
        '-pattern_type', 'glob', '-i',\
        f"{opt.outf}/{folder_substructure}/*.png", f"{opt.outf}/{folder_substructure}/video.mp4"]) 
    if opt.save_beliefs:
        subprocess.call(['ffmpeg', '-y',\
            '-framerate', '10', \
            # "-hide_banner", "-loglevel", "panic"\
            '-pattern_type', 'glob', '-i',\
            f"{opt.outf}/{folder_substructure}/beliefs/*.png", f"{opt.outf}/{folder_substructure}/video_beliefs.mp4"]) 
    if opt.graph:
        subprocess.call(
            [
            'python', 'run_add.py', '--data_prediction',f'{opt.outf}'
            ]
        )
if not opt.headless:
    cv2.destroyAllWindows()
