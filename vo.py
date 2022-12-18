import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
import random
from orb import  update_image
from voc_tree import *
import sys, os, argparse, glob
import re
import open3d.visualization.gui as gui

K =  5#聚类K类
L = 3 #字典树L层
T = 0.90 #相似度阈值


def cos_sim(vec_a, vec_b):
    """
    计算余弦相似度
    """
    vec_a = np.mat(vec_a)
    vec_b = np.mat(vec_b)
    if np.linalg.norm(vec_a) * np.linalg.norm(vec_b) == 0:
        return 0
    cos = float(vec_a * vec_b.T) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    sim = 0.5 + 0.5 * cos

    return sim


class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        
        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    axes = o3d.geometry.LineSet()
                    axes.points = o3d.utility.Vector3dVector( [[0, 0, 1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]])
                    axes.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3],[0,4],[1,2],[2,3],[3,4],[4,1]])  # X, Y, Z
                    axes.colors = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 0], [0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0]])  # R, G, B
                    axes.rotate(R)
                    axes.translate(t)
                    vis.add_geometry(axes)
            except: pass
            
            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def feature_matching(self,img1,img2):
        # Initiate ORB detector
        orb = cv.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        #for loop detection
        imgdes = des2.astype(int)
        imgdes = imgdes.tolist()

        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        points1 = np.array([kp1[m.queryIdx].pt for m in matches])
        points2 = np.array([kp2[m.trainIdx].pt for m in matches])

        # for loop detection
        kp2 = [kp2[m.trainIdx] for m in matches]

        return imgdes,points1,points2,kp2

    def triangulate(self,essentialmat,points1,points2):
        points, R, T, inlier, triangulatedPoint = cv.recoverPose(essentialmat, points1, points2, self.K,
                                                            distanceThresh=400)
        triangulatedPoint = triangulatedPoint.T
        points1 = np.array([point for i,point in enumerate(points1) if inlier[i][0] == 255])
        points2 = np.array([point for i, point in enumerate(points2) if inlier[i][0] == 255])

        triangulatedPoint = np.array([point/point[3] for i, point in enumerate(triangulatedPoint) if inlier[i][0] == 255])


        return R,T,points1,points2,triangulatedPoint

    def essential(self,points1,points2):
        essentialmat, inlier = cv.findEssentialMat(points1, points2, cameraMatrix=self.K)
        points1 = np.array([point for i,point in enumerate(points1) if inlier[i][0] == 1])
        points2 = np.array([point for i, point in enumerate(points2) if inlier[i][0] == 1])
        return essentialmat,points1,points2

    def get_scale(self,prepoints,pretriangulates,points1, curtriangulates):
        sameprepoints = []
        samepoints1 = []
        for i in range(len(prepoints)):
            for j in range(len(points1)):
                if prepoints[i][0] == points1[j][0] and prepoints[i][1] == points1[j][1]:
                    sameprepoints.append(pretriangulates[i])
                    samepoints1.append(curtriangulates[j])
                    break
        length = len(sameprepoints)
        if length <= 1:
            return 1
        ratios = []
        for i in range(20):
            idx1 = random.randint(0, length - 1)
            idx2 = random.randint(0, length - 1)
            if idx1 == idx2:
                continue
            pre = np.linalg.norm(sameprepoints[idx1] - sameprepoints[idx2])
            cur = np.linalg.norm(samepoints1[idx1] - samepoints1[idx2])
            ratios.append(cur/(pre+0.1))
        ratios = np.array(ratios)
        return np.median(ratios)

    def loop_detection(self,FEATS):
        has = False
        counter = 0
        pre = 0
        pre_loop = -1
        FEATSstack = np.vstack(FEATS)
        treeArray = constructTree(K, L, np.vstack(FEATSstack))
        tree = Tree(K, L, treeArray)
        pair = [-1, -1]
        for index, frame_path in enumerate(self.frame_paths[1:]):
            des = update_image(frame_path)
            tree.update_tree(frame_path, des)
            res = {}
            for index2, j in enumerate(self.frame_paths[:index + 1]):
                if index - index2 < 30:
                    break
                if cos_sim(tree.transform(frame_path), tree.transform(j)) >= T:
                    res[j] = cos_sim(tree.transform(frame_path), tree.transform(j))
            if res:
                r = max(res.items(), key=lambda x: x[1])[0]
                mapped = int(re.search(r'\d+', r).group())
                if counter == 0:
                    counter = counter + 1
                    pre = mapped
                    pair[0] = frame_path
                    pair[1] = r
                else:
                    if mapped - pre == 1:
                        counter = counter + 1
                        pre = mapped
                    else:
                        counter = 1
                        pre = mapped
                        pair[0] = frame_path
                        pair[1] = r
                    if counter == 3:
                        if index - pre_loop > 30:
                            print('Loop founding! Start at' +  pair[1] + ', end at ' + pair[0])
                            pre_loop = index
                            has = True
                        else:
                            pass
                        counter = 0
            else:
                counter = 0
        if has == False:
            print('There is no loop!')

    def process_frames(self, queue):
        Rt_prev = np.eye(4)

        preframe = cv.imread(self.frame_paths[0])
        preframe = cv.undistort(preframe,self.K,self.dist,None,self.K)

        FEATS = []
        des = update_image(self.frame_paths[0])
        FEATS += [np.array(des, dtype='float32')]
        for index,frame_path in enumerate(self.frame_paths[1:]):
            img = cv.imread(frame_path)
            img = cv.undistort(img, self.K, self.dist,None ,self.K)
            imgdes,points1, points2,kp2 = self.feature_matching(preframe, img)
            FEATS += [np.array(imgdes, dtype='float32')]
            essentialmat, points1, points2 = self.essential(points1, points2)
            R, t, points1, points2, curtriangulates = self.triangulate(essentialmat, points1, points2)

            if index == 0:
                scale = 1
            else:
                scale = self.get_scale(prepoints, pretriangulates, points1, curtriangulates)
                scale = prescale * scale
                scale = min(max(0.5,scale),2)
                t = t * scale

            Rt = np.concatenate((R, t), axis=1)
            Rt = np.concatenate((Rt, [[0, 0, 0, 1]]), axis=0)
            Rt = Rt_prev.dot(Rt)


            preframe = img
            prepoints = points2
            pretriangulates = curtriangulates
            prescale = scale
            Rt_prev = Rt



            img = cv.drawKeypoints(img, kp2, None, color=(0, 255, 0), flags=0)
            cv.imshow('frame', img)
            if cv.waitKey(30) == 27: break
            queue.put((Rt[:3, :3], Rt[:3, 3]))
        self.loop_detection(FEATS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
