import os
import dlib
import cv2
import time
import glob
import pickle
from collections import defaultdict
from flask import Flask, render_template, request
from camera import VideoRecorder
from processing import analyze_frame
from utils import calc_results, show_results
import configparser
import time

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# global configs
global recorder
global blink_counter  # total_blinks
global yawn_counter  # total_yawns
global frameCount
recorder = None
frameCount = 0

# Blink and Yawn initializations
blink_counter, yawn_counter = 0, 0
total_blinks, total_yawns = defaultdict(lambda: 0), defaultdict(lambda: 0)
total_drowsiness = defaultdict(lambda: 0)

# read the config file.
cfg = configparser.ConfigParser()
cfg.read('config.INI')


def gen(camera, videoId, username):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    global recorder, blink_counter, yawn_counter
    global frameCount
    frameCount = 0
    prev = 0

    while recorder:
        time_elapsed = time.time() - prev
        frame = camera.get_frame()
        if time_elapsed < 1/cfg.getint('CAMERA', 'fps'):    # to handle number of frames to be processed in a second.
            continue

        frameCount += 1
        prev = time.time()
        if len(frame)==0:
            print(videoId, frameCount/cfg.getint('CAMERA', 'fps'), total_blinks, total_drowsiness, total_yawns)
            break

        blinking, yawning, frame = analyze_frame(frame, cfg, detector, predictor)

        if blinking:
            blink_counter += 1
        else:
            if blink_counter > cfg.getint('YAWN', 'drowsiness_thresh'):
                total_drowsiness[videoId] += 1
                print("Added to the total drowsiness!")
            elif blink_counter > cfg.getint('YAWN', 'blink_thresh'):
                total_blinks[videoId] += 1
                print("Added to the total blinks!")
            blink_counter = 0

        if yawning:
            yawn_counter += 1
        else:
            if yawn_counter > cfg.getint('YAWN', 'yawn_thresh'):
                total_yawns[videoId] += 1
                print("Added to the total yawns!")
            yawn_counter = 0


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    try:
        videoId = request.args['id']
    except:
        videoId = 0
    
    try:
        username = request.args['username']
        # print(f"username: {username}")
    except:
        username = "user-0"
    global recorder
    recorder = VideoRecorder()
    gen(recorder, videoId, username)
    return {}


@app.route('/video_stop', methods=['GET', 'POST'])
def video_stop():
    try:
        videoId = request.args['id']
    except:
        videoId = 0
    try:
        username = request.args['username']
        # print(f"username: {username}")
    except:
        username = "user-0"

    global recorder, blink_counter, yawn_counter
    global frameCount
    del recorder
    recorder = None
    # reset blink and yawn counters.
    if blink_counter > cfg.getint('YAWN', 'drowsiness_thresh'):
        total_drowsiness[videoId] += 1
        print("Added to the total drowsiness!")
    elif blink_counter > cfg.getint('YAWN', 'blink_thresh'):
        total_blinks[videoId] += 1
        print("Added to the total blinks!")
    if yawn_counter > cfg.getint('YAWN', 'yawn_thresh'):
        total_yawns[videoId] += 1

    calc_results(username, videoId, frameCount, cfg, total_blinks, total_drowsiness, total_yawns)

    # reset dictionary values.
    total_drowsiness.clear()
    total_blinks.clear()
    total_yawns.clear()
    blink_counter, yawn_counter = 0, 0
    print("resource released!")

    videoData = show_results()

    return render_template('results.html', videoData = videoData)


@app.route('/results')
def results():
    videoData = show_results()
    return render_template('results.html', videoData = videoData)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='localhost', port=port)
