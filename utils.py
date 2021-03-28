from collections import defaultdict
import glob
import pickle


def show_results():
    videoData = {}
    userdict = defaultdict(lambda: defaultdict(lambda: 0))
    for file in glob.glob("*.pickle"): 
        with open(file, "rb") as f:
            data = pickle.load(f)
        for videoId, line in data.items():
            userdict[videoId]["userCount"] += 1
            userdict[videoId]["time"] += line[0]
            userdict[videoId]["total_blinks"] += line[1]
            userdict[videoId]["total_drowsiness"] += line[2]
            userdict[videoId]["total_yawns"] += line[3]
    
    for videoId in userdict:
        videoData[videoId] = {}
        userC = userdict[videoId]["userCount"]
        videoData[videoId]["userCount"] = userC
        videoData[videoId]["time"] = userdict[videoId]["time"] / userC
        videoData[videoId]["total_blinks"] = userdict[videoId]["total_blinks"] / userC
        videoData[videoId]["total_drowsiness"] = userdict[videoId]["total_drowsiness"] / userC
        videoData[videoId]["total_yawns"] = userdict[videoId]["total_yawns"] / userC
    return videoData


def calc_results(username, videoId, frameCount, cfg, total_blinks, total_drowsiness, total_yawns):
    try: 
        with open(str(username)+".pickle", "rb") as f:
            results = pickle.load(f)
    except:
        results = {}
    
    if videoId not in results:
        results[videoId] = (0, 0, 0, 0)
    res = results[videoId]
    results[videoId] = (res[0]+frameCount/cfg.getint('CAMERA', 'fps'), res[1]+total_blinks[videoId],
                        res[2]+total_drowsiness[videoId], res[3]+total_yawns[videoId])
    
    # dump pickle file
    with open(str(username)+".pickle", "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
