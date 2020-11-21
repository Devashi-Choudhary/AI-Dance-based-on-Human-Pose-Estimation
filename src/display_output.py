#The code is inspired from https://github.com/keshavoct98/DANCING-AI

import cv2
import time
import statistics
POSE_PAIRS = [ [1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[2,8],[8,9],[9,10],[5,11],[11,12],[12,13],[8,11]]

def distance(point_a, point_b): # Calculates distance between two points.
    return (((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)**0.5)

''' Returns True if outliers are present in the passed list of
    coordinates, otherwise returns false.'''
def check_outliers(points):
    lengths = []
    for pair in POSE_PAIRS:
        partA, partB = pair[0], pair[1]
        if points[partA] and points[partB]:
            lengths.append(distance(points[partA], points[partB]))
    for length in lengths:
        if length > (statistics.median(lengths) + 2.5 * statistics.pstdev(lengths)):
            return True
    return False


''' Returns video of human-stick figure dancing. Figures are created
    by joining predicted pose coordinates. prdictions with outliers are 
    removed. Figure is drawn over a background image.'''    
def displayResults(predictions, background_path):
    img = cv2.imread(background_path)
    vid_writer = cv2.VideoWriter('outputs/output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 5, (700,480))
    for k in range(predictions.shape[0]):
        time.sleep(0.18)
        frame = img.copy()
        
        points = []
        for i in range(0,28,2):
            points.append((int(predictions[k,i]),) + (int(predictions[k,i+1]),))     
        if check_outliers(points) or points[0][1] >= points[2][1] or points[0][1] >= points[5][1]:
            continue
            
        for pair in POSE_PAIRS:
            partA, partB = pair[0], pair[1]
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (146, 77, 14), 2, lineType = 8)
                cv2.circle(frame, points[partA], 5, (0, 0, 150), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 5, (0, 0, 150), thickness=-1, lineType=cv2.FILLED)
        x1, x2, y = int((points[0][0] + points[2][0])/2), int((points[0][0] + points[5][0])/2), points[0][1]
        cv2.circle(frame, (x1, y), 5, (0, 0, 150), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, (x2, y), 5, (0, 0, 150), thickness=-1, lineType=cv2.FILLED)
        
        cv2.imshow('Output-Skeleton', frame)
        vid_writer.write(frame)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    vid_writer.release()