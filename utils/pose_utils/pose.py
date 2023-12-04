import random
import cv2
import csv
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

from utils.operation_utils import Operation
from utils.timer_utils import Timer
from utils.drawing_utils import Draw
from utils.pose_utils.const import POSE, PRESENCE_THRESHOLD, VISIBILITY_THRESHOLD

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class Pose():
    """ Base: Pose Class """
    def __init__(self, video_reader) -> None:
        self.video_reader = video_reader
        self.operation = Operation()
        self.pushup_counter = self.plank_counter = self.squat_counter = 0
        self.key_points = self.prev_pose = self.current_pose = None
        self.ang1_tracker = []
        self.ang4_tracker = []
        self.pose_tracker = []
        self.DataFile = []
        self.headpoint_tracker = []
        self.width = int(self.video_reader.get_frame_width())
        self.height = int(self.video_reader.get_frame_height())
        self.video_fps = self.video_reader.get_video_fps()
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.draw = Draw(self.width, self.height)

    def pose_algorithm(self):
        """ Pose subclass algorithm """
        raise NotImplementedError("Requires Subclass implementation.")

    def measure(self):
        """ Pose subclass measure pose """
        raise NotImplementedError("Requires Subclass implementation.")

    def get_keypoints(self, image, pose_result):
        """ Get keypoints """
        key_points = {}
        image_rows, image_cols, _ = image.shape
        for idx, landmark in enumerate(pose_result.pose_landmarks.landmark):
            if ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                continue
            landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                            image_cols, image_rows)
            if landmark_px:
                key_points[idx] = landmark_px
        return key_points

    def is_point_in_keypoints(self, str_point):
        """ Check if point is in keypoints """
        return POSE[str_point] in self.key_points

    def get_point(self, str_point):
        """ Get point from keypoints """
        return self.key_points[POSE[str_point]] if self.is_point_in_keypoints(str_point) else None

    def get_available_point(self, points):
        """
        Get highest priority keypoint from points list.
        i.e. first index is 1st priority, second index is 2nd priority, and so on.
        """
        available_point = None
        for point in points:
            if self.is_point_in_keypoints(point) and available_point is None:
                available_point = self.get_point(point)
                break
        return available_point

    def two_line_angle(self, str_point1, str_point2, str_point3):
        """ Angle between two lines """
        coord1 = self.get_point(str_point1)
        coord2 = self.get_point(str_point2)
        coord3 = self.get_point(str_point3)
        return self.operation.angle(coord1, coord2, coord3)

    def one_line_angle(self, str_point1, str_point2):
        """ Angle of a line """
        coord1 = self.get_point(str_point1)
        coord2 = self.get_point(str_point2)
        return self.operation.angle_of_singleline(coord1, coord2)

    def predict_pose(self):
        """ Predict pose """
        ang1 = ang2 = ang3 = ang4 = None
        is_pushup = is_plank = is_squat = is_jumping_jack = False
        diff_head_hand_y = None

        # Calculate angle between lines shoulder-elbow, elbow-wrist
        if self.is_point_in_keypoints("left_shoulder") and \
            self.is_point_in_keypoints("left_elbow") and \
            self.is_point_in_keypoints("left_wrist"):
            ang1 = self.two_line_angle("left_shoulder", "left_elbow", "left_wrist")
        elif self.is_point_in_keypoints("right_shoulder") and \
            self.is_point_in_keypoints("right_elbow") and \
            self.is_point_in_keypoints("right_wrist"):
            ang1 = self.two_line_angle("right_shoulder", "right_elbow", "right_wrist")
        else:
            pass

        # Calculate angle between lines shoulder-hip, hip-ankle
        if self.is_point_in_keypoints("left_shoulder") and \
            self.is_point_in_keypoints("left_hip") and \
            self.is_point_in_keypoints("left_ankle"):
            ang2 = self.two_line_angle("left_shoulder", "left_hip", "left_ankle")
        elif self.is_point_in_keypoints("right_shoulder") and \
            self.is_point_in_keypoints("right_hip") and \
            self.is_point_in_keypoints("right_ankle"):
            ang2 = self.two_line_angle("right_shoulder", "right_hip", "right_ankle")
        else:
            pass

        # Calculate angle of line shoulder-ankle or hip-ankle
        left_shoulder_ankle = self.is_point_in_keypoints("left_shoulder") and self.is_point_in_keypoints("left_ankle")
        right_shoulder_ankle = self.is_point_in_keypoints("right_shoulder") and self.is_point_in_keypoints("right_ankle")
        left_hip_ankle = self.is_point_in_keypoints("left_hip") and self.is_point_in_keypoints("left_ankle")
        right_hip_ankle = self.is_point_in_keypoints("right_hip") and self.is_point_in_keypoints("right_ankle")
        if left_shoulder_ankle or right_shoulder_ankle:
            shoulder = "left_shoulder" if left_shoulder_ankle else "right_shoulder"
            ankle = "left_ankle" if left_shoulder_ankle else "right_ankle"
            ang3 = self.one_line_angle(shoulder, ankle)
        elif left_hip_ankle or right_hip_ankle:
            hip = "left_hip" if left_hip_ankle else "right_hip"
            ankle = "left_ankle" if left_hip_ankle else "right_ankle"
            ang3 = self.one_line_angle(hip, ankle)
        else:
            pass

        # Calculate angle of line elbow-wrist
        left_elbow_wrist = self.is_point_in_keypoints("left_elbow") and self.is_point_in_keypoints("left_wrist")
        right_elbow_wrist = self.is_point_in_keypoints("right_elbow") and self.is_point_in_keypoints("right_wrist")
        if left_elbow_wrist or right_elbow_wrist:
            elbow = "left_elbow" if left_elbow_wrist else "right_elbow"
            wrist = "left_wrist" if left_elbow_wrist else "right_wrist"
            ang4 = self.one_line_angle(elbow, wrist)
        else:
            pass

        # Calculate angle of line knee-ankle
        left_knee_ankle = self.is_point_in_keypoints("left_knee") and self.is_point_in_keypoints("left_ankle")
        right_knee_ankle = self.is_point_in_keypoints("right_knee") and self.is_point_in_keypoints("right_ankle")
        if left_knee_ankle or right_knee_ankle:
            knee = "left_knee" if left_knee_ankle else "right_knee"
            ankle = "left_ankle" if left_knee_ankle else "right_ankle"
            ang5 = self.one_line_angle(knee, ankle)
        else:
            pass

         # Calculate angle of line hip-knee
        left_hip_knee = self.is_point_in_keypoints("left_hip") and self.is_point_in_keypoints("left_knee")
        right_hip_knee = self.is_point_in_keypoints("right_hip") and self.is_point_in_keypoints("right_knee")
        if left_hip_knee or right_hip_knee:
            knee = "left_knee" if left_hip_knee else "right_knee"
            hip = "left_hip" if left_hip_knee else "right_hip"
            ang6 = self.one_line_angle(hip, knee)
        else:
            pass

        if ang3 is not None and ((0 <= ang3 <= 50) or (130 <= ang3 <= 180)):
            if (ang1 is not None or ang2 is not None) and ang4 is not None:
                if (160 <= ang2 <= 180) or (0 <= ang2 <= 20):
                    self.pushup_counter += 1
                    self.ang1_tracker.append(ang1)
                    self.ang4_tracker.append(ang4)

        if self.pushup_counter >= 24 and len(self.ang1_tracker) == 24 and len(self.ang4_tracker) == 24:
            ang1_diff1 = abs(self.ang1_tracker[0] - self.ang1_tracker[12])
            ang1_diff2 = abs(self.ang1_tracker[12] - self.ang1_tracker[23])
            ang1_diff_mean = (ang1_diff1 + ang1_diff2) / 2
            ang4_mean = sum(self.ang4_tracker) / len(self.ang4_tracker)
            del self.ang1_tracker[0]
            del self.ang4_tracker[0]
            if ang1_diff_mean < 5 and not 75 <= ang4_mean <= 105:
                is_plank = True
                is_pushup = is_squat = is_jumping_jack = False
            else:
                is_pushup = True
                is_plank = is_squat = is_jumping_jack = False

        # Distance algorithm
        head_point = self.get_available_point(["nose", "left_ear", "right_ear", "left_eye", "right_eye"])
        hip = self.get_available_point(["left_hip", "right_hip"])
        knee = self.get_available_point(["left_knee", "right_knee"])
        foot = self.get_available_point(["left_foot_index", "right_foot_index", "left_heel", "right_heel", "left_ankle", "right_ankle"])

        hand_point = self.get_available_point(["left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index"])
        if head_point is not None and hand_point is not None:
            self.headpoint_tracker.append(head_point[1]) # height only
            diff_head_hand_y = head_point[1] - hand_point[1]
        if ang3 is not None and ang5 is not None and diff_head_hand_y is not None:
            if ((70 <= ang3 <= 110) or (70 <= ang5 <= 110)) and len(self.headpoint_tracker) == 24:
                height_mean = int(sum(self.headpoint_tracker) / len(self.headpoint_tracker))
                height_norm = self.operation.normalize(height_mean, head_point[1], foot[1])
                del self.headpoint_tracker[0]
                if height_norm < 0 and diff_head_hand_y < 0 and not 70 <= abs(ang6) <= 110:
                    is_squat = True
                    is_pushup = is_plank = is_jumping_jack = False
                else:
                    is_squat = False

        if diff_head_hand_y is not None and ang3 is not None:
            if diff_head_hand_y > 0 and 80 <= ang3 <= 100:
                is_jumping_jack = True
                is_pushup = is_plank = is_squat = False
            if diff_head_hand_y < 0 and is_jumping_jack is True:
                is_jumping_jack = False

        if len(self.ang1_tracker) == 24:
            del self.ang1_tracker[0]
        if len(self.ang4_tracker) == 24:
            del self.ang4_tracker[0]
        if len(self.headpoint_tracker) == 24:
            del self.headpoint_tracker[0]

        if is_pushup:
            return "Pushup"
        elif is_plank:
            return "Plank"
        elif is_squat:
            return "Squat"
        elif is_jumping_jack:
            return "JumpingJack"

        return None

    def estimate(self) -> None:
        """ Estimate pose (base function) """
        if self.video_reader.is_opened() is False:
            print("Error File Not Found.")

        out = cv2.VideoWriter("output.avi", self.fourcc, self.video_fps, (self.width, self.height))
        while self.video_reader.is_opened():
            image = self.video_reader.read_frame()
            if image is None:
                print("Ignoring empty camera frame.")
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = self.draw.overlay(image)
            image = self.draw.skeleton(image, results)

            if results.pose_landmarks is not None:
                self.key_points = self.get_keypoints(image, results)
                estimated_pose = self.predict_pose()
                if estimated_pose is not None:
                    self.current_pose = estimated_pose
                    self.pose_tracker.append(self.current_pose)
                    if len(self.pose_tracker) == 10 and len(set(self.pose_tracker[-6:])) == 1:
                        image = self.draw.pose_text(image, "Prediction: " + estimated_pose)

            if len(self.pose_tracker) == 10:
                del self.pose_tracker[0]
                self.prev_pose = self.pose_tracker[-1]

            out.write(image)
            cv2.imshow('Estimation of Exercise', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.video_reader.release()

class Neutral(Pose):
    def __init__(self, video_reader) -> None:
        super().__init__(video_reader)
        self.video_reader = video_reader
        self.is_valid = False
    
    def pose_algorithm(self):
        """ Neutral algorithm """
        
        #Calculate angle between lines elbow-shoulder and X-Axis
        if self.is_point_in_keypoints("left_elbow") and \
            self.is_point_in_keypoints("left_shoulder"):
            # self.is_point_in_keypoints("right_hip"):
            self.ang1 = self.one_line_angle("left_elbow", "left_shoulder")
        
        else:
            pass

        # Calculate angle between lines wrist-elbow and X-Axis
        if self.is_point_in_keypoints("left_wrist") and \
            self.is_point_in_keypoints("left_elbow"):
            # self.is_point_in_keypoints("right_shoulder"):
            self.ang2 = self.one_line_angle("left_wrist", "left_elbow")
        
        else:
            pass
        
        #Angle between Shoulder and Wrist
        if(self.ang2<0):
            self.ang3 = 180 - abs(self.ang1) + abs(self.ang2)
        else:
            self.ang3 = 180 - abs(self.ang1) - abs(self.ang2)
        
        # #Test Pass or Fail Criteria
        # if((abs(self.ang1) < 90) and (self.ang2 > 18) and (self.ang3 > 70)):
        #     self.is_valid = True

    
    def measure(self) -> None:
        
        if self.video_reader.is_opened() is False:
            print("Error File Not Found.")
        progress_counter = 0
        progress_bar_color = (255, 255, 255)
        out = cv2.VideoWriter("Neutral.avi", self.fourcc, self.video_fps, (self.width, self.height))
        frame_counter = 0
        while self.video_reader.is_opened():
            image = self.video_reader.read_frame()
            frame_counter = frame_counter + 1
            if image is None:
                print("Ignoring empty camera frame.")
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # overlay
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = self.draw.overlay(image)
            image = self.draw.skeleton(image, results)

            # progress bar
            image = cv2.rectangle(image, (0, self.height//8 - 10), (self.width//60 * progress_counter, self.height//8),
                                        progress_bar_color, cv2.FILLED)
            if results.pose_landmarks is not None:
                self.key_points = self.get_keypoints(image, results)
                self.pose_algorithm()
                # image = self._draw(image)
                self.DataFile.append(str(frame_counter) + "," + str(self.ang1) + "," + str(self.ang2) 
                                         + "," + str(self.ang3))
                if(self.is_valid==False):
                    image = self.draw.pose_text(image, "Frame-" + str(frame_counter) +"::" + str(self.ang3))
                    
                else:
                    image = self.draw.pose_text(image, "Test Passed" )

            out.write(image)
            cv2.imshow('Neutral', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.video_reader.release()
        with open('Neutral.csv','w',newline='') as file:
            for item in self.DataFile:
                file.write(item + '\n')


class Shoulder_extension(Pose):

    def __init__(self, video_reader) -> None:
        super().__init__(video_reader)
        self.video_reader = video_reader
        self.is_valid = False
    
    
    def pose_algorithm(self):
        """ Shoulder_Extension algorithm """

        # Calculate angle between lines shoulder-elbow, elbow-wrist
        if self.is_point_in_keypoints("right_elbow") and \
            self.is_point_in_keypoints("right_shoulder"):
            # self.is_point_in_keypoints("right_hip"):
            self.ang1 = self.one_line_angle("right_elbow", "right_shoulder")
        elif self.is_point_in_keypoints("left_elbow") and \
            self.is_point_in_keypoints("left_shoulder"):
            # self.is_point_in_keypoints("left_hip"):
            self.ang1 = self.one_line_angle("left_elbow", "left_shoulder")
        
        else:
            pass

        if self.ang1 is not None and (( self.ang1 >= 70) or ( self.ang1 >= 110)):
            self.is_valid = True
    
    def measure(self) -> None:
        """ Measure planks (base function) """
        if self.video_reader.is_opened() is False:
            print("Error File Not Found.")

        progress_counter = 0
        frame_counter = 0
        
        progress_bar_color = (255, 255, 255)
        out = cv2.VideoWriter("shoulder_extension.avi", self.fourcc, self.video_fps, (self.width, self.height))
        while self.video_reader.is_opened():
            image = self.video_reader.read_frame()
            frame_counter = frame_counter + 1
            if image is None:
                print("Ignoring empty camera frame.")
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # overlay
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = self.draw.overlay(image)
            image = self.draw.skeleton(image, results)

            # progress bar
            image = cv2.rectangle(image, (0, self.height//8 - 10), (self.width//60 * progress_counter, self.height//8),
                                        progress_bar_color, cv2.FILLED)
            if results.pose_landmarks is not None:
                self.key_points = self.get_keypoints(image, results)
                self.pose_algorithm()
                # image = self._draw(image)
                self.DataFile.append(str(frame_counter)+ "," + str(self.ang1))
                if(self.is_valid==False):
                    image = self.draw.pose_text(image, "Angle: " + str(self.ang1))
                else:
                    image = self.draw.pose_text(image, "Test Passed" )

            out.write(image)
            cv2.imshow('Shoulder_Extension', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.video_reader.release()
        with open('shoulder_extension.csv','w',newline='') as file:
            for item in self.DataFile:
                file.write(item + '\n')

class Hand_to_shoulder(Pose):
    
    def __init__(self, video_reader) -> None:
        super().__init__(video_reader)
        self.video_reader = video_reader
        self.is_valid = False
    
    def pose_algorithm(self):
        """ Left Hand to Shoulder algorithm """
        

        #Calculate angle between lines elbow-shoulder and X-Axis
        if self.is_point_in_keypoints("left_elbow") and \
            self.is_point_in_keypoints("left_shoulder"):
            # self.is_point_in_keypoints("right_hip"):
            self.ang1 = self.one_line_angle("left_elbow", "left_shoulder")
        
        else:
            pass

        # Calculate angle between lines wrist-elbow and X-Axis
        if self.is_point_in_keypoints("left_wrist") and \
            self.is_point_in_keypoints("left_elbow"):
            # self.is_point_in_keypoints("right_shoulder"):
            self.ang2 = self.one_line_angle("left_wrist", "left_elbow")
        
        else:
            pass
        
        #Angle between Shoulder and Wrist
        if(self.ang2<0):
            self.ang3 = 180 - abs(self.ang1) + abs(self.ang2)
        else:
            self.ang3 = 180 - abs(self.ang1) - abs(self.ang2)
        
        #Test Pass or Fail Criteria
        if((abs(self.ang1) < 90) and (self.ang2 > 18) and (self.ang3 > 70)):
            self.is_valid = True

    
    def measure(self) -> None:
        
        if self.video_reader.is_opened() is False:
            print("Error File Not Found.")
        progress_counter = 0
        progress_bar_color = (255, 255, 255)
        out = cv2.VideoWriter("hand_to_shoulder.avi", self.fourcc, self.video_fps, (self.width, self.height))
        frame_counter = 0
        while self.video_reader.is_opened():
            image = self.video_reader.read_frame()
            frame_counter = frame_counter + 1
            if image is None:
                print("Ignoring empty camera frame.")
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # overlay
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = self.draw.overlay(image)
            image = self.draw.skeleton(image, results)

            # progress bar
            image = cv2.rectangle(image, (0, self.height//8 - 10), (self.width//60 * progress_counter, self.height//8),
                                        progress_bar_color, cv2.FILLED)
            if results.pose_landmarks is not None:
                self.key_points = self.get_keypoints(image, results)
                self.pose_algorithm()
                # image = self._draw(image)
                self.DataFile.append(str(frame_counter) + "," + str(self.ang1) + "," + str(self.ang2) 
                                         + "," + str(self.ang3))
                if(self.is_valid==False):
                    image = self.draw.pose_text(image, "Frame-" + str(frame_counter) +"::" + str(self.ang3))
                    
                else:
                    image = self.draw.pose_text(image, "Test Passed" )

            out.write(image)
            cv2.imshow('Hand_to_Shoulder', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.video_reader.release()
        with open('left_hand_to_shoulder.csv','w',newline='') as file:
            for item in self.DataFile:
                file.write(item + '\n')
            
class Forward_reach(Pose):
    def __init__(self, video_reader) -> None:
        super().__init__(video_reader)
        self.video_reader = video_reader
        self.is_valid = False
    
    
    def pose_algorithm(self):
        """ Forward_reach algorithm """

        # Calculate angle between lines shoulder-elbow, elbow-wrist
        if self.is_point_in_keypoints("right_elbow") and \
            self.is_point_in_keypoints("right_shoulder"):
            # self.is_point_in_keypoints("right_hip"):
            self.ang1 = self.one_line_angle("right_elbow", "right_shoulder")
        elif self.is_point_in_keypoints("left_elbow") and \
            self.is_point_in_keypoints("left_shoulder"):
            # self.is_point_in_keypoints("left_hip"):
            self.ang1 = self.one_line_angle("left_elbow", "left_shoulder")
        
        else:
            pass

        # if self.ang1 is not None and (( self.ang1 >= 70) or ( self.ang1 >= 110)):
        #     self.is_valid = True
    
    def measure(self) -> None:

        if self.video_reader.is_opened() is False:
            print("Error File Not Found.")

        frame_counter = 0
        progress_counter = 0
        progress_bar_color = (255, 255, 255)
        out = cv2.VideoWriter("Right_forward_reach.avi", self.fourcc, self.video_fps, (self.width, self.height))
        while self.video_reader.is_opened():
            image = self.video_reader.read_frame()
            frame_counter = frame_counter + 1

            if image is None:
                print("Ignoring empty camera frame.")
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # overlay
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = self.draw.overlay(image)
            image = self.draw.skeleton(image, results)

            # progress bar
            image = cv2.rectangle(image, (0, self.height//8 - 10), (self.width//60 * progress_counter, self.height//8),
                                        progress_bar_color, cv2.FILLED)
            if results.pose_landmarks is not None:
                self.key_points = self.get_keypoints(image, results)
                self.pose_algorithm()
                # image = self._draw(image)

                self.DataFile.append(str(frame_counter)+ "," + str(self.ang1))
                if(self.is_valid==False):
                    image = self.draw.pose_text(image, "Angle: " + str(self.ang1))
                    
                else:
                    image = self.draw.pose_text(image, "Test Passed" )

            out.write(image)
            cv2.imshow('Forward_reach', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.video_reader.release()
        with open('Forward_reach.csv','w',newline='') as file:
            for item in self.DataFile:
                file.write(item + '\n')

class external_rotation(Pose):
    def __init__(self, video_reader) -> None:
        super().__init__(video_reader)
        self.video_reader = video_reader
        self.is_valid = False  

    def pose_algorithim(self):
        "algorithim for external rotation"
        if self.is_point_in_keypoints("right_shoulder") and \
            self.is_point_in_keypoints("right_elbow"):
                self.ang1 = self.one_line_angle("right_elbow", "right_shoulder")
        else:
            print('error')
        
        if self.is_point_in_keypoints("right_elbow") and \
            self.is_point_in_keypoints("right_wrist"):
                self.ang2 = self.one_line_angle("right_elbow", "right_wrist")
        
    
    def measure(self) -> None:
        pass
