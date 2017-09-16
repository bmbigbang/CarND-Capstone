from styx_msgs.msg import TrafficLight
import cv2
import rospy


class TLClassifier(object):
    def __init__(self, model):
        #TODO load classifier
        self.c = 0
        self.model = model

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        self.c += 1


        # This model currently assumes that the features of the model are just the images.
        a = self.model.predict(image, batch_size=12)
        rospy.logwarn(a)
        if a > 0.9:
            return TrafficLight.GREEN
        else:
            return TrafficLight.RED
