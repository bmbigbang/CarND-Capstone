from styx_msgs.msg import TrafficLight
import cv2


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.c = 0
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        self.c += 1
        # backtorgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('/home/student/classifier_images/image-{}.jpg'.format(self.c), image)
        return TrafficLight.GREEN
