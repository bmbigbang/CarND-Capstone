from styx_msgs.msg import TrafficLight
from matplotlib import pyplot

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
        pyplot.savefig('/home/student/classifier_images/image-{}'.format(self.c), dpi=None, facecolor='w', edgecolor='w',
                orientation='landscape', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)
        return TrafficLight.GREEN
