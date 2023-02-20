import cv2 as open_cv
import numpy as np
import logging
from drawing_utils import draw_contours
from colors import COLOR_GREEN, COLOR_WHITE, COLOR_BLUE
import requests
from json import dump


class MotionDetector:
    LAPLACIAN = 1.4
    DETECT_DELAY = 1

    def __init__(self, video, coordinates, start_frame):
        self.video = video
        self.coordinates_data = coordinates
        self.start_frame = start_frame
        self.contours = []
        self.bounds = []
        self.mask = []

    def detect_motion(self):
        capture = open_cv.VideoCapture(self.video)
        capture.set(open_cv.CAP_PROP_POS_FRAMES, self.start_frame)

        coordinates_data = self.coordinates_data
        logging.debug("coordinates data: %s", coordinates_data)

        for p in coordinates_data:
            coordinates = self._coordinates(p)
            logging.debug("coordinates: %s", coordinates)

            rect = open_cv.boundingRect(coordinates)
            logging.debug("rect: %s", rect)

            new_coordinates = coordinates.copy()
            new_coordinates[:, 0] = coordinates[:, 0] - rect[0]
            new_coordinates[:, 1] = coordinates[:, 1] - rect[1]
            logging.debug("new_coordinates: %s", new_coordinates)

            self.contours.append(coordinates)
            self.bounds.append(rect)

            mask = open_cv.drawContours(
                np.zeros((rect[3], rect[2]), dtype=np.uint8),
                [new_coordinates],
                contourIdx=-1,
                color=255,
                thickness=-1,
                lineType=open_cv.LINE_8)

            mask = mask == 255
            self.mask.append(mask)
            logging.debug("mask: %s", self.mask)

        statuses = [False] * len(coordinates_data)
        times = [None] * len(coordinates_data)

        frame_number = 0
        free_spaces: int = 0
        while capture.isOpened():
            result, frame = capture.read()
            if frame is None:
                break

            if not result:
                raise CaptureReadError(
                    "Error reading video capture on frame %s" % str(frame))

            frame_number += 1
            blurred = open_cv.GaussianBlur(frame.copy(), (5, 5), 3)
            grayed = open_cv.cvtColor(blurred, open_cv.COLOR_BGR2GRAY)
            new_frame = frame.copy()
            logging.debug("new_frame: %s", new_frame)

            position_in_seconds = capture.get(
                open_cv.CAP_PROP_POS_MSEC) / 1000.0

            for index, c in enumerate(coordinates_data):
                status = self.__apply(grayed, index, c)

                if times[index] is not None and self.same_status(statuses, index, status):
                    times[index] = None
                    continue

                if times[index] is not None and self.status_changed(statuses, index, status):
                    if position_in_seconds - times[index] >= MotionDetector.DETECT_DELAY:
                        statuses[index] = status
                        times[index] = None
                    continue

                if times[index] is None and self.status_changed(statuses, index, status):
                    times[index] = position_in_seconds

            for index, p in enumerate(coordinates_data):
                coordinates = self._coordinates(p)

                color = COLOR_GREEN if statuses[index] else COLOR_BLUE
                draw_contours(new_frame, coordinates, str(
                    p["id"] + 1), COLOR_WHITE, color)

            open_cv.imshow(str(self.video), new_frame)

            # Wait 10 seconds and then print the number of empty spaces
            free_spaces_in_frame = len(statuses) - statuses.count(0)
            if free_spaces != free_spaces_in_frame:
                free_spaces = free_spaces_in_frame
                print(free_spaces, "spaces are empty")
                parking_value = free_spaces/len(statuses)
                json = {
                        "id": 1,
                        "name": "Henry Street #4",
                        "latitude": "52.663797090256100",
                        "longitude": "-8.628752240173640",
                        "ProbabilityParkingAvailable": parking_value,
                        #"free_spaces": free_spaces
                        }
                print(json)
                MotionDetector.send_my_put_request(1, json)
            k = open_cv.waitKey(1)
            if k == ord("q"):
                break
        capture.release()
        open_cv.destroyAllWindows()

    def __apply(self, grayed, index, p):
        coordinates = self._coordinates(p)
        logging.debug("points: %s", coordinates)

        rect = self.bounds[index]
        logging.debug("rect: %s", rect)

        roi_gray = grayed[rect[1]:(rect[1] + rect[3]),
                          rect[0]:(rect[0] + rect[2])]
        laplacian = open_cv.Laplacian(roi_gray, open_cv.CV_64F)
        logging.debug("laplacian: %s", laplacian)

        coordinates[:, 0] = coordinates[:, 0] - rect[0]
        coordinates[:, 1] = coordinates[:, 1] - rect[1]

        status = np.mean(
            np.abs(laplacian * self.mask[index])) < MotionDetector.LAPLACIAN
        logging.debug("status: %s", status)

        return status

    @staticmethod
    def _coordinates(p):
        return np.array(p["coordinates"])

    @staticmethod
    def same_status(coordinates_status, index, status):
        return status == coordinates_status[index]

    @staticmethod
    def status_changed(coordinates_status, index, status):
        return status != coordinates_status[index]

    @staticmethod
    def send_my_put_request(lot_monitor_id, json):
        token = "0f412f508358b8c1156d688d1db671e5ba4f1457"
        header = {"Authorization": "Token " + token,
                  "Content-Type": "application/json",
                  }
        url = f"http://127.0.0.1:8000/api-auth/parking-lot-monitors/{lot_monitor_id}/"
        MotionDetector.send_put_request(
            200, url,  header, json)

    @staticmethod
    def send_put_request(expected_status_code, url, headers, json):
        """
        """
        from requests.auth import HTTPBasicAuth
        basic_auth = HTTPBasicAuth('parkingMonitor', 'Letmein1$')
        
        
        response = requests.put(url, auth=basic_auth, headers=headers, json=json)
        #data = dump.dump_all(response)
        # print("\n\n--------------Request--------------------------\n",data.decode('utf-8'))
        assert response.status_code == expected_status_code


class CaptureReadError(Exception):
    pass
