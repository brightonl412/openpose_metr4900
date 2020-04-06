import sys

class Patient(object):
    """"Patient Object

    Stores patient values to be used for calculations.
    """
    def __init__(self, gender, height, mass):
        """Initialises patient object.
        
        Args:
            gender: Male or Female
            height: Height of patient in cm
            mass:   Mass of patient in kg
        """
        self.gender = gender
        self.height = height
        self.mass = mass
        self.pixel_cm = 0
        self.nose_offset = self.set_nose_offset()
    
    #Add edge cases i.e Male etc
    def set_nose_offset(self):
        """"Sets nose offset

        The nose offset is the distance between the top of the head to the nose 

        Returns: 
            int- nose offset
        """
        if self.gender == "male":
            return 10.77
        elif self.gender == "female":
            return 10.06
        else:
            print("Not a valid gender")
            sys.exit(-1)

    def body_perc(self):
        """"Sets body pecentages based on gender

        The percentages of body parts are set depending on gender. 

        Args:
            gender: str- must be "male" or "female"
        Returns: 
            dict- body mass percentages
        """
        if self.gender == "male":
            return {
                "head": 0.0694,
                "body": 0.3229,
                "pelvis": 0.1117,
                "arm": 0.0271,
                "forearm": 0.0162,
                "hand": 0.0061,
                "thigh": 0.1416,
                "shank": 0.0433,
                "foot": 0.0137
            }
        elif self.gender == "female":
            return {
                "head": 0.0668,
                "body": 0.301,
                "pelvis": 0.1247,
                "arm": 0.0255,
                "forearm": 0.0138,
                "hand": 0.0056,
                "thigh": 0.1478,
                "shank": 0.0481,
                "foot": 0.0129
            }
        else:
            print("Not a valid gender")
            sys.exit(-1)
    
    def set_pixel_cm(self, pixels):
        """"Calculates pixels per cm for patient depth

        pixel/cm is calculated by obtaining the pixel height between foot and 
        nose and then dividing by the height of the patients nose 
        (patient height - nose offset)

        Args:
            pixels: int- pixel height between foot and nose
        """
        current_pixel_cm = pixels / (self.height - self.nose_offset)
        self.pixel_cm = max(self.pixel_cm, current_pixel_cm)