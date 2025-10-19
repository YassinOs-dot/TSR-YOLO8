def getClassName(classNo):
    classes = [
        'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 
        'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
        'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
        'No Overtaking', 'No Overtaking for Heavy Vehicles', 'Right-of-Way at next Intersection',
        'Priority Road', 'Yield', 'Stop', 'No Vehicles', 'Heavy Vehicles Prohibited',
        'No Entry', 'General Caution', 'Dangerous Left Curve', 'Dangerous Right Curve',
        'Double Curve', 'Bumpy Road', 'Slippery Road', 'Narrowing Road', 'Road Work',
        'Traffic Signals', 'Pedestrian', 'Children', 'Bike', 'Snow', 'Deer',
        'End of Limits', 'Turn Right Ahead', 'Turn Left Ahead', 'Ahead Only',
        'Go Straight or Right', 'Go Straight or Left', 'Keep Right', 'Keep Left',
        'Roundabout Mandatory', 'End of No Overtaking', 'End of No Overtaking for Heavy Vehicles'
    ]
    return classes[classNo]
