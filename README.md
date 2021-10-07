# AGC - Nuvi_RecycleNet3

Effective trash detection model for AI Grand Challenge 2020 track 3 round 3.  
Nuvilab Solution.
The classes to be detected are:

- paper
- paper_pack
- steel
- glass
- PET
- plastic
- plasticbag


## Usage:

Installation

    cd mmvc
    MMCV_WITH_OPS=1 pip install -e .
    pip install mmdet==2.6.0



Run inference

Will return a dictionary with the following format:  

    {'Annotations': [{'Label': 'steel', 'Bbox': [X1, Y1, X2, Y2], 'Confidence': 0~1}]}

The bbox format follows: [xmin, ymin, xmax, ymax]
