import os, random
from os import listdir
from PIL import Image, ImageDraw

# transparent images folder directory
folder_dir = "/home/kai/SURF22/yolov5/data/"

# initialize count
i = 0

while i <= 1999:
    # pick a random spacecraft image
    random_spacecraft = random.choice(os.listdir(folder_dir + "prelim_data/sc_trans_random"))

    # open the image
    spacecraft = Image.open(folder_dir + "prelim_data/sc_trans_random/" + random_spacecraft)

    # make copy of image to create bounding box on 
    # want to preserve original image
    sc_copy = spacecraft.copy()

    # convert to alpha mode
    spacecraft_rgba = sc_copy.convert("RGBA")

    # randomly scale image
    scale_factor = random.randint(10 , 80) / 100
    w_scaled = int(spacecraft_rgba.width * scale_factor)
    h_scaled = int(spacecraft_rgba.height * scale_factor)
    scaled_image = spacecraft_rgba.resize((w_scaled, h_scaled), (Image.ANTIALIAS))

    # call getbbox() to retrieve bounding box coordinates (left, upper, right, lower)
    spacecraft_bbox = scaled_image.getbbox()

    # define points needed to draw bounding box
    x_0 = spacecraft_bbox[0]
    y_0 = spacecraft_bbox[1]
    x_1 = spacecraft_bbox[2]
    y_1 = spacecraft_bbox[3]

    '''
    # draw rectangle around spacecraft 
    draw_bbox = ImageDraw.Draw(scaled_image)
    draw_bbox.rectangle([x_0, y_0, x_1, y_1], fill=None, outline="red", width=5)
    '''

    # variables to hold image size
    # .size() outputs a tuple (width, height) in pixels
    img_coords = scaled_image.size
    img_w = img_coords[0]
    img_h = img_coords[1]

    # randomly overlay spacecraft and random Earth background

    # pick a random earth background
    random_earth = random.choice(os.listdir(folder_dir + "prelim_data/earth"))

    # open the image
    background = Image.open(folder_dir + "prelim_data/earth/" + random_earth)

    # make a copy of the background image to paste spacecraft image on
    # want to preserve original image
    background_copy = background.copy()

    # retrieve background image's size
    background_coords = background_copy.size
    background_w = background_coords[0]
    background_h = background_coords[1]

    # define range maximums
    x_max = abs(background_w - img_w)
    y_max = abs(background_h - img_h)

    # randomly choose point (x, y) to place the upper left-hand corner of spacecraft  
    # on the Earth background image
    x = random.randint(0, x_max)
    y = random.randint(0, y_max)

    # paste spacecraft image on top of background image 
    background_copy.paste(scaled_image, (x, y), mask=scaled_image)

    # use spacecraft and background size variables to normalize output of getbbox() 
    # for YOLOv5 .txt file
    w = (x_1 - x_0) / background_w
    h = (y_1 - y_0) / background_h
    x_center = (w / 2) / background_w
    print(x_center)
    y_center = (h / 2) / background_h   

    # split data into training (80%), testing (10%), and validation sets (10%)
    if i <= 1599:
        # save image to training folder
        background_copy.save(folder_dir + "images/train/bbox" + str(i) + ".png", "PNG")

        # create .txt file for bounded image with normalized variables
        # if we create more classes 0 will need to change
        with open(folder_dir + "labels/train/bbox" + str(i) + ".txt", 'w') as file:
            file.write('0 ' + str(x_center) + ' ' + str(w) + ' ' + str(y_center) + ' ' + str(h))
        
        # increase count
        i += 1
    elif i > 1599 and i <= 1799:
        # save image to testing folder
        background_copy.save(folder_dir + "images/test/bbox" + str(i) + ".png", "PNG")

        # create .txt file for bounded image with normalized variables
        # if we create more classes 0 will need to change
        with open(folder_dir + "labels/test/bbox" + str(i) + ".txt", 'w') as file:
            file.write('0 ' + str(x_center) + ' ' + str(w) + ' ' + str(y_center) + ' ' + str(h))
        
        # increase count
        i += 1
    elif i > 1799 and i <= 1999:
        # save image to validation folder
        background_copy.save(folder_dir + "images/validate/bbox" + str(i) + ".png", "PNG")

        # create .txt file for bounded image with normalized variables
        # if we create more classes 0 will need to change
        with open(folder_dir + "labels/validate/bbox" + str(i) + ".txt", 'w') as file:
            file.write('0 ' + str(x_center) + ' ' + str(w) + ' ' + str(y_center) + ' ' + str(h))
        
        # increase count
        i += 1