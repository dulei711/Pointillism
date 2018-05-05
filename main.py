#  This code changes an image into pointillistic art. 
#  Details about each ellipse are written into a csv for the purpose of visualization in Tableau.
#
#  Code written by Matteo Ronchetti (https://matteo.ronchetti.xyz)
#  Small modifications made by Ken Flerlage to write the data to csv.
#

import cv2
import argparse
import math
import progressbar
from base64 import b16encode
from pointillism import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import *
 
# Prompt for a File.
def get_file():
    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select File",filetypes = (("JPEG Image","*.jpg"),("All Files","*.*")))
    root.withdraw()

    return root.filename 

# Convert RGB Colors to Hex
def convertRGBtoHex (Red, Green, Blue):
    triplet = (Red, Green, Blue)
    colorHex = b'#'+ b16encode(bytes(triplet))
    colorHex = str(colorHex)
    colorHex = colorHex[2:len(colorHex)-1]
    
    return colorHex;

parser = argparse.ArgumentParser(description='...')
parser.add_argument('--palette-size', default=20, type=int, help="Number of colors of the base palette")
parser.add_argument('--stroke-scale', default=0, help="Scale of the brush strokes (0 = automatic)")
parser.add_argument('--gradient-smoothing-radius', default=0, help="Radius of the smooth filter applied to the gradient (0 = automatic)")
parser.add_argument('--limit-image-size', default=0, help="Limit the image size (0 = no limits)")
parser.add_argument('img_path', nargs='?', default="images/desert.jpg")

args = parser.parse_args()

# Get the input file.
img_path = get_file()
if img_path == "":
    messagebox.showinfo("Error", "No file selected. Program will now quit.")
    sys.exit()

# Set names of the output files - image, data, and colors
res_path = img_path.rsplit(".", -1)[0] + "_drawing.jpg"
csv_path = img_path.rsplit(".", -1)[0] + "_data.csv"
color_path = img_path.rsplit(".", -1)[0] + "_color.txt"

#res_path = args.img_path.rsplit(".", -1)[0] + "_drawing.jpg"
#csv_path = args.img_path.rsplit(".", -1)[0] + "_data.csv"
#color_path = args.img_path.rsplit(".", -1)[0] + "_color.txt"

# Read the input image.
img = cv2.imread(img_path)

if args.limit_image_size > 0:
    img = limit_size(img, args.limit_image_size)

if args.stroke_scale == 0:
    stroke_scale = int(math.ceil(max(img.shape) / 1000))
    print("Automatically chosen stroke scale: %d" % stroke_scale)
else:
    stroke_scale = int(args.stroke_scale)

if args.gradient_smoothing_radius == 0:
    gradient_smoothing_radius = int(round(max(img.shape) / 50))
    print("Automatically chosen gradient smoothing radius: %d" % gradient_smoothing_radius)
else:
    gradient_smoothing_radius = args.stroke_scale

# Convert the image to grayscale to compute the gradient
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Computing color palette...")
palette = ColorPalette.from_image(img, args.palette_size)

print("Extending color palette...")
palette = palette.extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)])

# Now that we have our pallette, write the color file.
print("Writing the color palette file...")
out_color = open(color_path,'w') 
outString = '<color-palette name="Pointillism Palette" type="regular" >'
out_color.write (outString)
out_color.write('\n')

color_list = list()
for p in palette:
    color_hex = convertRGBtoHex(int(p[2]),int(p[1]),int(p[0]))
    color_list.append (color_hex)

color_list.sort()

for p in color_list:
    outString = '	<color>' + p + '</color>'
    out_color.write (outString)
    out_color.write('\n')

outString = '</color-palette>'
out_color.write (outString)
out_color.write('\n')
out_color.close()

# Display the color palette
cv2.imshow("palette", palette.to_image())
cv2.waitKey(200)

print("Computing gradient...")
gradient = VectorField.from_gradient(gray)

print("Smoothing gradient...")
gradient.smooth(gradient_smoothing_radius)

print("Drawing image...")

# Create a "cartonized" version of the image to use as a base for the painting
res = cv2.medianBlur(img, 11)

# Define a randomized grid of locations for the brush strokes
grid = randomized_grid(img.shape[0], img.shape[1], scale=3)
batch_size = 10000

out = open(csv_path,'w') 
outString = 'EllipseID,X,Y,Length,StrokeScale,Angle,color'
out.write (outString)
out.write('\n')

counter=1

bar = progressbar.ProgressBar()
for h in bar(range(0, len(grid), batch_size)):
    # Get the pixel colors at each point of the grid
    pixels = np.array([img[x[0], x[1]] for x in grid[h:min(h + batch_size, len(grid))]])

    # Precompute the probabilities for each color in the palette. Lower values of k means more randomness
    color_probabilities = compute_color_probabilities(pixels, palette, k=9)

    for i, (y, x) in enumerate(grid[h:min(h + batch_size, len(grid))]):
        color = color_select(color_probabilities[i], palette)

        # OpenCV uses BGR instead of RGB.
        color_hex = convertRGBtoHex(int(color[2]),int(color[1]),int(color[0]))

        angle = math.degrees(gradient.direction(y, x)) + 90
        length = int(round(stroke_scale + stroke_scale * math.sqrt(gradient.magnitude(y, x))))

        # draw the brush stroke
        color2=[color[0],color[1],color[2]]
        cv2.ellipse(res,  (x, y), (length, stroke_scale), angle, 0, 360, color, -1, cv2.LINE_AA)
     
        outString = str(counter) + ',' + str(x) + ',' + str(y) + ',' + str(length) + ',' + str(stroke_scale) + ',' + str(angle) + ',' + str(color_hex) 
        out.write (outString)
        out.write('\n')

        counter+=1

out.close()
cv2.imshow("res", limit_size(res, 1080))
cv2.imwrite(res_path, res)
messagebox.showinfo("Finished", "Data, Color Pallette, and Final Image file have been written to the same directory as the input file you selected.")
