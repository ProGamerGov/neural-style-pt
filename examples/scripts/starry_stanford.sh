# To run this script you'll need to download the ultra-high res
# scan of Starry Night from the Google Art Project, using this command:
# wget -c https://upload.wikimedia.org/wikipedia/commons/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg -O starry_night_gigapixel.jpg
# Or you can manually download the image from here: https://commons.wikimedia.org/wiki/File:Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg

STYLE_IMAGE=starry_night_gigapixel.jpg
CONTENT_IMAGE=examples/inputs/hoovertowernight.jpg

STYLE_WEIGHT=5e2
STYLE_SCALE=1.0

STYLE_WEIGHT2=2500 # Style weight for image size 2048 and above

PYTHON=python3 # Change to Python if using Python 2
SCRIPT=neural_style.py
GPU=0

NEURAL_STYLE=$PYTHON
NEURAL_STYLE+=" "
NEURAL_STYLE+=$SCRIPT

# Uncomment if using pip package
#NEURAL_STYLE=neural-style


$NEURAL_STYLE \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -style_scale $STYLE_SCALE \
  -print_iter 1 \
  -style_weight $STYLE_WEIGHT \
  -image_size 256 \
  -output_image out1.png \
  -tv_weight 0 \
  -gpu $GPU \
  -backend cudnn -cudnn_autotune

$NEURAL_STYLE \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -init image -init_image out1.png \
  -style_scale $STYLE_SCALE \
  -print_iter 1 \
  -style_weight $STYLE_WEIGHT \
  -image_size 512 \
  -num_iterations 500 \
  -output_image out2.png \
  -tv_weight 0 \
  -gpu $GPU \
  -backend cudnn -cudnn_autotune

$NEURAL_STYLE \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -init image -init_image out2.png \
  -style_scale $STYLE_SCALE \
  -print_iter 1 \
  -style_weight $STYLE_WEIGHT \
  -image_size 1024 \
  -num_iterations 200 \
  -output_image out3.png \
  -tv_weight 0 \
  -gpu $GPU \
  -backend cudnn -cudnn_autotune

$NEURAL_STYLE \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -init image -init_image out3.png \
  -style_scale $STYLE_SCALE \
  -print_iter 1 \
  -style_weight $STYLE_WEIGHT2 \
  -image_size 2048 \
  -num_iterations 200 \
  -output_image out4.png \
  -tv_weight 0 \
  -gpu $GPU \
  -backend cudnn

$NEURAL_STYLE \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -init image -init_image out4.png \
  -style_scale $STYLE_SCALE \
  -print_iter 1 \
  -style_weight $STYLE_WEIGHT2 \
  -image_size 2350 \
  -num_iterations 200 \
  -output_image out5.png \
  -tv_weight 0 \
  -gpu $GPU \
  -backend cudnn -optimizer adam
